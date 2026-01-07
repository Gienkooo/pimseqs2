#include "DpuCommunicationManager.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <stdexcept>

namespace mmseqs::dpu {

DpuCommunicationManager::DpuCommunicationManager(uint32_t num_dpus_requested)
  : num_dpus_available_(num_dpus_requested),
    num_dpus_active_(num_dpus_requested),
    async_in_progress_(false),
    is_simulator_(false) {
  const char* profile = nullptr;

  const char* prof_env = getenv("DPU_PROFILE");
  profile_enabled_ = prof_env && std::strcmp(prof_env, "0") != 0;

  // Smart backend detection to suppress warnings
  // If UPMEM_PROFILE_BASE is not set, check for hardware presence.
  if (!getenv("UPMEM_PROFILE_BASE")) {
    bool has_hardware = false;
    struct stat st;
    // Check if the DPU driver sysfs directory exists
    if (stat("/sys/class/dpu_rank", &st) == 0 && S_ISDIR(st.st_mode)) {
      has_hardware = true;
    }
      
    if (!has_hardware) {
      profile = "backend=simulator";
      is_simulator_ = true;
      fprintf(stderr, "[DPU] No DPU hardware detected. Using simulator profile.\n");
    }
  } else {
    // Check if env var specifies simulator
    std::string env = getenv("UPMEM_PROFILE_BASE");
    if (env.find("simulator") != std::string::npos) {
      is_simulator_ = true;
    }
  }

  // Allocate aggregate set, then derive per-DPU views from it.
  dpu_error_t status = dpu_alloc(num_dpus_requested, profile, &dpu_set_);
  checkStatus(status, "DPU allocation (aggregate)");

  status = dpu_get_nr_dpus(dpu_set_, &num_dpus_available_);
  checkStatus(status, "Getting rank count");

  num_dpus_active_ = num_dpus_available_;
  dpu_sets_.clear();
  async_per_dpu_.assign(num_dpus_active_, false);

  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set_, dpu) {
    dpu_sets_.push_back(dpu);
  }

  fprintf(stderr, "[DPU] Allocated %u DPUs (Profile: %s)\n", num_dpus_available_, profile ? profile : "default");
}

DpuCommunicationManager::~DpuCommunicationManager() {
  if (async_in_progress_) {
    waitForKernels();
  }
  dpu_free(dpu_set_);
}

std::vector<struct dpu_set_t> DpuCommunicationManager::getRankSets() {
  std::vector<struct dpu_set_t> ranks;
  struct dpu_set_t rank;

  DPU_RANK_FOREACH(dpu_set_, rank) {
    ranks.push_back(rank);
  }
  return ranks;
}

void DpuCommunicationManager::broadcastData(
    const void* host_data, uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::Broadcast));
  
  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  // Single call to aggregate set lets the driver broadcast to all DPUs efficiently.
  dpu_error_t status = dpu_copy_to(dpu_set_, "__sys_used_mram_end", dpu_mram_offset,
                                   (void*)host_data, size_bytes);
  checkStatus(status, "Broadcast (aggregate)");

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(size_bytes) * num_dpus_active_;
}

void DpuCommunicationManager::scatterDataToDPU(
    uint32_t dpu_id, const void* host_data,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::ScatterSingle));
  
  if (dpu_id >= num_dpus_active_) {
    fprintf(stderr, "[DPU ERROR] DPU ID %u out of range\n", dpu_id);
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  dpu_error_t status = dpu_copy_to(dpu_sets_[dpu_id], "__sys_used_mram_end", dpu_mram_offset,
                                    (void*)host_data, size_bytes);
  checkStatus(status, "Scatter");

  if (timer.entry) timer.entry->bytes += size_bytes;
}

void DpuCommunicationManager::scatterDataParallel(
    const std::vector<std::vector<uint8_t>>& per_dpu_data,
    uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::ScatterParallel));
  
  if (per_dpu_data.size() != num_dpus_active_) {
    fprintf(stderr, "[DPU ERROR] scatterDataParallel: data vector size mismatch\n");
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  // Find maximum size to determine transfer length
  size_t max_size = 0;
  for (const auto& data : per_dpu_data) {
    max_size = std::max(max_size, data.size());
  }
  
  if (max_size == 0) return;

  uint32_t aligned_len = alignToMram(static_cast<uint32_t>(max_size));

  // Store temporary padded buffers to keep them alive until push_xfer
  std::vector<std::vector<uint8_t>> temp_buffers;
  temp_buffers.reserve(num_dpus_active_);

  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_FOREACH(dpu_set_, dpu, each_dpu) {
    if (each_dpu >= per_dpu_data.size()) break;

    const auto& data = per_dpu_data[each_dpu];
    if (data.size() < aligned_len) {
      // Create padded copy
      temp_buffers.emplace_back(aligned_len, 0);
      if (!data.empty()) {
        std::memcpy(temp_buffers.back().data(), data.data(), data.size());
      }
      dpu_error_t status = dpu_prepare_xfer(dpu, temp_buffers.back().data());
      checkStatus(status, "Prepare xfer (padded)");
    } else {
      // Use existing buffer
      dpu_error_t status = dpu_prepare_xfer(dpu, (void*)data.data());
      checkStatus(status, "Prepare xfer");
    }
  }

  // Perform parallel transfer
  dpu_error_t status = dpu_push_xfer(dpu_set_, DPU_XFER_TO_DPU, "__sys_used_mram_end", 
                                     dpu_mram_offset, aligned_len, DPU_XFER_DEFAULT);
  checkStatus(status, "Push xfer (scatter)");

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(aligned_len) * num_dpus_active_;
}

void DpuCommunicationManager::gatherDataFromDPU(
    uint32_t dpu_id, void* host_buffer,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::GatherSingle));
  
  if (dpu_id >= num_dpus_active_) {
    fprintf(stderr, "[DPU ERROR] DPU ID %u out of range\n", dpu_id);
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  dpu_error_t status = dpu_copy_from(dpu_sets_[dpu_id], "__sys_used_mram_end", dpu_mram_offset,
                                      host_buffer, size_bytes);
  checkStatus(status, "Gather");

  if (timer.entry) timer.entry->bytes += size_bytes;
}

void DpuCommunicationManager::gatherDataParallel(
    std::vector<std::vector<uint8_t>>& per_dpu_buffers,
    uint32_t size_per_dpu, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::GatherParallel));
  
  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_per_dpu % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  per_dpu_buffers.resize(num_dpus_active_);
  
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_FOREACH(dpu_set_, dpu, each_dpu) {
    per_dpu_buffers[each_dpu].resize(size_per_dpu);
    dpu_error_t status = dpu_prepare_xfer(dpu, per_dpu_buffers[each_dpu].data());
    checkStatus(status, "Prepare xfer (gather)");
  }

  dpu_error_t status = dpu_push_xfer(dpu_set_, DPU_XFER_FROM_DPU, "__sys_used_mram_end", 
                                     dpu_mram_offset, size_per_dpu, DPU_XFER_DEFAULT);
  checkStatus(status, "Push xfer (gather)");

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(size_per_dpu) * num_dpus_active_;
}

void DpuCommunicationManager::loadKernel(const char* kernel_binary_path) {
  ScopedTimer timer(slot(ProfileSlot::LoadKernel));
  if (async_in_progress_) {
    waitForKernels();
  }
  dpu_error_t status = dpu_load(dpu_set_, kernel_binary_path, NULL);
  checkStatus(status, "Kernel load (aggregate)");
}

void DpuCommunicationManager::loadKernel(uint32_t dpu_id,
                                         const char* kernel_binary_path) {
  ScopedTimer timer(slot(ProfileSlot::LoadKernel));
  if (dpu_id >= num_dpus_active_) {
    throw std::runtime_error("loadKernel: invalid dpu_id");
  }
  if (async_per_dpu_[dpu_id]) {
    waitForKernel(dpu_id);
  }
  dpu_error_t status = dpu_load(dpu_sets_[dpu_id], kernel_binary_path, NULL);
  checkStatus(status, "Kernel load (single DPU)");
}

void DpuCommunicationManager::executeKernels() {
  ScopedTimer timer(slot(ProfileSlot::LaunchSync));
  if (async_in_progress_) {
    waitForKernels();
  }
  // Launch all per-DPU sets synchronously
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
    dpu_error_t status = dpu_launch(dpu_sets_[i], DPU_SYNCHRONOUS);
    if (status != DPU_OK) {
      readAndPrintLog();
    }
    checkStatus(status, "Kernel launch (per-DPU)");
  }
}

void DpuCommunicationManager::executeKernelsAsync() {
  ScopedTimer timer(slot(ProfileSlot::LaunchAsync));
  if (async_in_progress_) {
    waitForKernels();
  }
  // Launch all per-DPU sets asynchronously
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
    dpu_error_t status = dpu_launch(dpu_sets_[i], DPU_ASYNCHRONOUS);
    checkStatus(status, "Async kernel launch (per-DPU)");
    async_per_dpu_[i] = true;
  }
  async_in_progress_ = true;
}

void DpuCommunicationManager::waitForKernels() {
  ScopedTimer timer(slot(ProfileSlot::WaitAsync));
  if (!async_in_progress_) return;
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
    if (!async_per_dpu_[i]) continue;
    dpu_error_t status = dpu_sync(dpu_sets_[i]);
    if (status != DPU_OK) {
      readAndPrintLog();
    }
    checkStatus(status, "Kernel sync (per-DPU)");
    async_per_dpu_[i] = false;
  }
  async_in_progress_ = false;
}

void DpuCommunicationManager::executeKernel(uint32_t dpu_id) {
  ScopedTimer timer(slot(ProfileSlot::LaunchSync));
  if (dpu_id >= num_dpus_active_) {
    throw std::runtime_error("executeKernel: invalid dpu_id");
  }
  if (async_per_dpu_[dpu_id]) {
    waitForKernel(dpu_id);
  }
  dpu_error_t status = dpu_launch(dpu_sets_[dpu_id], DPU_SYNCHRONOUS);
  if (status != DPU_OK) {
    readAndPrintLog();
  }
  checkStatus(status, "Kernel launch (single DPU)");
}

void DpuCommunicationManager::executeKernelAsync(uint32_t dpu_id) {
  ScopedTimer timer(slot(ProfileSlot::LaunchAsync));
  if (dpu_id >= num_dpus_active_) {
    throw std::runtime_error("executeKernelAsync: invalid dpu_id");
  }
  dpu_error_t status = dpu_launch(dpu_sets_[dpu_id], DPU_ASYNCHRONOUS);
  checkStatus(status, "Async kernel launch (single DPU)");
  async_in_progress_ = true;
  async_per_dpu_[dpu_id] = true;
}

bool DpuCommunicationManager::isExecutionComplete() {
  if (!async_in_progress_) return true;
  bool all_done = true;
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
    if (!async_per_dpu_[i]) continue;
    bool done = false;
    bool fault = false;
    dpu_error_t status = dpu_status(dpu_sets_[i], &done, &fault);
    checkStatus(status, "DPU status check (per-DPU)");
    if (fault) {
      fprintf(stderr, "[DPU ERROR] DPU fault detected during async execution (dpu=%u)\n", i);
    }
    if (done) {
      async_per_dpu_[i] = false;
    } else {
      all_done = false;
    }
  }
  if (all_done) async_in_progress_ = false;
  return all_done;
}

void DpuCommunicationManager::waitForKernel(uint32_t dpu_id) {
  ScopedTimer timer(slot(ProfileSlot::WaitSync));
  if (dpu_id >= num_dpus_active_) {
    throw std::runtime_error("waitForKernel: invalid dpu_id");
  }
  if (!async_per_dpu_[dpu_id]) return;
  dpu_error_t status = dpu_sync(dpu_sets_[dpu_id]);
  if (status != DPU_OK) {
    readAndPrintLog();
  }
  checkStatus(status, "Kernel sync (single DPU)");
  async_per_dpu_[dpu_id] = false;
  async_in_progress_ = std::any_of(async_per_dpu_.begin(), async_per_dpu_.end(), [](bool v){ return v; });
}

bool DpuCommunicationManager::isExecutionComplete(uint32_t dpu_id) {
  if (dpu_id >= num_dpus_active_) {
    throw std::runtime_error("isExecutionComplete: invalid dpu_id");
  }
  if (!async_per_dpu_[dpu_id]) return true;
  bool done = false;
  bool fault = false;
  dpu_error_t status = dpu_status(dpu_sets_[dpu_id], &done, &fault);
  checkStatus(status, "DPU status check (single DPU)");
  if (fault) {
    fprintf(stderr, "[DPU ERROR] DPU fault detected during async execution (dpu=%u)\n", dpu_id);
  }
  if (done) {
    async_per_dpu_[dpu_id] = false;
    async_in_progress_ = std::any_of(async_per_dpu_.begin(), async_per_dpu_.end(), [](bool v){ return v; });
  }
  return done;
}

void DpuCommunicationManager::checkStatus(dpu_error_t status,
                                          const char* context) {
  if (status != DPU_OK) {
    fprintf(stderr, "[DPU FATAL] %s failed: %s\n", context,
            dpu_error_to_string(status));
    exit(EXIT_FAILURE);
  }
}

void DpuCommunicationManager::readAndPrintLog() {
  struct dpu_set_t dpu;

  DPU_FOREACH(dpu_set_, dpu) {
    dpu_error_t status = dpu_log_read(dpu, stderr);
    if (status != DPU_OK) {
        fprintf(stderr, "[DPU WARNING] Failed to read log from a DPU: %s\n", dpu_error_to_string(status));
    }
    fflush(stderr);
  }
}

void DpuCommunicationManager::readLogs() {
  readAndPrintLog();
}

void DpuCommunicationManager::resetProfile() {
  for (auto &e : profile_) {
    e = ProfileEntry{};
  }
}

void DpuCommunicationManager::dumpProfile(const char* tag) const {
  if (!profile_enabled_) return;
  static const char* kNames[] = {
    "broadcast",
    "scatter_single",
    "scatter_parallel",
    "gather_single",
    "gather_parallel",
    "load_kernel",
    "launch_sync",
    "launch_async",
    "wait_sync",
    "wait_async"
  };

  fprintf(stderr, "[DPU PROFILE] %s\n", tag ? tag : "comm");
  for (size_t i = 0; i < static_cast<size_t>(ProfileSlot::Count); ++i) {
    const auto &e = profile_[i];
    if (e.count == 0) continue;
    double avg = e.total_ms / static_cast<double>(e.count);
    double mb = e.bytes / (1024.0 * 1024.0);
    fprintf(stderr, "  %-17s count=%6llu total=%10.3f ms avg=%8.3f ms max=%8.3f ms bytes=%.2f MB\n",
            kNames[i],
            static_cast<unsigned long long>(e.count),
            e.total_ms,
            avg,
            e.max_ms,
            mb);
  }
}

}  // namespace mmseqs::dpu