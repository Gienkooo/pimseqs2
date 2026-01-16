#include "DpuCommunicationManager.h"
#include "DpuKernelManager.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include "Debug.h"

namespace mmseqs::dpu {

DpuCommunicationManager::DpuCommunicationManager(uint32_t num_dpus_requested)
  : num_dpus_available_(0),
    num_dpus_active_(0),
    async_in_progress_(false),
    is_simulator_(false),
    allocated_from_system_(false) {
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
      Debug(Debug::INFO) << "[DPU] No DPU hardware detected. Using simulator profile." << "\n";
    }
  } else {
    // Check if env var specifies simulator
    std::string env = getenv("UPMEM_PROFILE_BASE");
    if (env.find("simulator") != std::string::npos) {
      is_simulator_ = true;
    }
  }

  const std::string base_profile = is_simulator_ ? "backend=simulator" : (profile ? profile : "backend=hw");
  const uint32_t ranks_to_scan = 40; // max number of ranks in system
  const bool allocate_all = (num_dpus_requested == DPU_ALLOCATE_ALL);
  uint32_t remaining = allocate_all ? 0 : num_dpus_requested;

  Debug(Debug::INFO) << "[DPU] Allocating system (DPU_ALLOCATE_ALL) with profile: " << base_profile << "\n";

  dpu_error_t status = dpu_alloc(DPU_ALLOCATE_ALL, base_profile.c_str(), &system_set_);
  if (status != DPU_OK) {
    Debug(Debug::ERROR) << "[DPU ERROR] System alloc failed: " << dpu_error_to_string(status) << "\n";
    exit(EXIT_FAILURE);
  }
  allocated_from_system_ = true;

  const char* health_check_env = getenv("DPU_HEALTH_CHECK");
  bool do_health_check = health_check_env && std::strcmp(health_check_env, "1") == 0;
  double health_check_time = 0.0;

  // Diagnostic health check
  if (!is_simulator_ && do_health_check) {
    auto start = std::chrono::high_resolution_clock::now();
    std::string boot_kernel_path = DpuKernelManager::resolvePath(DpuKernelManager::KernelType::BOOT);
    status = dpu_load(system_set_, boot_kernel_path.c_str(), NULL);
    if (status != DPU_OK) {
      Debug(Debug::ERROR) << "[DPU ERROR] Failed to load diagnostic kernel '" << boot_kernel_path << "': " << dpu_error_to_string(status) << "\n";
      dpu_free(system_set_);
      exit(EXIT_FAILURE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    health_check_time += std::chrono::duration<double>(end - start).count();
  }

  struct dpu_set_t rank;
  DPU_RANK_FOREACH(system_set_, rank) {
    if (!allocate_all && num_dpus_active_ >= num_dpus_requested) break;

    bool healthy = true;
    if (!is_simulator_ && do_health_check) {
      auto start = std::chrono::high_resolution_clock::now();
      status = dpu_launch(rank, DPU_SYNCHRONOUS);
      auto end = std::chrono::high_resolution_clock::now();
      health_check_time += std::chrono::duration<double>(end - start).count();
      if (status != DPU_OK) {
        Debug(Debug::WARNING) << "[DPU] Warning: Rank failed diagnostics (" << dpu_error_to_string(status) << ") -> EXCLUDED\n";
        healthy = false;
      }
    }

    if (healthy) {
      rank_sets_.push_back(rank);
      struct dpu_set_t dpu;
      DPU_FOREACH(rank, dpu) {
        dpu_sets_.push_back(dpu);
        async_per_dpu_.push_back(false);
        ++num_dpus_active_;
        if (!allocate_all && num_dpus_active_ >= num_dpus_requested) break;
      }
    }
  }
  
  num_dpus_available_ = num_dpus_active_;

  if (num_dpus_requested != DPU_ALLOCATE_ALL && num_dpus_active_ < num_dpus_requested) {
    Debug(Debug::ERROR) << "[DPU ERROR] Unable to allocate requested DPUs (" << num_dpus_requested << " requested, " << num_dpus_active_ << " acquired)\n";
    exit(EXIT_FAILURE);
  }

  if (num_dpus_active_ == 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] No healthy DPUs found during incremental allocation." << "\n";
    exit(EXIT_FAILURE);
  }

  if (!is_simulator_ && do_health_check) {
    Debug(Debug::INFO) << "[DPU] Health check completed in " << health_check_time << " s\n";
  }

  Debug(Debug::INFO) << "[DPU] Allocated " << num_dpus_active_ << " DPUs across " << rank_sets_.size() << " ranks (Profile: " << base_profile << ")\n";
}

DpuCommunicationManager::~DpuCommunicationManager() {
  if (async_in_progress_) {
    waitForKernels();
  }
  if (allocated_from_system_) {
    dpu_free(system_set_);
  } else {
    for (auto &rank_set : rank_sets_) {
      dpu_free(rank_set);
    }
  }
}

std::vector<struct dpu_set_t> DpuCommunicationManager::getRankSets() {
  return rank_sets_;
}

void DpuCommunicationManager::broadcastData(
    const void* host_data, uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::Broadcast));
  
  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset/size not 8-byte aligned" << "\n";
    exit(EXIT_FAILURE);
  }

  for (auto &rank_set : rank_sets_) {
    dpu_error_t status = dpu_copy_to(rank_set, "__sys_used_mram_end", dpu_mram_offset,
                                     (void*)host_data, size_bytes);
    checkStatus(status, "Broadcast (rank)");
  }

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(size_bytes) * num_dpus_active_;
}

void DpuCommunicationManager::scatterDataToDPU(
    uint32_t dpu_id, const void* host_data,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::ScatterSingle));
  
  if (dpu_id >= num_dpus_active_) {
    Debug(Debug::ERROR) << "[DPU ERROR] DPU ID " << dpu_id << " out of range" << "\n";
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset/size not 8-byte aligned" << "\n";
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
    Debug(Debug::ERROR) << "[DPU ERROR] scatterDataParallel: data vector size mismatch" << "\n";
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset not 8-byte aligned" << "\n";
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

  size_t global_idx = 0;
  for (auto &rank_set : rank_sets_) {
    struct dpu_set_t dpu;
    DPU_FOREACH(rank_set, dpu) {
      if (global_idx >= per_dpu_data.size()) break;

      const auto &data = per_dpu_data[global_idx];
      if (data.size() < aligned_len) {
        temp_buffers.emplace_back(aligned_len, 0);
        if (!data.empty()) {
          std::memcpy(temp_buffers.back().data(), data.data(), data.size());
        }
        dpu_error_t status = dpu_prepare_xfer(dpu, temp_buffers.back().data());
        checkStatus(status, "Prepare xfer (padded)");
      } else {
        dpu_error_t status = dpu_prepare_xfer(dpu, (void*)data.data());
        checkStatus(status, "Prepare xfer");
      }

      ++global_idx;
    }

    dpu_error_t status = dpu_push_xfer(rank_set, DPU_XFER_TO_DPU, "__sys_used_mram_end",
                                       dpu_mram_offset, aligned_len, DPU_XFER_DEFAULT);
    checkStatus(status, "Push xfer (scatter rank)");
  }

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(aligned_len) * num_dpus_active_;
}

void DpuCommunicationManager::gatherDataFromDPU(
    uint32_t dpu_id, void* host_buffer,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::GatherSingle));
  
  if (dpu_id >= num_dpus_active_) {
    Debug(Debug::ERROR) << "[DPU ERROR] DPU ID " << dpu_id << " out of range" << "\n";
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset/size not 8-byte aligned" << "\n";
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
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset/size not 8-byte aligned" << "\n";
    exit(EXIT_FAILURE);
  }

  per_dpu_buffers.resize(num_dpus_active_);

  size_t global_idx = 0;
  for (auto &rank_set : rank_sets_) {
    struct dpu_set_t dpu;
    DPU_FOREACH(rank_set, dpu) {
      if (global_idx >= per_dpu_buffers.size()) break;
      per_dpu_buffers[global_idx].resize(size_per_dpu);
      dpu_error_t status = dpu_prepare_xfer(dpu, per_dpu_buffers[global_idx].data());
      checkStatus(status, "Prepare xfer (gather)");
      ++global_idx;
    }

    dpu_error_t status = dpu_push_xfer(rank_set, DPU_XFER_FROM_DPU, "__sys_used_mram_end",
                                       dpu_mram_offset, size_per_dpu, DPU_XFER_DEFAULT);
    checkStatus(status, "Push xfer (gather rank)");
  }

  if (timer.entry) timer.entry->bytes += static_cast<uint64_t>(size_per_dpu) * num_dpus_active_;
}

void DpuCommunicationManager::gatherDataParallelVariable(
    std::vector<std::vector<uint8_t>>& per_dpu_buffers,
    const std::vector<uint32_t>& sizes_per_dpu,
    uint32_t dpu_mram_offset) {
  ScopedTimer timer(slot(ProfileSlot::GatherParallel));
  
  if (dpu_mram_offset % MRAM_ALIGN != 0) {
    Debug(Debug::ERROR) << "[DPU ERROR] MRAM offset not 8-byte aligned" << "\n";
    exit(EXIT_FAILURE);
  }

  if (sizes_per_dpu.size() != num_dpus_active_) {
    Debug(Debug::ERROR) << "[DPU ERROR] gatherDataParallelVariable: sizes vector size mismatch" << "\n";
    exit(EXIT_FAILURE);
  }

  per_dpu_buffers.resize(num_dpus_active_);

  // Find max size per rank for efficient rank-level transfers
  // Group DPUs by rank and find max size within each rank
  size_t global_idx = 0;
  uint64_t total_bytes = 0;
  
  for (auto &rank_set : rank_sets_) {
    // Find max size in this rank
    uint32_t max_size_in_rank = 0;
    size_t rank_start = global_idx;
    
    struct dpu_set_t dpu;
    DPU_FOREACH(rank_set, dpu) {
      if (global_idx >= sizes_per_dpu.size()) break;
      uint32_t aligned_size = alignToMram(sizes_per_dpu[global_idx]);
      if (aligned_size > max_size_in_rank) max_size_in_rank = aligned_size;
      ++global_idx;
    }
    
    if (max_size_in_rank == 0) continue;
    
    // Prepare transfers for this rank using max size
    global_idx = rank_start;
    DPU_FOREACH(rank_set, dpu) {
      if (global_idx >= per_dpu_buffers.size()) break;
      per_dpu_buffers[global_idx].resize(max_size_in_rank);
      dpu_error_t status = dpu_prepare_xfer(dpu, per_dpu_buffers[global_idx].data());
      checkStatus(status, "Prepare xfer (variable gather)");
      ++global_idx;
    }

    dpu_error_t status = dpu_push_xfer(rank_set, DPU_XFER_FROM_DPU, "__sys_used_mram_end",
                                       dpu_mram_offset, max_size_in_rank, DPU_XFER_DEFAULT);
    checkStatus(status, "Push xfer (variable gather rank)");
    
    total_bytes += static_cast<uint64_t>(max_size_in_rank) * (global_idx - rank_start);
  }

  // Resize buffers to actual requested sizes (trim padding)
  for (size_t i = 0; i < num_dpus_active_; ++i) {
    uint32_t actual_size = alignToMram(sizes_per_dpu[i]);
    if (per_dpu_buffers[i].size() > actual_size) {
      per_dpu_buffers[i].resize(actual_size);
    }
  }

  if (timer.entry) timer.entry->bytes += total_bytes;
}

void DpuCommunicationManager::loadKernel(const char* kernel_binary_path) {
  ScopedTimer timer(slot(ProfileSlot::LoadKernel));
  if (async_in_progress_) {
    waitForKernels();
  }
  for (auto &rank_set : rank_sets_) {
    dpu_error_t status = dpu_load(rank_set, kernel_binary_path, NULL);
    checkStatus(status, "Kernel load (rank)");
  }
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
  // Launch all ranks in parallel (reference pattern: dpu_launch on rank set)
  // This is much faster than per-DPU launch as it uses rank-level parallelism
  for (auto& rank_set : rank_sets_) {
    dpu_error_t status = dpu_launch(rank_set, DPU_SYNCHRONOUS);
    if (status != DPU_OK) {
      readAndPrintLog();
    }
    checkStatus(status, "Kernel launch (rank)");
  }
}

void DpuCommunicationManager::executeKernelsAsync() {
  ScopedTimer timer(slot(ProfileSlot::LaunchAsync));
  if (async_in_progress_) {
    waitForKernels();
  }
  // Launch all ranks asynchronously in parallel (reference pattern)
  for (auto& rank_set : rank_sets_) {
    dpu_error_t status = dpu_launch(rank_set, DPU_ASYNCHRONOUS);
    checkStatus(status, "Async kernel launch (rank)");
  }
  // Mark all DPUs as in-flight
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
    async_per_dpu_[i] = true;
  }
  async_in_progress_ = true;
}

void DpuCommunicationManager::waitForKernels() {
  ScopedTimer timer(slot(ProfileSlot::WaitAsync));
  if (!async_in_progress_) return;
  // Sync all ranks in parallel (reference pattern: dpu_sync on rank set)
  for (auto& rank_set : rank_sets_) {
    dpu_error_t status = dpu_sync(rank_set);
    if (status != DPU_OK) {
      readAndPrintLog();
    }
    checkStatus(status, "Kernel sync (rank)");
  }
  // Clear all per-DPU flags
  for (uint32_t i = 0; i < num_dpus_active_; ++i) {
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
      Debug(Debug::ERROR) << "[DPU ERROR] DPU fault detected during async execution (dpu=" << i << ")\n";
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
    Debug(Debug::ERROR) << "[DPU ERROR] DPU fault detected during async execution (dpu=" << dpu_id << ")\n";
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
    Debug(Debug::ERROR) << "[DPU FATAL] " << context << " failed: " << dpu_error_to_string(status) << "\n";
    exit(EXIT_FAILURE);
  }
}

void DpuCommunicationManager::readAndPrintLog() {
  for (auto &dpu_set : dpu_sets_) {
    dpu_error_t status = dpu_log_read(dpu_set, stderr);
    if (status != DPU_OK) {
      Debug(Debug::WARNING) << "[DPU WARNING] Failed to read log from a DPU: " << dpu_error_to_string(status) << "\n";
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

  Debug(Debug::INFO) << "[DPU PROFILE] " << (tag ? tag : "comm") << "\n";
  for (size_t i = 0; i < static_cast<size_t>(ProfileSlot::Count); ++i) {
    const auto &e = profile_[i];
    if (e.count == 0) continue;
    double avg = e.total_ms / static_cast<double>(e.count);
    double mb = e.bytes / (1024.0 * 1024.0);
    Debug(Debug::INFO) << "  " << kNames[i] << " count=" << static_cast<unsigned long long>(e.count)
                       << " total=" << e.total_ms << " ms avg=" << avg << " ms max=" << e.max_ms
                       << " ms bytes=" << mb << " MB\n";
  }
}

}  // namespace mmseqs::dpu