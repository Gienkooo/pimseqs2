#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <dpu.h>
#ifdef __cplusplus
}
#endif
#include <cstdint>
#include <vector>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <array>

namespace mmseqs::dpu {

class DpuCommunicationManager {
 public:
  explicit DpuCommunicationManager(uint32_t num_dpus_requested);
  ~DpuCommunicationManager();

  uint32_t getNumDPUsActive() const { return num_dpus_active_; }
  uint32_t getNumDPUsAvailable() const { return num_dpus_available_; }

  void broadcastData(const void* host_data, uint32_t size_bytes,
                     uint32_t dpu_mram_offset);

  void scatterDataToDPU(uint32_t dpu_id, const void* host_data,
                        uint32_t size_bytes, uint32_t dpu_mram_offset);

  void gatherDataFromDPU(uint32_t dpu_id, void* host_buffer,
                         uint32_t size_bytes, uint32_t dpu_mram_offset);

  // Bulk parallel operations - more efficient than per-DPU scatter/gather
  // Each vector element corresponds to one DPU's data
  void scatterDataParallel(const std::vector<std::vector<uint8_t>>& per_dpu_data,
                           uint32_t dpu_mram_offset);
  
  void gatherDataParallel(std::vector<std::vector<uint8_t>>& per_dpu_buffers,
                          uint32_t size_per_dpu, uint32_t dpu_mram_offset);

  // Optimized gather that reads variable sizes per DPU using provided size vector
  // More efficient when DPUs have different result sizes (avoids wasted bandwidth)
  void gatherDataParallelVariable(std::vector<std::vector<uint8_t>>& per_dpu_buffers,
                                   const std::vector<uint32_t>& sizes_per_dpu,
                                   uint32_t dpu_mram_offset);

  void loadKernel(const char* kernel_binary_path);
  void executeKernels(); 
  void executeKernelsAsync();  // Non-blocking launch
  void waitForKernels();       // Wait for async execution to complete
  bool isExecutionComplete();  // Check if async execution finished (non-blocking)
  bool isAsyncInProgress() const { return async_in_progress_; }
  void readLogs();             // Read and print logs from all DPUs

  // Optional lightweight profiling (enable with env var DPU_PROFILE=1)
  void dumpProfile(const char* tag = nullptr) const;
  void resetProfile();
  bool isProfilingEnabled() const { return profile_enabled_; }

  // Per-DPU controls for fine-grained scheduling
  void loadKernel(uint32_t dpu_id, const char* kernel_binary_path);
  void executeKernel(uint32_t dpu_id);
  void executeKernelAsync(uint32_t dpu_id);
  void waitForKernel(uint32_t dpu_id);
  bool isExecutionComplete(uint32_t dpu_id);

  static constexpr uint32_t MRAM_SIZE = 64 * 1024 * 1024;  // 64 MB
  static constexpr uint32_t WRAM_SIZE = 64 * 1024;         // 64 KB
  static constexpr uint32_t MRAM_ALIGN = 8;                // 8-byte

  static inline uint32_t alignToMram(uint32_t size) {
    return ((size + MRAM_ALIGN - 1) / MRAM_ALIGN) * MRAM_ALIGN;
  }

  void readAndPrintLog();

  bool isSimulator() const { return is_simulator_; }

  // Access to raw DPU sets for advanced operations
  struct dpu_set_t& getDpuSet(uint32_t dpu_id) { return dpu_sets_.at(dpu_id); }
  std::vector<struct dpu_set_t> getRankSets();
  const std::vector<struct dpu_set_t>& getDpuSets() const { return dpu_sets_; }

  // ============== PUBLIC PROFILING INTERFACE ==============
  
  // All profile slots - DPU communication AND host processing
  enum class ProfileSlot {
    // DPU Communication
    Broadcast = 0,
    ScatterSingle,
    ScatterParallel,
    GatherSingle,
    GatherParallel,
    LoadKernel,
    LaunchSync,
    LaunchAsync,
    WaitSync,
    WaitAsync,
    // Host Processing
    HostBuildQueryBatch,    // Building query batches (PSSM generation)
    HostBuildTargetBatch,   // Assembling target data for DPUs
    HostProcessHits,        // processDpuHits() - the suspected bottleneck
    HostResultWrite,        // Writing results to disk
    HostDispatcherWait,     // Time spent in dispatcher polling loop
    HostTotalBatch,         // Total time per query batch (end-to-end)
    Count
  };

  struct ProfileEntry {
    uint64_t count = 0;
    uint64_t bytes = 0;
    uint64_t items = 0;      // For counting hits, queries, etc.
    double total_ms = 0.0;
    double max_ms = 0.0;
  };

  using Clock = std::chrono::steady_clock;

  // RAII timer - automatically records duration on destruction
  struct ScopedTimer {
    Clock::time_point start;
    ProfileEntry* entry;
    ScopedTimer(ProfileEntry* e) : start(Clock::now()), entry(e) {}
    ~ScopedTimer() {
      if (!entry) return;
      auto end = Clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      entry->count += 1;
      entry->total_ms += ms;
      if (ms > entry->max_ms) entry->max_ms = ms;
    }
    // Disable copy, enable move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&& other) noexcept : start(other.start), entry(other.entry) {
      other.entry = nullptr;  // Prevent double recording
    }
    ScopedTimer& operator=(ScopedTimer&&) = delete;
  };

  // Public access for host-side profiling
  inline ProfileEntry* getProfileSlot(ProfileSlot s) {
    return profile_enabled_ ? &profile_[static_cast<size_t>(s)] : nullptr;
  }
  
  // Convenience: create a scoped timer for a slot
  inline ScopedTimer timeSlot(ProfileSlot s) {
    return ScopedTimer(getProfileSlot(s));
  }
  
  // Record additional metrics (bytes, items) for a slot
  inline void recordSlotMetrics(ProfileSlot s, uint64_t bytes, uint64_t items = 0) {
    if (!profile_enabled_) return;
    auto& e = profile_[static_cast<size_t>(s)];
    e.bytes += bytes;
    e.items += items;
  }

 private:
  std::vector<struct dpu_set_t> rank_sets_; // one set per rank
  std::vector<struct dpu_set_t> dpu_sets_; // one set per DPU for independent control
  std::vector<bool> async_per_dpu_;
  uint32_t num_dpus_available_;
  uint32_t num_dpus_active_;
  bool async_in_progress_ = false;
  bool is_simulator_ = false;
  bool profile_enabled_ = false;
  bool allocated_from_system_ = false;
  struct dpu_set_t system_set_;

  std::array<ProfileEntry, static_cast<size_t>(ProfileSlot::Count)> profile_{};

  inline ProfileEntry* slot(ProfileSlot s) {
    return profile_enabled_ ? &profile_[static_cast<size_t>(s)] : nullptr;
  }

  void checkStatus(dpu_error_t status, const char* context);
};

}  // namespace mmseqs::dpu