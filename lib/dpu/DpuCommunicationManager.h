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

  void loadKernel(const char* kernel_binary_path);
  void executeKernels(); 
  void executeKernelsAsync();  // Non-blocking launch
  void waitForKernels();       // Wait for async execution to complete
  bool isExecutionComplete();  // Check if async execution finished (non-blocking)
  bool isAsyncInProgress() const { return async_in_progress_; }
  void readLogs();             // Read and print logs from all DPUs

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
  struct dpu_set_t& getDpuSet() { return dpu_set_; }
  struct dpu_set_t& getDpuSet(uint32_t dpu_id) { return dpu_sets_.at(dpu_id); }
  const std::vector<struct dpu_set_t>& getDpuSets() const { return dpu_sets_; }

 private:
  struct dpu_set_t dpu_set_;
  struct dpu_set_t rank_;
  std::vector<struct dpu_set_t> dpu_sets_; // one set per DPU for independent control
  std::vector<bool> async_per_dpu_;
  uint32_t num_dpus_available_;
  uint32_t num_dpus_active_;
  bool async_in_progress_ = false;
  bool is_simulator_ = false;

  void checkStatus(dpu_error_t status, const char* context);
};

}  // namespace mmseqs::dpu