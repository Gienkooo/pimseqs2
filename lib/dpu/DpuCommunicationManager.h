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

  void loadKernel(const char* kernel_binary_path);
  void executeKernels(); 

  static constexpr uint32_t MRAM_SIZE = 64 * 1024 * 1024;  // 64 MB
  static constexpr uint32_t WRAM_SIZE = 64 * 1024;         // 64 KB
  static constexpr uint32_t MRAM_ALIGN = 8;                // 8-byte

  static inline uint32_t alignToMram(uint32_t size) {
    return ((size + MRAM_ALIGN - 1) / MRAM_ALIGN) * MRAM_ALIGN;
  }

  void readAndPrintLog();

 private:
  struct dpu_set_t dpu_set_;
  struct dpu_set_t rank_;
  uint32_t num_dpus_available_;
  uint32_t num_dpus_active_;

  void checkStatus(dpu_error_t status, const char* context);
};

}  // namespace mmseqs::dpu