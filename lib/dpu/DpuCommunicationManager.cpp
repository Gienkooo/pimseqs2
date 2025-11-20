#include "DpuCommunicationManager.h"
#include <cstdio>
#include <cstdlib>

namespace mmseqs::dpu {

DpuCommunicationManager::DpuCommunicationManager(uint32_t num_dpus_requested)
    : num_dpus_available_(num_dpus_requested),
      num_dpus_active_(num_dpus_requested) {
  
  dpu_error_t status = dpu_alloc(num_dpus_requested, NULL, &dpu_set_);
  checkStatus(status, "DPU allocation");

  status = dpu_get_nr_ranks(dpu_set_, &num_dpus_available_);
  checkStatus(status, "Getting rank count");

  fprintf(stderr, "[DPU] Allocated %u DPUs\n", num_dpus_available_);
}

DpuCommunicationManager::~DpuCommunicationManager() {
  dpu_free(dpu_set_);
}

void DpuCommunicationManager::broadcastData(
    const void* host_data, uint32_t size_bytes, uint32_t dpu_mram_offset) {
  
  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set_, dpu) {
    dpu_error_t status = dpu_copy_to(dpu, "__sys_used_mram_end", dpu_mram_offset,
                                      (void*)host_data, size_bytes);
    checkStatus(status, "Broadcast");
  }
}

void DpuCommunicationManager::scatterDataToDPU(
    uint32_t dpu_id, const void* host_data,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  
  if (dpu_id >= num_dpus_active_) {
    fprintf(stderr, "[DPU ERROR] DPU ID %u out of range\n", dpu_id);
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  uint32_t idx = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set_, dpu) {
    if (idx == dpu_id) {
      dpu_error_t status = dpu_copy_to(dpu, "__sys_used_mram_end", dpu_mram_offset,
                                        (void*)host_data, size_bytes);
      checkStatus(status, "Scatter");
      return;
    }
    idx++;
  }
}

void DpuCommunicationManager::gatherDataFromDPU(
    uint32_t dpu_id, void* host_buffer,
    uint32_t size_bytes, uint32_t dpu_mram_offset) {
  
  if (dpu_id >= num_dpus_active_) {
    fprintf(stderr, "[DPU ERROR] DPU ID %u out of range\n", dpu_id);
    exit(EXIT_FAILURE);
  }

  if (dpu_mram_offset % MRAM_ALIGN != 0 || size_bytes % MRAM_ALIGN != 0) {
    fprintf(stderr, "[DPU ERROR] MRAM offset/size not 8-byte aligned\n");
    exit(EXIT_FAILURE);
  }

  uint32_t idx = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set_, dpu) {
    if (idx == dpu_id) {
      dpu_error_t status = dpu_copy_from(dpu, "__sys_used_mram_end", dpu_mram_offset,
                                          host_buffer, size_bytes);
      checkStatus(status, "Gather");
      return;
    }
    idx++;
  }
}

void DpuCommunicationManager::loadKernel(const char* kernel_binary_path) {
  dpu_error_t status = dpu_load(dpu_set_, kernel_binary_path, NULL);
  checkStatus(status, "Kernel load");
}

void DpuCommunicationManager::executeKernels() {
  dpu_error_t status = dpu_launch(dpu_set_, DPU_SYNCHRONOUS);
  checkStatus(status, "Kernel launch");
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
    } else {
        fprintf(stderr, "[DPU INFO] Successfully read log for a DPU\n");
    }
    fflush(stderr);
  }
}

}  // namespace mmseqs::dpu