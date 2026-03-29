#pragma once

#include "DpuCommunicationManager.h"
#include <cstdint>

namespace mmseqs::dpu {

// A lightweight, 8-byte aligned bump allocator for DPU MRAM.
// Safely partitions MRAM space sequentially for structures, preventing
// manually hardcoded offsets and overlap bugs.
class DpuMramAllocator {
public:
    DpuMramAllocator(uint32_t total_mram_size) 
        : total_capacity_(total_mram_size), current_offset_(0) {}

    // Allocate an aligned block of MRAM and return its start offset.
    // Returns UINT32_MAX if out of memory.
    uint32_t alloc(uint32_t size) {
        uint32_t aligned_size = DpuCommunicationManager::alignToMram(size);
        if (current_offset_ + aligned_size > total_capacity_) {
            return UINT32_MAX;
        }
        uint32_t offset = current_offset_;
        current_offset_ += aligned_size;
        return offset;
    }

    // Peek the next offset without allocating
    uint32_t peek() const {
        return current_offset_;
    }

    // Get total bytes allocated so far
    uint32_t getUsedBytes() const {
        return current_offset_;
    }

    // Get remaining free MRAM capacity
    uint32_t getRemainingCapacity() const {
        return total_capacity_ - current_offset_;
    }

    // Allocate the entire remaining aligned capacity to a dynamic buffer.
    uint32_t allocRemaining(uint32_t& out_actual_size) {
        uint32_t remaining = getRemainingCapacity();
        // Downward align to 8 bytes to be perfectly safe
        uint32_t aligned_remaining = (remaining / 8) * 8; 
        
        if (aligned_remaining == 0) {
            out_actual_size = 0;
            return UINT32_MAX;
        }
        
        out_actual_size = aligned_remaining;
        uint32_t offset = current_offset_;
        current_offset_ += aligned_remaining;
        return offset;
    }

private:
    uint32_t total_capacity_;
    uint32_t current_offset_;
};

} // namespace mmseqs::dpu
