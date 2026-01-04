#pragma once

#include "DpuCommunicationManager.h"
#include "shared/DpuSharedTypes.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace mmseqs::dpu {

/* DpuWorkflow: MRAM layout, broadcast/scatter and gather primitives. */
class DpuWorkflow {
public:
    explicit DpuWorkflow(DpuCommunicationManager& comm) : comm_(comm) {}

    // ----------------------------------------------------------------------
    // 1. Data Preparation Helpers
    // ----------------------------------------------------------------------
    
    struct MramLayout {
        uint32_t common_data_offset;    // PSSM/QueryMeta starts here
        uint32_t target_meta_offset;    // Target Meta starts here
        uint32_t target_data_offset;    // Target Sequences start here
        uint32_t results_offset;        // Results start here
        uint32_t results_capacity;      // Bytes allocated for results
        uint32_t total_mram_used;
    };

    // Calculate MRAM layout with 8-byte alignment.
    MramLayout calculateLayout(uint32_t descriptor_size,
                           uint32_t common_data_size,
                           uint32_t num_targets,
                           uint32_t num_queries,
                           uint32_t target_data_size,
                           uint32_t result_struct_size,
                           uint32_t extra_results_bytes = 0) 
    {
        MramLayout layout;
        
        // Descriptor (e.g. Kmer/Gapped/Ungapped) is at 0
        uint32_t bd_size = DpuCommunicationManager::alignToMram(descriptor_size);
        
        layout.common_data_offset = bd_size;
        
        uint32_t common_aligned = DpuCommunicationManager::alignToMram(common_data_size);
        layout.target_meta_offset = layout.common_data_offset + common_aligned;
        
        uint32_t tmeta_size = DpuCommunicationManager::alignToMram(num_targets * sizeof(TargetMetadata));
        layout.target_data_offset = layout.target_meta_offset + tmeta_size;
        
        uint32_t tdata_size = DpuCommunicationManager::alignToMram(target_data_size);
        layout.results_offset = layout.target_data_offset + tdata_size;
        
        // Allocate space for results: (Count:8bytes) + (Hits for all query/target pairs) + (Extra/Scratch)
        const uint64_t hits_bytes = static_cast<uint64_t>(num_targets) * static_cast<uint64_t>(num_queries) * static_cast<uint64_t>(result_struct_size);
        uint32_t hits_aligned = DpuCommunicationManager::alignToMram(static_cast<uint32_t>(hits_bytes + 64));
        uint32_t extra_aligned = DpuCommunicationManager::alignToMram(extra_results_bytes);
        uint32_t res_bytes = hits_aligned + extra_aligned;
        layout.results_capacity = DpuCommunicationManager::alignToMram(res_bytes);
        
        layout.total_mram_used = layout.results_offset + layout.results_capacity;
        
        return layout;
    }

    // ----------------------------------------------------------------------
    // 2. Communication Primitives
    // ----------------------------------------------------------------------

    // Broadcast common data (PSSM/Query Meta/Hash Table) to all DPUs.
    void broadcastCommon(const void* data, uint32_t size, uint32_t mram_offset) {
        if (size == 0) return;
        uint32_t aligned_size = DpuCommunicationManager::alignToMram(size);
        
        // If data is already aligned, use directly, else copy to buffer
        if (size == aligned_size) {
            comm_.broadcastData(data, aligned_size, mram_offset);
        } else {
            std::vector<uint8_t> buf(aligned_size, 0);
            memcpy(buf.data(), data, size);
            comm_.broadcastData(buf.data(), aligned_size, mram_offset);
        }
    }

    // Scatter batch to a specific DPU: Descriptor, TargetMeta, TargetData.
    template <typename BatchDescT>
    void scatterBatch(uint32_t dpu_id,
                      const BatchDescT& bd,
                      const std::vector<TargetMetadata>& t_meta,
                      const std::vector<uint8_t>& t_data,
                      const MramLayout& layout) 
    {
        // 1. Send Batch Descriptor (always at 0)
        uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(BatchDescT));
        // Ensure padded write if needed
        BatchDescT bd_copy = bd; // Copy to allow potential padding modification if needed
        comm_.scatterDataToDPU(dpu_id, &bd_copy, bd_size, 0);

        if (t_meta.empty()) return;

        // 2. Send Target Metadata
        uint32_t tmeta_bytes = t_meta.size() * sizeof(TargetMetadata);
        uint32_t tmeta_aligned = DpuCommunicationManager::alignToMram(tmeta_bytes);
        
        // Use temp buffer for alignment if strict (vector is usually contiguous, but size alignment matters)
        if (tmeta_bytes == tmeta_aligned) {
            comm_.scatterDataToDPU(dpu_id, t_meta.data(), tmeta_aligned, layout.target_meta_offset);
        } else {
             std::vector<uint8_t> buf(tmeta_aligned, 0);
             memcpy(buf.data(), t_meta.data(), tmeta_bytes);
             comm_.scatterDataToDPU(dpu_id, buf.data(), tmeta_aligned, layout.target_meta_offset);
        }

        // 3. Send Target Data
        uint32_t tdata_bytes = t_data.size();
        uint32_t tdata_aligned = DpuCommunicationManager::alignToMram(tdata_bytes);
        
        if (tdata_bytes == tdata_aligned) {
            comm_.scatterDataToDPU(dpu_id, t_data.data(), tdata_aligned, layout.target_data_offset);
        } else {
            // Target data assembled by `assembleTargetBatch` is usually already padded to 8 bytes,
            // but we check to be safe.
            std::vector<uint8_t> buf(tdata_aligned, 0);
            memcpy(buf.data(), t_data.data(), tdata_bytes);
            comm_.scatterDataToDPU(dpu_id, buf.data(), tdata_aligned, layout.target_data_offset);
        }
    }

    // Gather hits from DPU results (count at offset 0, hits at offset 8).
    template <typename HitType>
    std::vector<HitType> gatherResults(uint32_t dpu_id, uint32_t results_mram_offset) {
        // 1. Read Count (first 8 bytes)
        uint64_t count_buf = 0; // 8 bytes aligned
        comm_.gatherDataFromDPU(dpu_id, &count_buf, 8, results_mram_offset);
        
        // The kernel writes 32-bit count, 32-bit padding.
        // We interpret the first 32 bits as count.
        uint32_t hit_count = (uint32_t)count_buf;
        
        if (hit_count == 0) {
            return {};
        }

        // 2. Read Hits
        // Hits start immediately after the 8-byte count
        uint32_t hits_offset = results_mram_offset + 8;
        uint32_t data_size = hit_count * sizeof(HitType);
        uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);
        
        std::vector<HitType> hits(hit_count);
        
        // We gather slightly more (aligned), so we need a buffer if hits.data() isn't enough?
        // std::vector guarantees contiguous memory. DPU transfer writes `aligned_size`.
        // We must ensure the vector has capacity or write to temp buffer if alignment exceeds size significantly.
        // However, DPU gather writes to Host memory. It's safe to overwrite the end of vector *if* we resize it?
        // Safer: read to temporary buffer if aligned_size != data_size, or resize vector to aligned capacity.
        
        if (aligned_size > data_size) {
            std::vector<uint8_t> buf(aligned_size);
            comm_.gatherDataFromDPU(dpu_id, buf.data(), aligned_size, hits_offset);
            memcpy(hits.data(), buf.data(), data_size);
        } else {
            comm_.gatherDataFromDPU(dpu_id, hits.data(), aligned_size, hits_offset);
        }
        
        return hits;
    }

    // Gather with clamping to result capacity and export overflow flag (upper 32 bits of header)
    template <typename HitType>
    std::vector<HitType> gatherResultsClamped(uint32_t dpu_id,
                                              uint32_t results_mram_offset,
                                              uint32_t result_capacity_bytes,
                                              uint32_t* overflow_out) {
        uint64_t hdr = 0;
        comm_.gatherDataFromDPU(dpu_id, &hdr, 8, results_mram_offset);
        uint32_t hitcount = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
        uint32_t overflow = static_cast<uint32_t>(hdr >> 32);
        if (overflow_out) *overflow_out = overflow;

        if (result_capacity_bytes <= 8) return {};
        uint32_t maxHits = (result_capacity_bytes - 8) / sizeof(HitType);
        if (hitcount > maxHits) hitcount = maxHits;

        if (hitcount == 0) return {};

        uint32_t hits_offset = results_mram_offset + 8;
        uint32_t data_size = hitcount * sizeof(HitType);
        uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);

        std::vector<HitType> hits(hitcount);
        if (aligned_size != data_size) {
            std::vector<uint8_t> buf(aligned_size);
            comm_.gatherDataFromDPU(dpu_id, buf.data(), aligned_size, hits_offset);
            memcpy(hits.data(), buf.data(), data_size);
        } else {
            comm_.gatherDataFromDPU(dpu_id, hits.data(), aligned_size, hits_offset);
        }
        return hits;
    }

    // Scatter only the descriptor to DPU (leave targets in MRAM)
    template <typename BatchDescT>
    void scatterDescriptor(uint32_t dpu_id, const BatchDescT& bd) {
        uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(BatchDescT));
        BatchDescT bd_copy = bd;
        comm_.scatterDataToDPU(dpu_id, &bd_copy, bd_size, 0);
    }

    // Parallel Scatter Batch
    template <typename BatchDescT>
    void scatterBatchParallel(
        const std::vector<BatchDescT>& descriptors,
        const std::vector<std::vector<TargetMetadata>>& t_meta,
        const std::vector<std::vector<uint8_t>>& t_data,
        const MramLayout& layout) 
    {
        uint32_t num_dpus = descriptors.size();
        if (num_dpus == 0) return;

        // 1. Send Batch Descriptors
        std::vector<std::vector<uint8_t>> desc_bufs(num_dpus);
        uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(BatchDescT));
        for (uint32_t i = 0; i < num_dpus; ++i) {
            desc_bufs[i].resize(bd_size);
            memcpy(desc_bufs[i].data(), &descriptors[i], sizeof(BatchDescT));
        }
        comm_.scatterDataParallel(desc_bufs, 0);

        // 2. Send Target Metadata
        std::vector<std::vector<uint8_t>> meta_bufs(num_dpus);
        for (uint32_t i = 0; i < num_dpus; ++i) {
            if (!t_meta[i].empty()) {
                uint32_t tmeta_bytes = t_meta[i].size() * sizeof(TargetMetadata);
                meta_bufs[i].resize(tmeta_bytes); 
                memcpy(meta_bufs[i].data(), t_meta[i].data(), tmeta_bytes);
            }
        }
        comm_.scatterDataParallel(meta_bufs, layout.target_meta_offset);

        // 3. Send Target Data
        comm_.scatterDataParallel(t_data, layout.target_data_offset);
    }

    // Parallel Gather Results
    template <typename HitType>
    std::vector<std::vector<HitType>> gatherResultsParallel(
        uint32_t results_mram_offset,
        uint32_t result_capacity_bytes) 
    {
        uint32_t num_dpus = comm_.getNumDPUsActive();
        std::vector<std::vector<HitType>> all_hits(num_dpus);

        // 1. Gather Counts (8 bytes each)
        std::vector<std::vector<uint8_t>> count_bufs;
        comm_.gatherDataParallel(count_bufs, 8, results_mram_offset);

        uint32_t max_hits = 0;
        std::vector<uint32_t> hit_counts(num_dpus);

        for (uint32_t i = 0; i < num_dpus; ++i) {
            uint64_t hdr = 0;
            memcpy(&hdr, count_bufs[i].data(), 8);
            uint32_t count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
            
            if (result_capacity_bytes > 8) {
                uint32_t max_cap = (result_capacity_bytes - 8) / sizeof(HitType);
                if (count > max_cap) count = max_cap;
            } else {
                count = 0;
            }
            
            hit_counts[i] = count;
            if (count > max_hits) max_hits = count;
        }

        if (max_hits == 0) return all_hits;

        uint32_t transfer_size = max_hits * sizeof(HitType);
        uint32_t aligned_transfer = DpuCommunicationManager::alignToMram(transfer_size);
        
        std::vector<std::vector<uint8_t>> hit_bufs;
        comm_.gatherDataParallel(hit_bufs, aligned_transfer, results_mram_offset + 8);

        for (uint32_t i = 0; i < num_dpus; ++i) {
            if (hit_counts[i] > 0) {
                all_hits[i].resize(hit_counts[i]);
                memcpy(all_hits[i].data(), hit_bufs[i].data(), hit_counts[i] * sizeof(HitType));
            }
        }

        return all_hits;
    }

private:
    DpuCommunicationManager& comm_;
};

} // namespace mmseqs::dpu