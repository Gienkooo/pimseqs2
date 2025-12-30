#include "DpuDbSplitter.h"
#include "Debug.h"
#include <algorithm>
#include <cmath>
#include <queue>

namespace mmseqs::dpu {

std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabase(
    DBReader<unsigned int>* tdbr,
    uint32_t num_dpus,
    size_t mram_limit_bytes,
    uint32_t max_seqs_per_dpu
) {
    size_t total_seqs = tdbr->getSize();
    size_t total_estimated_bytes = 0;
    
    // 1. Gather Metadata and Sort
    std::vector<SequenceMetadata> seqs;
    seqs.reserve(total_seqs);
    
    for (size_t i = 0; i < total_seqs; ++i) {
        uint32_t key = tdbr->getDbKey(i);
        uint32_t len = tdbr->getSeqLen(i);
        size_t size = estimateSequenceSizeBytes(len);
        
        // Safety check
        if (size > mram_limit_bytes) {
            Debug(Debug::ERROR) << "[DPU] Sequence " << key << " is too large (" 
                                << size/1024/1024 << "MB) for DPU MRAM limit (" 
                                << mram_limit_bytes/1024/1024 << "MB)\n";
            return {};
        }
        
        seqs.push_back({key, len, size});
        total_estimated_bytes += size;
    }

    // Sort Descending (Longest Processing Time first)
    std::sort(seqs.begin(), seqs.end(), [](const SequenceMetadata& a, const SequenceMetadata& b) {
        return a.estimated_size > b.estimated_size;
    });

    // 2. Calculate Theoretical Minimum Waves
    size_t min_chunks_by_seq = (total_seqs + max_seqs_per_dpu - 1) / max_seqs_per_dpu;
    size_t min_chunks_by_ram = (total_estimated_bytes + mram_limit_bytes - 1) / mram_limit_bytes;
    size_t min_chunks = std::max(min_chunks_by_seq, min_chunks_by_ram);

    // Round up to multiple of DPUs to minimize idle DPUs in the final wave
    size_t num_waves = (min_chunks + num_dpus - 1) / num_dpus;
    
    // We start with this optimal number of waves. 
    // If packing fails (fragmentation), we increment waves.
    while (true) {
        size_t num_chunks = num_waves * num_dpus;
        
        // Priority Queue to keep track of the "emptiest" chunk
        // Min-heap based on current_estimated_bytes
        auto cmp = [](const DpuChunk* a, const DpuChunk* b) {
            return a->current_estimated_bytes > b->current_estimated_bytes;
        };
        std::priority_queue<DpuChunk*, std::vector<DpuChunk*>, decltype(cmp)> pq(cmp);
        
        // Allocate chunks
        std::vector<DpuChunk> chunks(num_chunks);
        for (size_t i = 0; i < num_chunks; ++i) {
            chunks[i].sequence_ids.reserve(total_seqs / num_chunks); // heuristic reserve
            pq.push(&chunks[i]);
        }

        bool fit_successful = true;

        // 3. Distribute Sequences
        for (const auto& seq : seqs) {
            // Get the least loaded chunk
            DpuChunk* best_chunk = pq.top();
            pq.pop();

            // Check Hard Constraints
            bool size_ok = (best_chunk->current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
            bool count_ok = (best_chunk->current_seq_count + 1) <= max_seqs_per_dpu;

            if (size_ok && count_ok) {
                // Add to chunk
                best_chunk->sequence_ids.push_back(seq.db_key);
                best_chunk->current_seq_count++;
                best_chunk->current_estimated_bytes += seq.estimated_size;
                best_chunk->current_total_length += seq.length;
                pq.push(best_chunk);
            } else {
                // The least loaded chunk cannot fit this sequence.
                // This implies NO chunk can fit it (since we picked the emptiest).
                // We need more capacity (more waves).
                fit_successful = false;
                break;
            }
        }

        if (fit_successful) {
            // Convert to output format
            std::vector<std::vector<uint32_t>> result;
            result.reserve(num_chunks);
            
            size_t min_load = SIZE_MAX, max_load = 0;
            
            for (const auto& chunk : chunks) {
                // We filter out completely empty chunks if we allocated too many
                // (Though usually we want to keep them to maintain wave alignment, 
                // but empty chunks in a wave just mean idle DPUs, which is unavoidable if total data is small)
                if (!chunk.sequence_ids.empty()) {
                    result.push_back(chunk.sequence_ids);
                    min_load = std::min(min_load, chunk.current_estimated_bytes);
                    max_load = std::max(max_load, chunk.current_estimated_bytes);
                }
            }
            
            Debug(Debug::INFO) << "[DPU] Database split into " << result.size() 
                               << " chunks (" << num_waves << " waves)." 
                               << " Load Balance (Bytes): Min=" << min_load/1024 
                               << "KB Max=" << max_load/1024 << "KB\n";
            return result;
        }

        // 4. Retry Logic 
        // If we failed, it means fragmentation prevented perfect packing.
        // Add exactly one wave of capacity.
        num_waves++;
        Debug(Debug::INFO) << "[DPU] Packing constraints triggered. Increasing to " << num_waves << " waves...\n";
    }
}

} // namespace mmseqs::dpu