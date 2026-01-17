#include "DpuDbSplitter.h"
#include "Debug.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <numeric> 

namespace mmseqs::dpu {

    std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabase(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    ) {
        size_t total_seqs = tdbr->getSize();
        
        // 1. Gather Metadata
        std::vector<SequenceMetadata> seqs;
        seqs.reserve(total_seqs);
        
        size_t total_db_bytes = 0;
        uint32_t min_seq_len = UINT32_MAX;
        uint32_t max_seq_len = 0;
        
        for (size_t i = 0; i < total_seqs; ++i) {
            uint32_t key = tdbr->getDbKey(i);
            uint32_t len = tdbr->getSeqLen(i);
            size_t size = estimateSequenceSizeBytes(len);
            
            if (size > mram_limit_bytes) {
                Debug(Debug::ERROR) << "[DPU] Sequence " << key << " is too large (" 
                                    << size/1024/1024 << "MB) for DPU MRAM limit (" 
                                    << mram_limit_bytes/1024/1024 << "MB)\n";
                return {};
            }
            
            seqs.push_back({key, len, size});
            total_db_bytes += size;
            if (len < min_seq_len) min_seq_len = len;
            if (len > max_seq_len) max_seq_len = len;
        }

        // Sort Descending 
        std::sort(seqs.begin(), seqs.end(), [](const SequenceMetadata& a, const SequenceMetadata& b) {
            return a.estimated_size > b.estimated_size;
        });

        // 2. Greedy Linear Packing
        std::vector<DpuChunk> chunks;
        if (!seqs.empty()) {
            chunks.emplace_back(); 
        }

        for (const auto& seq : seqs) {
            DpuChunk& current = chunks.back();
            
            bool size_ok = (current.current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
            bool count_ok = (current.current_seq_count + 1) <= max_seqs_per_dpu;

            if (size_ok && count_ok) {
                // Fits in current chunk
                current.sequence_ids.push_back(seq.db_key);
                current.current_seq_count++;
                current.current_estimated_bytes += seq.estimated_size;
                current.current_total_length += seq.length;
            } else {
                // Must start a new chunk
                chunks.emplace_back();
                DpuChunk& next = chunks.back();
                
                // We already validated that the seq fits in an empty chunk in step 1
                next.sequence_ids.push_back(seq.db_key);
                next.current_seq_count++;
                next.current_estimated_bytes += seq.estimated_size;
                next.current_total_length += seq.length;
            }
        }

        // 3. Convert to output format
        std::vector<std::vector<uint32_t>> result;
        result.reserve(chunks.size());
        
        size_t min_load = 0;
        size_t max_load = 0;
        if (!chunks.empty()) min_load = SIZE_MAX;
        
        for (const auto& chunk : chunks) {
            if (!chunk.sequence_ids.empty()) {
                result.push_back(chunk.sequence_ids);
                size_t bytes = chunk.current_estimated_bytes;
                if (bytes < min_load) min_load = bytes;
                if (bytes > max_load) max_load = bytes;
            }
        }
        
        size_t num_waves = 0;
        if (num_dpus > 0) {
            num_waves = (result.size() + num_dpus - 1) / num_dpus;
        }
        
        // Enhanced diagnostics
        Debug(Debug::INFO) << "[DPU] === Database Splitting Summary ===\n";
        Debug(Debug::INFO) << "[DPU]   Input: " << total_seqs << " sequences, " 
                           << total_db_bytes/1024 << "KB total\n";
        Debug(Debug::INFO) << "[DPU]   Seq lengths: [" << min_seq_len << ".." << max_seq_len << "]\n";
        Debug(Debug::INFO) << "[DPU]   Constraints: MRAM=" << mram_limit_bytes/1024/1024 
                           << "MB, max_seqs=" << max_seqs_per_dpu << "\n";
        Debug(Debug::INFO) << "[DPU]   Output: " << result.size() << " chunks across " 
                           << num_waves << " waves (for " << num_dpus << " DPUs)\n";
        Debug(Debug::INFO) << "[DPU]   Load Balance: Min=" << min_load/1024 
                           << "KB Max=" << max_load/1024 << "KB";
        if (max_load > 0) {
            double imbalance = 100.0 * (max_load - min_load) / max_load;
            Debug(Debug::INFO) << " (imbalance=" << imbalance << "%)";
        }
        Debug(Debug::INFO) << "\n";
        Debug(Debug::INFO) << "[DPU] ====================================\n";
        
        return result;
    }

    // ============================================================================
    // distributeForParallelism: LPT algorithm for maximum DPU utilization
    // Now supports MULTIPLE WAVES when database exceeds single-wave capacity
    // ============================================================================
    std::vector<std::vector<uint32_t>> DpuDbSplitter::distributeForParallelism(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    ) {
        size_t total_seqs = tdbr->getSize();
        if (total_seqs == 0 || num_dpus == 0) return {};

        // 1. Gather metadata with simpler size estimate (no k-mer index)
        std::vector<SequenceMetadata> seqs;
        seqs.reserve(total_seqs);
        
        size_t total_db_bytes = 0;
        uint32_t min_seq_len = UINT32_MAX;
        uint32_t max_seq_len = 0;
        
        for (size_t i = 0; i < total_seqs; ++i) {
            uint32_t key = tdbr->getDbKey(i);
            uint32_t len = tdbr->getSeqLen(i);
            size_t size = estimateSequenceSizeBytesSimple(len);
            
            if (size > mram_limit_bytes) {
                Debug(Debug::ERROR) << "[DPU] Sequence " << key << " is too large (" 
                                    << size/1024 << "KB) for DPU MRAM limit (" 
                                    << mram_limit_bytes/1024/1024 << "MB)\n";
                return {};
            }
            
            seqs.push_back({key, len, size});
            total_db_bytes += size;
            if (len < min_seq_len) min_seq_len = len;
            if (len > max_seq_len) max_seq_len = len;
        }

        // 2. Sort by length descending (LPT: Longest Processing Time first)
        std::sort(seqs.begin(), seqs.end(), [](const SequenceMetadata& a, const SequenceMetadata& b) {
            return a.length > b.length;
        });

        // 3. Use greedy bin-packing that creates new chunks when needed (like splitDatabase)
        // This allows multiple waves when database exceeds single-wave capacity
        std::vector<DpuChunk> chunks;
        chunks.reserve(num_dpus); // Start with one wave worth
        
        // Initialize first wave of chunks
        for (uint32_t d = 0; d < num_dpus; ++d) {
            chunks.emplace_back();
        }

        // 4. Assign sequences using modified LPT with multi-wave support
        size_t assigned = 0;
        
        for (const auto& seq : seqs) {
            bool placed = false;
            
            // Try to find a chunk that can fit this sequence
            // Prefer chunks in earlier waves (lower indices) for better load balance
            size_t best_chunk = SIZE_MAX;
            size_t best_load = SIZE_MAX;
            
            for (size_t c = 0; c < chunks.size(); ++c) {
                DpuChunk& chunk = chunks[c];
                bool size_ok = (chunk.current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
                bool count_ok = (chunk.current_seq_count + 1) <= max_seqs_per_dpu;
                
                if (size_ok && count_ok) {
                    // Prefer chunk with least load (LPT)
                    if (chunk.current_estimated_bytes < best_load) {
                        best_load = chunk.current_estimated_bytes;
                        best_chunk = c;
                    }
                }
            }
            
            if (best_chunk != SIZE_MAX) {
                // Found a suitable chunk
                DpuChunk& chunk = chunks[best_chunk];
                chunk.sequence_ids.push_back(seq.db_key);
                chunk.current_seq_count++;
                chunk.current_estimated_bytes += seq.estimated_size;
                chunk.current_total_length += seq.length;
                placed = true;
                assigned++;
            }
            
            if (!placed) {
                // All existing chunks are full - create a new wave of chunks
                size_t old_size = chunks.size();
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    chunks.emplace_back();
                }
                
                // Place in first chunk of new wave
                DpuChunk& new_chunk = chunks[old_size];
                new_chunk.sequence_ids.push_back(seq.db_key);
                new_chunk.current_seq_count++;
                new_chunk.current_estimated_bytes += seq.estimated_size;
                new_chunk.current_total_length += seq.length;
                assigned++;
            }
        }

        // 5. Build result (only include non-empty chunks)
        std::vector<std::vector<uint32_t>> result;
        result.reserve(chunks.size());
        
        size_t min_load = SIZE_MAX;
        size_t max_load = 0;
        uint32_t active_chunks = 0;
        size_t min_seqs = SIZE_MAX;
        size_t max_seqs = 0;
        
        for (auto& chunk : chunks) {
            if (!chunk.sequence_ids.empty()) {
                result.push_back(std::move(chunk.sequence_ids));
                size_t bytes = chunk.current_estimated_bytes;
                size_t count = chunk.current_seq_count;
                
                if (bytes < min_load) min_load = bytes;
                if (bytes > max_load) max_load = bytes;
                if (count < min_seqs) min_seqs = count;
                if (count > max_seqs) max_seqs = count;
                active_chunks++;
            }
        }
        
        if (active_chunks == 0) {
            min_load = 0;
            min_seqs = 0;
        }

        size_t num_waves = (result.size() + num_dpus - 1) / num_dpus;

        // 6. Enhanced diagnostics
        Debug(Debug::INFO) << "[DPU] === Parallel Distribution Summary ===\n";
        Debug(Debug::INFO) << "[DPU]   Input: " << total_seqs << " sequences, " 
                           << total_db_bytes/1024 << "KB total\n";
        Debug(Debug::INFO) << "[DPU]   Seq lengths: [" << min_seq_len << ".." << max_seq_len << "]\n";
        Debug(Debug::INFO) << "[DPU]   Available DPUs: " << num_dpus << ", Active: " << active_chunks << "\n";
        Debug(Debug::INFO) << "[DPU]   Waves: " << num_waves << "\n";
        Debug(Debug::INFO) << "[DPU]   Assigned: " << assigned << ", Skipped: " << (total_seqs - assigned) << "\n";
        Debug(Debug::INFO) << "[DPU]   Seqs/DPU: [" << min_seqs << ".." << max_seqs << "]\n";
        Debug(Debug::INFO) << "[DPU]   Load Balance: Min=" << min_load/1024 
                           << "KB Max=" << max_load/1024 << "KB";
        if (max_load > 0) {
            double imbalance = 100.0 * (max_load - min_load) / max_load;
            Debug(Debug::INFO) << " (imbalance=" << imbalance << "%)";
        }
        Debug(Debug::INFO) << "\n";
        Debug(Debug::INFO) << "[DPU] ========================================\n";
        
        return result;
    }

} // namespace mmseqs::dpu