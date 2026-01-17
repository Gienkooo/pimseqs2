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

        // 3. Simple O(n) greedy round-robin assignment
        // For each sequence: try current DPU, if full move to next, if wave full start new wave
        struct DpuState {
            size_t total_bytes = 0;
            size_t seq_count = 0;
            std::vector<uint32_t> sequence_ids;
        };
        
        std::vector<std::vector<DpuState>> waves;
        waves.emplace_back(num_dpus);  // First wave
        
        size_t current_wave = 0;
        uint32_t current_dpu = 0;
        size_t assigned = 0;
        
        for (const auto& seq : seqs) {
            bool placed = false;
            
            // Try DPUs in current wave starting from current_dpu
            for (uint32_t attempts = 0; attempts < num_dpus && !placed; ++attempts) {
                DpuState& state = waves[current_wave][current_dpu];
                
                bool size_ok = (state.total_bytes + seq.estimated_size) <= mram_limit_bytes;
                bool count_ok = (state.seq_count + 1) <= max_seqs_per_dpu;
                
                if (size_ok && count_ok) {
                    state.sequence_ids.push_back(seq.db_key);
                    state.total_bytes += seq.estimated_size;
                    state.seq_count++;
                    placed = true;
                    assigned++;
                    // Round-robin to next DPU for load balance
                    current_dpu = (current_dpu + 1) % num_dpus;
                } else {
                    // This DPU can't fit it, try next
                    current_dpu = (current_dpu + 1) % num_dpus;
                }
            }
            
            if (!placed) {
                // All DPUs in current wave are full, start new wave
                waves.emplace_back(num_dpus);
                current_wave++;
                current_dpu = 0;
                
                // Place in first DPU of new wave
                DpuState& state = waves[current_wave][0];
                state.sequence_ids.push_back(seq.db_key);
                state.total_bytes += seq.estimated_size;
                state.seq_count++;
                assigned++;
                current_dpu = 1 % num_dpus;
            }
        }

        // 4. Build result (flatten waves, only include non-empty chunks)
        std::vector<std::vector<uint32_t>> result;
        result.reserve(waves.size() * num_dpus);
        
        size_t min_load = SIZE_MAX;
        size_t max_load = 0;
        uint32_t active_chunks = 0;
        size_t min_seqs = SIZE_MAX;
        size_t max_seqs = 0;
        
        for (auto& wave : waves) {
            for (auto& state : wave) {
                if (!state.sequence_ids.empty()) {
                    result.push_back(std::move(state.sequence_ids));
                    
                    if (state.total_bytes < min_load) min_load = state.total_bytes;
                    if (state.total_bytes > max_load) max_load = state.total_bytes;
                    if (state.seq_count < min_seqs) min_seqs = state.seq_count;
                    if (state.seq_count > max_seqs) max_seqs = state.seq_count;
                    active_chunks++;
                }
            }
        }
        
        if (active_chunks == 0) {
            min_load = 0;
            min_seqs = 0;
        }

        size_t num_waves_count = waves.size();

        // 6. Enhanced diagnostics
        Debug(Debug::INFO) << "[DPU] === Parallel Distribution Summary ===\n";
        Debug(Debug::INFO) << "[DPU]   Input: " << total_seqs << " sequences, " 
                           << total_db_bytes/1024 << "KB total\n";
        Debug(Debug::INFO) << "[DPU]   Seq lengths: [" << min_seq_len << ".." << max_seq_len << "]\n";
        Debug(Debug::INFO) << "[DPU]   Available DPUs: " << num_dpus << ", Active: " << active_chunks << "\n";
        Debug(Debug::INFO) << "[DPU]   Waves: " << num_waves_count << "\n";
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