#include "DpuDbSplitter.h"
#include "shared/DpuSharedTypes.h" 
#include "Debug.h"
#include "DpuLog.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <atomic>
#include <numeric>
#include <tuple> 

namespace mmseqs::dpu {

// ============================================================================
// Helper: Metadata Gathering (From HEAD - uses OpenMP)
// ============================================================================
std::vector<DpuDbSplitter::SequenceMetadata> DpuDbSplitter::getMetadata(
    DBReader<unsigned int>* tdbr, 
    size_t mram_limit_bytes
) {
    size_t total_seqs = tdbr->getSize();
    std::vector<DpuDbSplitter::SequenceMetadata> seqs;
    seqs.resize(total_seqs); // Resize upfront for thread-safe indexed writes

    if (mram_limit_bytes <= DPU_FIXED_INDEX_OVERHEAD) {
        Debug(Debug::ERROR) << "[DPU] MRAM limit (" << mram_limit_bytes 
                            << ") is smaller than fixed overhead (" << DPU_FIXED_INDEX_OVERHEAD << ")\n";
        return {};
    }

    size_t available_for_entries = mram_limit_bytes - DPU_FIXED_INDEX_OVERHEAD;

    std::atomic<bool> too_large{false};
    std::atomic<uint32_t> too_large_key{0};

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_seqs; ++i) {
        uint32_t key = tdbr->getDbKey(i);
        uint32_t len = tdbr->getSeqLen(i);
        size_t marginal_size = estimateSequenceSizeBytes(len);

        if (marginal_size > available_for_entries) {
            too_large_key.store(key, std::memory_order_relaxed);
            too_large.store(true, std::memory_order_relaxed);
            continue;
        }

        seqs[i] = {static_cast<uint32_t>(i), key, len, marginal_size};
    }

    if (too_large.load(std::memory_order_relaxed)) {
        Debug(Debug::ERROR) << "[DPU] Sequence " << too_large_key.load() << " is too large.\n";
        return {};
    }

    // LPT Sort (Descending)
    std::sort(seqs.begin(), seqs.end(), [](const auto& a, const auto& b) {
        return a.estimated_size > b.estimated_size;
    });

    return seqs;
}

// ============================================================================
// Greedy Kmer Strategy (From HEAD)
// ============================================================================
std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabaseGreedyKmer(
    DBReader<unsigned int>* tdbr,
    uint32_t num_dpus,
    size_t mram_limit_bytes,
    uint32_t max_seqs_per_dpu
) {
    if (tdbr->getSize() == 0) return {};
    auto seqs = getMetadata(tdbr, mram_limit_bytes);
    if (seqs.empty()) return {};

    std::vector<DpuChunk> chunks;
    chunks.emplace_back(); 
    chunks.back().current_estimated_bytes = DPU_FIXED_INDEX_OVERHEAD;

    for (const auto& seq : seqs) {
        DpuChunk& current = chunks.back();
        bool size_ok = (current.current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
        bool count_ok = (current.current_seq_count + 1) <= max_seqs_per_dpu;

        if (size_ok && count_ok) {
            current.sequence_ids.push_back(seq.db_key);
            current.current_seq_count++;
            current.current_estimated_bytes += seq.estimated_size;
            current.current_total_length += seq.len;
        } else {
            chunks.emplace_back();
            DpuChunk& next = chunks.back();
            next.current_estimated_bytes = DPU_FIXED_INDEX_OVERHEAD;
            
            next.sequence_ids.push_back(seq.db_key);
            next.current_seq_count++;
            next.current_estimated_bytes += seq.estimated_size;
            next.current_total_length += seq.len;
        }
    }

    std::vector<std::vector<uint32_t>> result;
    result.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        if (!chunk.sequence_ids.empty()) result.push_back(chunk.sequence_ids);
    }
    
    LOG_TRACE("Greedy Split: " << result.size() << " chunks created.");
    return result;
}

// ============================================================================
// Balanced Kmer Strategy (From HEAD)
// ============================================================================
std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabaseBalancedKmer(
    DBReader<unsigned int>* tdbr,
    uint32_t num_dpus,
    size_t mram_limit_bytes,
    uint32_t max_seqs_per_dpu
) {
    if (tdbr->getSize() == 0) return {};
    auto seqs = getMetadata(tdbr, mram_limit_bytes);
    if (seqs.empty()) return {};

    size_t total_marginal_bytes = 0;
    for(const auto& s : seqs) total_marginal_bytes += s.estimated_size;

    size_t usable_mram = mram_limit_bytes - DPU_FIXED_INDEX_OVERHEAD;
    size_t min_chunks_seq = (seqs.size() + max_seqs_per_dpu - 1) / max_seqs_per_dpu;
    size_t min_chunks_ram = (total_marginal_bytes + usable_mram - 1) / usable_mram;
    size_t min_chunks = std::max(min_chunks_seq, min_chunks_ram);

    size_t num_waves = (min_chunks + num_dpus - 1) / num_dpus;
    if (num_waves == 0) num_waves = 1;

    LOG_TRACE("Balanced Splitter: Initial guess " << num_waves << " waves.");

    while (true) {
        size_t num_chunks = num_waves * num_dpus;
        auto cmp = [](const DpuChunk* a, const DpuChunk* b) {
            return a->current_estimated_bytes > b->current_estimated_bytes;
        };
        std::priority_queue<DpuChunk*, std::vector<DpuChunk*>, decltype(cmp)> pq(cmp);

        std::vector<DpuChunk> chunks(num_chunks);
        for (size_t i = 0; i < num_chunks; ++i) {
            chunks[i].current_estimated_bytes = DPU_FIXED_INDEX_OVERHEAD;
            pq.push(&chunks[i]);
        }

        bool fit_successful = true;
        for (const auto& seq : seqs) {
            DpuChunk* best_chunk = pq.top();
            pq.pop();

            bool size_ok = (best_chunk->current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
            bool count_ok = (best_chunk->current_seq_count + 1) <= max_seqs_per_dpu;

            if (size_ok && count_ok) {
                best_chunk->sequence_ids.push_back(seq.db_key);
                best_chunk->current_seq_count++;
                best_chunk->current_estimated_bytes += seq.estimated_size;
                best_chunk->current_total_length += seq.len;
                pq.push(best_chunk);
            } else {
                fit_successful = false;
                break;
            }
        }

        if (fit_successful) {
            std::vector<std::vector<uint32_t>> result;
            result.reserve(num_chunks);
            size_t min_load = SIZE_MAX, max_load = 0;

            for (const auto& chunk : chunks) {
                result.push_back(chunk.sequence_ids);
                if (!chunk.sequence_ids.empty()) {
                    min_load = std::min(min_load, chunk.current_estimated_bytes);
                    max_load = std::max(max_load, chunk.current_estimated_bytes);
                }
            }
            LOG_TRACE("Balanced Split: " << num_waves << " waves (" 
                      << result.size() << " chunks). Load (Min/Max): " 
                      << min_load/1024 << "KB / " << max_load/1024 << "KB");
            return result;
        }

        num_waves++;
        if (num_waves > seqs.size()) {
             Debug(Debug::ERROR) << "[DPU] Splitter failed to converge.\n";
             return {};
        }
    }
}

// ============================================================================
// Original Wrapper (From HEAD)
// ============================================================================
std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabase(
    DBReader<unsigned int>* tdbr,
    uint32_t num_dpus,
    size_t mram_limit_bytes,
    uint32_t max_seqs_per_dpu
) {
    // This function exists only to ensure compilation/linking for existing calls.
    // Forwards to the Balanced strategy to maintain legacy behavior.
    return splitDatabaseBalancedKmer(tdbr, num_dpus, mram_limit_bytes, max_seqs_per_dpu);
}

// ============================================================================
// distributeForParallelism: LPT algorithm for maximum DPU utilization (From MASTER)
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
    // Note: We duplicate metadata gathering here because the estimation function differs
    // from the kmer-based strategies.
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
        
        seqs.emplace_back(SequenceMetadata{static_cast<uint32_t>(i), key, len, size});
        total_db_bytes += size;
        if (len < min_seq_len) min_seq_len = len;
        if (len > max_seq_len) max_seq_len = len;
    }

    // 2. Sort by length descending (LPT: Longest Processing Time first)
    std::sort(seqs.begin(), seqs.end(), [](const SequenceMetadata& a, const SequenceMetadata& b) {
        return a.len > b.len; // Fixed: .length -> .len
    });

    // 3. Use heap-based LPT with multi-wave support
    // When all DPUs in current wave are full, create a new wave
    struct DpuState {
        size_t total_bytes = 0;      // MRAM bytes used
        uint64_t total_residues = 0; // Workload proxy (sum of target lengths)
        size_t seq_count = 0;
        std::vector<uint32_t> sequence_ids;
    };
    
    // Wave-based storage: each wave has num_dpus DPU states
    std::vector<std::vector<DpuState>> waves;
    waves.emplace_back(num_dpus);  // First wave
    
    // Min-heap tracks (total_residues, total_bytes, wave_idx, dpu_idx)
    // Prioritize the DPU with the fewest residues (workload), tie-break by bytes
    auto cmp = [](const std::tuple<uint64_t, size_t, uint32_t, uint32_t>& a,
                  const std::tuple<uint64_t, size_t, uint32_t, uint32_t>& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
        return std::get<1>(a) > std::get<1>(b);
    };
    std::priority_queue<std::tuple<uint64_t, size_t, uint32_t, uint32_t>,
                        std::vector<std::tuple<uint64_t, size_t, uint32_t, uint32_t>>,
                        decltype(cmp)> heap(cmp);
    
    // Initialize heap with first wave
    for (uint32_t d = 0; d < num_dpus; ++d) {
        heap.push({0ull, 0ull, 0u, d});
    }

    // 4. Assign sequences using heap-based LPT (O(n log n))
    size_t assigned = 0;
    
    for (const auto& seq : seqs) {
        bool placed = false;
        std::vector<std::tuple<uint64_t, size_t, uint32_t, uint32_t>> temp_storage;
        
        while (!heap.empty()) {
            auto [res_load, byte_load, wave_idx, dpu_idx] = heap.top();
            heap.pop();
            
            DpuState& state = waves[wave_idx][dpu_idx];
            // Reserve 8 bytes once for chunk-level padding/canary
            bool size_ok = (state.total_bytes + seq.estimated_size) <= (mram_limit_bytes > 8 ? (mram_limit_bytes - 8) : 0);
            bool count_ok = (state.seq_count + 1) <= max_seqs_per_dpu;
            
            if (size_ok && count_ok) {
                // Assign to this DPU
                state.sequence_ids.push_back(seq.seq_idx);
                state.total_bytes += seq.estimated_size;
                state.total_residues += seq.len; // Fixed: .length -> .len
                state.seq_count++;
                
                // Push back with updated workload and bytes
                heap.push({state.total_residues, state.total_bytes, wave_idx, dpu_idx});
                placed = true;
                assigned++;
                
                // Restore temporarily removed entries
                for (auto& p : temp_storage) heap.push(p);
                break;
            } else {
                // This DPU is full, try next
                temp_storage.push_back({res_load, byte_load, wave_idx, dpu_idx});
            }
        }
        
        if (!placed) {
            // All DPUs in all waves are full - create new wave
            uint32_t new_wave_idx = waves.size();
            waves.emplace_back(num_dpus);
            
            // Add new wave's DPUs to heap
            for (uint32_t d = 0; d < num_dpus; ++d) {
                heap.push({0ull, 0ull, new_wave_idx, d});
            }
            
            // Restore temporarily removed entries
            for (auto& p : temp_storage) {
                heap.push(p);
            }
            
            // Place in first DPU of new wave
            DpuState& state = waves[new_wave_idx][0];
            state.sequence_ids.push_back(seq.seq_idx);
            state.total_bytes += seq.estimated_size;
            state.total_residues += seq.len; // Fixed: .length -> .len
            state.seq_count++;
            
            // Update heap entry for this DPU
            heap.push({state.total_residues, state.total_bytes, new_wave_idx, 0});
            assigned++;
        }
    }

    // 5. Build result (flatten waves, only include non-empty chunks)
    std::vector<std::vector<uint32_t>> result;
    result.reserve(waves.size() * num_dpus);
    
    size_t min_load = SIZE_MAX;
    size_t max_load = 0;
    uint32_t active_chunks = 0;
    size_t min_seqs = SIZE_MAX;
    size_t max_seqs = 0;
    uint64_t min_residues = UINT64_MAX;
    uint64_t max_residues = 0;
    
    for (auto& wave : waves) {
        for (auto& state : wave) {
            if (!state.sequence_ids.empty()) {
                result.push_back(std::move(state.sequence_ids));
                
                if (state.total_bytes < min_load) min_load = state.total_bytes;
                if (state.total_bytes > max_load) max_load = state.total_bytes;
                if (state.seq_count < min_seqs) min_seqs = state.seq_count;
                if (state.seq_count > max_seqs) max_seqs = state.seq_count;
                if (state.total_residues < min_residues) min_residues = state.total_residues;
                if (state.total_residues > max_residues) max_residues = state.total_residues;
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
    Debug(Debug::INFO) << "[DPU]   Residues/DPU: [" << min_residues << ".." << max_residues << "]\n";
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