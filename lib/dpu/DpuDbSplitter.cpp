#include "DpuDbSplitter.h"
#include "shared/DpuSharedTypes.h" 
#include "Debug.h"
#include "DpuLog.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <atomic>

namespace mmseqs::dpu {

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

        seqs[i] = {key, len, marginal_size};
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

} // namespace mmseqs::dpu