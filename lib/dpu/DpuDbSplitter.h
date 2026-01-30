#pragma once

#include "DBReader.h"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace mmseqs::dpu {

struct DpuChunk {
    std::vector<uint32_t> sequence_ids;
    size_t current_seq_count = 0;
    size_t current_estimated_bytes = 0;
    size_t current_total_length = 0; 
};

class DpuDbSplitter {
public:
    static std::vector<std::vector<uint32_t>> splitDatabaseGreedyKmer(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    );

    static std::vector<std::vector<uint32_t>> splitDatabaseBalancedKmer(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    );

    // Original: Pack densely into MRAM (for kmer mode - data-resident)
    static std::vector<std::vector<uint32_t>> splitDatabase(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    );

    // New: Distribute across ALL DPUs for maximum parallelism (for ungapped/gapped)
    // Uses LPT (Longest Processing Time first) algorithm for load balancing
    static std::vector<std::vector<uint32_t>> distributeForParallelism(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    );

private:
    struct SequenceMetadata {
        uint32_t seq_idx;
        uint32_t db_key;
        uint32_t len;
        size_t estimated_size;
    };

    static std::vector<SequenceMetadata> getMetadata(
        DBReader<unsigned int>* tdbr, 
        size_t mram_limit_bytes
    );

    // Estimates Kmer size in MRAM (Index Entries only).
    // Estimates size in MRAM: Metadata + Sequence Data + Alignment overhead
    // For ungapped/gapped: simpler estimate (no k-mer index)
    static size_t estimateSequenceSizeBytes(uint32_t len) {
        return len * 10; 
    }

    // Simpler estimate for ungapped/gapped (no k-mer index)
    static size_t estimateSequenceSizeBytesSimple(uint32_t len) {
        // 16 bytes target metadata + sequence data (1 byte/residue) + 8-byte alignment
        return 16 + ((len + 7) & ~7u);
    }
};

} // namespace mmseqs::dpu