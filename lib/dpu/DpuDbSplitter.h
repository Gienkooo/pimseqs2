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

    static std::vector<std::vector<uint32_t>> splitDatabase(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    );

private:
    struct SequenceMetadata {
        uint32_t db_key;
        uint32_t len;
        size_t estimated_size;
    };

    // Helper moved inside the class to access private types
    static std::vector<SequenceMetadata> getMetadata(
        DBReader<unsigned int>* tdbr, 
        size_t mram_limit_bytes
    );

    // Estimates Kmer size in MRAM (Index Entries only).
    static size_t estimateSequenceSizeBytes(uint32_t len) {
        return len * 10; 
    }
};

} // namespace mmseqs::dpu