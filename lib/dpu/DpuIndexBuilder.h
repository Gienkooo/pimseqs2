#pragma once

#include "DBReader.h"
#include "Indexer.h"
#include "SubstitutionMatrix.h"
#include "shared/DpuSharedTypes.h"
#include <vector>
#include <cstdint>

namespace mmseqs::dpu {

/**
 * DpuIndexBuffer - Serialized database index for DPU transfer
 * 
 * Structure:
 * - keys: Sorted array of unique k-mer values
 * - offsets: Start index in entries array for each key (size = keys.size() + 1)
 * - entries: Flat array of {target_id, position} pairs
 * - hints: Prefix lookup table (400 entries for 20x20 AA combinations)
 */
struct DpuIndexBuffer {
    std::vector<uint32_t> keys;                    // Sorted unique k-mers
    std::vector<uint32_t> offsets;                 // Offset into entries for each key
    std::vector<KmerCompactIndexEntry> entries;     // {target_id, pos_j} pairs
    std::vector<uint32_t> hints;                   // Hint table [401] (last is sentinel)
    
    size_t getTotalBytes() const {
        return keys.size() * sizeof(uint32_t) +
               offsets.size() * sizeof(uint32_t) +
               entries.size() * sizeof(KmerCompactIndexEntry) +
               hints.size() * sizeof(uint32_t);
    }
};

/**
 * DpuIndexBuilder - Builds compact sorted index from database sequences
 */
class DpuIndexBuilder {
public:
    /**
     * Build compact index for a subset of database sequences
     * 
     * @param tdbr Database reader
     * @param target_ids List of database IDs to include (will be remapped to local 0..N-1)
     * @param kmer_size K-mer size
     * @param subMat Substitution matrix (for amino acid encoding)
     * @param useSpacedKmers Whether to use spaced k-mers
     * @param spacedPattern Array of positions to sample (for spaced k-mers)
     * @param patternSpan Total span of the spaced pattern
     * @return Serialized index buffer ready for DPU transfer
     */
    static DpuIndexBuffer build(
        DBReader<unsigned int>* tdbr,
        const std::vector<uint32_t>& target_ids,
        int kmer_size,
        BaseMatrix* subMat,
        bool useSpacedKmers = false,
        const uint8_t* spacedPattern = nullptr,
        int patternSpan = 0
    );
    
    /**
     * Calculate hint index from k-mer value
     * Hint = top 2 amino acids = kmer / 20^(k-2)
     * 
     * @param kmer_value Encoded k-mer
     * @param kmer_size K-mer size
     * @return Hint index [0, 399]
     */
    static inline uint32_t calculateHintIndex(uint32_t kmer_value, int kmer_size) {
        if (kmer_size < 2) return 0;
        uint32_t divisor = 1;
        for (int i = 0; i < kmer_size - 2; ++i) {
            divisor *= 20;  
        }
        uint32_t hint = kmer_value / divisor;
        return (hint < HINT_TABLE_SIZE) ? hint : (HINT_TABLE_SIZE - 1);
    }

private:
    struct TempIndexEntry {
        uint32_t kmer;
        uint16_t local_id;
        uint16_t pos;
        
        bool operator<(const TempIndexEntry& other) const {
            return kmer < other.kmer;
        }
    };
};

} // namespace mmseqs::dpu
