#pragma once

#include "DBReader.h"
#include "Indexer.h"
#include "SubstitutionMatrix.h"
#include "shared/DpuSharedTypes.h"
#include <vector>
#include <cstdint>

namespace mmseqs::dpu {

/**
 * DpuIndexBuffer - Serialized bucketed hash index for DPU transfer
 * 
 * Structure (Bucketed Hash Map):
 * - buckets: Raw byte buffer containing KmerBucket structures (256 bytes each)
 *            First NUM_BUCKETS are primary, overflow buckets follow
 * - entries: Flat array of {target_id, position} pairs
 * - num_buckets: Total number of buckets (primary + overflow)
 */
struct DpuIndexBuffer {
    std::vector<uint8_t> buckets;               // Raw bucket data (KmerBucket array)
    std::vector<KmerCompactIndexEntry> entries; // {target_id, pos_j} pairs
    uint32_t num_buckets;                       // Total buckets (NUM_BUCKETS + overflow)
    
    DpuIndexBuffer() : num_buckets(0) {}
    
    size_t getTotalBytes() const {
        return buckets.size() + entries.size() * sizeof(KmerCompactIndexEntry);
    }
    
    size_t getBucketsBytes() const {
        return buckets.size();
    }
    
    size_t getEntriesBytes() const {
        return entries.size() * sizeof(KmerCompactIndexEntry);
    }
};

/**
 * DpuIndexBuilder - Builds bucketed hash index from database sequences
 */
class DpuIndexBuilder {
public:
    /**
     * Build bucketed hash index for a subset of database sequences
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
     * Compute bucket index from k-mer value using MurmurHash3
     */
    static inline uint32_t computeBucketIndex(uint32_t kmer_value) {
        return dpu_compute_hash(kmer_value);
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
