#pragma once

#include "KmerGenerator.h"
#include "Indexer.h"
#include "DpuIndexBuilder.h"
#include "shared/DpuSharedTypes.h"
#include <vector>
#include <cstdint>

namespace mmseqs::dpu {

/**
 * DpuQueryPacketGenerator - Generates query k-mer packets for DPU streaming
 * 
 * For each query position:
 * 1. Extract k-mer
 * 2. Use KmerGenerator to generate ALL similar k-mers (10²-10⁴ variants)
 * 3. Create QueryPacket for each similar k-mer
 * 
 * This is where the "similar k-mer expansion" happens (query side only).
 */
class DpuQueryPacketGenerator {
public:
    /**
     * Generate query packets for a single query sequence
     * 
     * @param query_seq Encoded query sequence (amino acid indices)
     * @param kmer_size K-mer size
     * @param kmer_gen KmerGenerator instance (with threshold set)
     * @param indexer Indexer instance for k-mer encoding
     * @param use_spaced_kmers Whether to use spaced k-mers
     * @param spaced_pattern Spaced k-mer pattern (if applicable)
     * @param pattern_span Span of spaced pattern
     * @param take_only_best_kmer If true, only use exact k-mer (no similar expansion)
     * @return Vector of query packets ready for DPU transfer
     */
    static std::vector<KmerQueryPacket> generate(
        const std::vector<uint8_t>& query_seq,
        int kmer_size,
        KmerGenerator* kmer_gen,
        Indexer* indexer,
        bool use_spaced_kmers = false,
        const uint8_t* spaced_pattern = nullptr,
        int pattern_span = 0,
        bool take_only_best_kmer = false,
        const float* compositionBias = nullptr,
        int kmerThr = 0
    );
    
    /**
     * Get statistics about last generation
     */
    struct Stats {
        size_t num_query_positions;
        size_t total_similar_kmers;
        size_t total_packets;
        double avg_similar_per_position;
    };
    
    static Stats getLastStats() { return last_stats_; }

private:
    static Stats last_stats_;
};

} // namespace mmseqs::dpu
