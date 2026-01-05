#pragma once

#include "KmerGenerator.h"
#include "Indexer.h"
#include "DpuIndexBuilder.h"
#include "DBReader.h"
#include "BaseMatrix.h"
#include "SubstitutionMatrix.h"
#include "shared/DpuSharedTypes.h"
#include <vector>
#include <cstdint>
#include <utility>

namespace mmseqs::dpu {

/**
 * DpuQueryPacketGenerator - Streaming packet generator for DPU k-mer prefiltering
 * 
 * This class acts as a stateful iterator that generates query k-mer packets
 * on-demand directly into a fixed-size buffer.
 */
class DpuQueryPacketGenerator {
public:
    /**
     * Construct a streaming packet generator
     * 
     * @param qdbr Query database reader
     * @param kmer_gen KmerGenerator instance (with threshold set)
     * @param indexer Indexer instance for k-mer encoding
     * @param subMat Substitution matrix for encoding and bias correction
     * @param kmer_size Size of k-mers
     * @param use_spaced_kmers Whether to use spaced k-mers
     * @param spaced_pattern Spaced k-mer pattern (if applicable)
     * @param pattern_span Span of spaced pattern
     * @param take_only_best_kmer If true, only use exact k-mer (no similar expansion)
     * @param use_comp_bias Whether to apply composition bias correction
     * @param comp_bias_scale Scale factor for composition bias
     * @param kmerThr K-mer threshold for similar k-mer generation
     */
    DpuQueryPacketGenerator(
        DBReader<unsigned int>* qdbr,
        KmerGenerator* kmer_gen,
        Indexer* indexer,
        BaseMatrix* subMat,
        int kmer_size,
        bool use_spaced_kmers = false,
        const uint8_t* spaced_pattern = nullptr,
        int pattern_span = 0,
        bool take_only_best_kmer = false,
        bool use_comp_bias = false,
        float comp_bias_scale = 1.0f,
        int kmerThr = 0
    );
    
    /**
     * Fill buffer with packets until full or all queries exhausted
     * 
     * Packets for multiple queries are separated by sentinel packets (KMER_PACKET_SENTINEL_KEY). 
     * The DPU kernel uses these to reset its state table between queries.
     * 
     * @param buffer Pointer to DMA-aligned packet buffer
     * @param max_packets Maximum capacity of buffer (in packets)
     * @return Number of packets written (including sentinels)
     */
    size_t fillNextBatch(KmerQueryPacket* buffer, size_t max_packets);

    // Check if all queries have been processed
    bool isFinished() const;
    
    // Reset to process queries from the beginning
    void reset();

    // Get the query indices included in the last batch
    // Call after fillNextBatch() to know which queries are in the batch
    const std::vector<size_t>& getLastBatchQueryIndices() const { return last_batch_query_indices_; }
    
    // Get the number of complete queries in the last batch
    // Excludes any query that was split across batch boundary
    size_t getLastBatchCompleteQueryCount() const { return last_batch_complete_queries_; }
    
    struct Stats {
        size_t queries_started;         // Queries that began processing
        size_t queries_completed;       // Queries fully sent (including sentinel)
        size_t total_positions;         // Total k-mer positions processed
        size_t total_similar_kmers;     // Total similar k-mers generated (expansion)
        size_t total_packets;           // Total packets generated (excluding sentinels)
        size_t total_sentinels;         // Total sentinel packets generated
        size_t spillover_events;        // Times k-mer expansion spanned batches
    };
    
    Stats getStats() const { return stats_; }

private:
    DBReader<unsigned int>* qdbr_;
    KmerGenerator* kmer_gen_;
    Indexer* indexer_;
    BaseMatrix* subMat_;
    
    // Configuration
    int kmer_size_;
    bool use_spaced_kmers_;
    const uint8_t* spaced_pattern_;
    int pattern_span_;
    bool take_only_best_kmer_;
    bool use_comp_bias_;
    float comp_bias_scale_;
    int kmer_thr_;
    
    // Iteration State
    size_t current_query_idx_;
    size_t current_seq_pos_;
    bool current_query_loaded_;             // Has current query been loaded?
    bool current_query_sentinel_pending_;   // Need to emit sentinel for current query?
    
    // Current query data 
    std::vector<uint8_t> current_encoded_seq_;
    std::vector<float> current_comp_bias_;
    size_t current_num_positions_; 
    
    // Spillover State (when k-mer expansion spans batches)
    std::pair<size_t*, size_t> pending_kmer_list_;  // Ptr to KmerGenerator's internal buffer
    size_t pending_kmer_idx_;                       // Index into pending list (how many already written)
    bool has_pending_kmers_;                        // Do we have leftovers from previous call?
    uint16_t pending_query_pos_;                    // Query position for pending k-mers
    
    // Batch tracking
    std::vector<size_t> last_batch_query_indices_;
    size_t last_batch_complete_queries_;
    
    // Statistics
    Stats stats_;
    
    // Helper methods
    void loadCurrentQuery();
    bool advanceToNextQuery();
    size_t writePacketsFromKmerList(KmerQueryPacket* buffer, size_t max_packets,
                                     size_t* kmer_list, size_t list_size,
                                     size_t start_idx, uint16_t query_pos);
};

} // namespace mmseqs::dpu
