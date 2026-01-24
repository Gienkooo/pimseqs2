#include "DpuQueryPacketGenerator.h"
#include "Debug.h"
#include "DpuLog.h"

namespace mmseqs::dpu {

DpuQueryPacketGenerator::DpuQueryPacketGenerator(
    DBReader<unsigned int>* qdbr,
    KmerGenerator* kmer_gen,
    Indexer* indexer,
    BaseMatrix* subMat,
    int kmer_size,
    bool use_spaced_kmers,
    const uint8_t* spaced_pattern,
    int pattern_span,
    bool take_only_best_kmer,
    bool use_comp_bias,
    float comp_bias_scale,
    int kmerThr
) : qdbr_(qdbr),
    kmer_gen_(kmer_gen),
    indexer_(indexer),
    subMat_(subMat),
    kmer_size_(kmer_size),
    use_spaced_kmers_(use_spaced_kmers),
    spaced_pattern_(spaced_pattern),
    pattern_span_(pattern_span),
    take_only_best_kmer_(take_only_best_kmer),
    use_comp_bias_(use_comp_bias),
    comp_bias_scale_(comp_bias_scale),
    kmer_thr_(kmerThr),
    current_query_idx_(0),
    current_seq_pos_(0),
    current_query_loaded_(false),
    current_query_sentinel_pending_(false),
    current_num_positions_(0),
    pending_kmer_idx_(0),
    has_pending_kmers_(false),
    pending_query_pos_(0),
    last_batch_complete_queries_(0),
    stats_{0, 0, 0, 0, 0, 0, 0}
{
    pending_kmer_list_ = {nullptr, 0};
}

bool DpuQueryPacketGenerator::isFinished() const {
    // Finished when we've processed all queries AND have no pending sentinels
    return current_query_idx_ >= qdbr_->getSize() && 
           !current_query_sentinel_pending_ &&
           !has_pending_kmers_;
}

void DpuQueryPacketGenerator::reset() {
    current_query_idx_ = 0;
    current_seq_pos_ = 0;
    current_query_loaded_ = false;
    current_query_sentinel_pending_ = false;
    current_encoded_seq_.clear();
    current_comp_bias_.clear();
    current_num_positions_ = 0;
    pending_kmer_list_ = {nullptr, 0};
    pending_kmer_idx_ = 0;
    has_pending_kmers_ = false;
    pending_query_pos_ = 0;
    last_batch_query_indices_.clear();
    last_batch_complete_queries_ = 0;
    stats_ = {0, 0, 0, 0, 0, 0, 0};
}

void DpuQueryPacketGenerator::loadCurrentQuery() {
    if (current_query_loaded_ || current_query_idx_ >= qdbr_->getSize()) {
        return;
    }
    
    uint32_t queryLen = qdbr_->getSeqLen(current_query_idx_);
    const char* querySeq = qdbr_->getData(current_query_idx_, 0);
    
    // Encode query sequence
    current_encoded_seq_.resize(queryLen);
    for (size_t i = 0; i < queryLen; ++i) {
        unsigned char aa = static_cast<unsigned char>(querySeq[i]);
        current_encoded_seq_[i] = (subMat_->aa2num) ? subMat_->aa2num[aa] : 20;
        if (current_encoded_seq_[i] >= 21) current_encoded_seq_[i] = 20;
    }
    
    // Calculate composition bias if enabled
    if (use_comp_bias_) {
        current_comp_bias_.resize(queryLen);
        SubstitutionMatrix::calcLocalAaBiasCorrection(
            subMat_, current_encoded_seq_.data(), (int)queryLen,
            current_comp_bias_.data(), comp_bias_scale_
        );
    } else {
        current_comp_bias_.clear();
    }
    
    // Calculate number of k-mer positions
    int window_size = use_spaced_kmers_ ? pattern_span_ : kmer_size_;
    if (queryLen >= (size_t)window_size) {
        current_num_positions_ = queryLen - window_size + 1;
    } else {
        current_num_positions_ = 0;
        LOG_TRACE("Streamer: Query " << current_query_idx_ << " too short (" << queryLen << " < " << window_size << ")");
    }
    
    current_query_loaded_ = true;
    current_seq_pos_ = 0;
    stats_.queries_started++;
    
    LOG_TRACE("Streamer: Loaded query " << current_query_idx_ << " (len=" << queryLen << ", positions=" << current_num_positions_ << ")");
}

bool DpuQueryPacketGenerator::advanceToNextQuery() {
    current_query_idx_++;
    current_seq_pos_ = 0;
    current_query_loaded_ = false;
    current_query_sentinel_pending_ = false;
    current_encoded_seq_.clear();
    current_comp_bias_.clear();
    current_num_positions_ = 0;
    
    return current_query_idx_ < qdbr_->getSize();
}

size_t DpuQueryPacketGenerator::writePacketsFromKmerList(
    KmerQueryPacket* buffer, size_t max_packets,
    size_t* kmer_list, size_t list_size,
    size_t start_idx, uint16_t query_pos
) {
    size_t written = 0;
    
    for (size_t i = start_idx; i < list_size && written < max_packets; ++i) {
        uint32_t kmer_idx = static_cast<uint32_t>(kmer_list[i]);
        uint16_t bucket_idx = static_cast<uint16_t>(dpu_compute_hash(kmer_idx));
        
        buffer[written].kmer_idx = kmer_idx;
        buffer[written].bucket_idx = bucket_idx;
        buffer[written].query_pos = query_pos;
        written++;
    }
    
    return written;
}

size_t DpuQueryPacketGenerator::fillNextBatch(KmerQueryPacket* buffer, size_t max_packets) {
    size_t written = 0;
    last_batch_query_indices_.clear();
    last_batch_complete_queries_ = 0;
    
    // Track the query currently being processed if we are resuming state.
    if (current_query_loaded_ || has_pending_kmers_ || current_query_sentinel_pending_) {
        last_batch_query_indices_.push_back(current_query_idx_);
    }
    
    unsigned char kmer_buf[32];
    
    while (written < max_packets) {
        // 1. Handle pending sentinel from previous batch
        if (current_query_sentinel_pending_) {
            buffer[written].kmer_idx = KMER_PACKET_SENTINEL;
            buffer[written].bucket_idx = 0;
            buffer[written].query_pos = 0;
            written++;
            stats_.total_sentinels++;
            stats_.queries_completed++;
            last_batch_complete_queries_++;
            
            LOG_TRACE("Streamer: Wrote deferred sentinel for query " << current_query_idx_);
            
            if (!advanceToNextQuery()) {
                break;  // No more queries
            }
            continue;
        }
        
        // 2. Handle pending spillover (leftovers from previous batch) 
        if (has_pending_kmers_) {
            size_t remaining_in_list = pending_kmer_list_.second - pending_kmer_idx_;
            size_t available_space = max_packets - written;
            size_t to_write = std::min(available_space, remaining_in_list);
            
            size_t actually_written = writePacketsFromKmerList(
                buffer + written, to_write,
                pending_kmer_list_.first, pending_kmer_list_.second,
                pending_kmer_idx_, pending_query_pos_
            );
            
            written += actually_written;
            pending_kmer_idx_ += actually_written;
            stats_.total_packets += actually_written;
            
            if (pending_kmer_idx_ >= pending_kmer_list_.second) {
                // Done with spillover, advance position
                has_pending_kmers_ = false;
                current_seq_pos_++;
            } else {
                // Buffer full, more spillover remaining
                return written;
            }
        }
        
        // 3. Check if loading a query is needed
        if (!current_query_loaded_) {
            if (current_query_idx_ >= qdbr_->getSize()) {
                break;  
            }
            loadCurrentQuery();
            last_batch_query_indices_.push_back(current_query_idx_);
        }
        
        // 4. Process current query positions
        while (current_seq_pos_ < current_num_positions_ && written < max_packets) {
            // Extract k-mer at this position
            const unsigned char* kmer;
            bool contains_x = false;
            
            if (use_spaced_kmers_ && spaced_pattern_) {
                for (int j = 0; j < kmer_size_; ++j) {
                    unsigned char aa = current_encoded_seq_[current_seq_pos_ + spaced_pattern_[j]];
                    if (aa >= 20) { 
                        contains_x = true; 
                        break; 
                    } 
                    kmer_buf[j] = aa;
                }
                kmer = kmer_buf;
            } else {
                for (int j = 0; j < kmer_size_; ++j) {
                    if (current_encoded_seq_[current_seq_pos_ + j] >= 20) { 
                        contains_x = true; 
                        break; 
                    } 
                }
                kmer = current_encoded_seq_.data() + current_seq_pos_;
            }
            
            if (contains_x) {
                current_seq_pos_++;
                continue;
            }
            
            // Apply bias correction if enabled
            if (!current_comp_bias_.empty()) {
                float biasCorrection = 0.0f;
                if (use_spaced_kmers_ && spaced_pattern_) {
                    for (int i = 0; i < kmer_size_; ++i) {
                        biasCorrection += current_comp_bias_[current_seq_pos_ + spaced_pattern_[i]];
                    }
                } else {
                    for (int i = 0; i < kmer_size_; ++i) {
                        biasCorrection += current_comp_bias_[current_seq_pos_ + i];
                    }
                }
                short bias = static_cast<short>((biasCorrection < 0.0f) ? (biasCorrection - 0.5f) : (biasCorrection + 0.5f));
                short localThreshold = std::max(kmer_thr_ - bias, 0);
                kmer_gen_->setThreshold(localThreshold);
            }
            
            stats_.total_positions++;
            uint16_t query_pos = static_cast<uint16_t>(current_seq_pos_);
            
            if (take_only_best_kmer_) {
                // Exact k-mer matching only - always fits in 1 packet
                size_t exactKmer = indexer_->int2index(kmer, 0, kmer_size_);
                uint32_t kmer_idx = static_cast<uint32_t>(exactKmer);
                uint16_t bucket_idx = static_cast<uint16_t>(dpu_compute_hash(kmer_idx));
                
                buffer[written].kmer_idx = kmer_idx;
                buffer[written].bucket_idx = bucket_idx;
                buffer[written].query_pos = query_pos;
                written++;
                stats_.total_similar_kmers++;
                stats_.total_packets++;
                current_seq_pos_++;
            } else {
                // Similar k-mer matching: expand using substitution matrix
                std::pair<size_t*, size_t> similar_kmers = kmer_gen_->generateKmerList(kmer);
                stats_.total_similar_kmers += similar_kmers.second;
                
                size_t available_space = max_packets - written;
                
                if (similar_kmers.second <= available_space) {
                    // All k-mers fit in buffer
                    size_t actually_written = writePacketsFromKmerList(
                        buffer + written, similar_kmers.second,
                        similar_kmers.first, similar_kmers.second,
                        0, query_pos
                    );
                    written += actually_written;
                    stats_.total_packets += actually_written;
                    current_seq_pos_++;
                } else {
                    // SPILLOVER: k-mer expansion spans batch boundary
                    stats_.spillover_events++;
                    
                    // Write what we can
                    size_t actually_written = writePacketsFromKmerList(
                        buffer + written, available_space,
                        similar_kmers.first, similar_kmers.second,
                        0, query_pos
                    );
                    written += actually_written;
                    stats_.total_packets += actually_written;
                    
                    // Save state for next batch
                    pending_kmer_list_ = similar_kmers;
                    pending_kmer_idx_ = actually_written;
                    pending_query_pos_ = query_pos;
                    has_pending_kmers_ = true;
                    
                    LOG_TRACE("Streamer: Spillover: " << similar_kmers.second 
                              << " k-mers, wrote " << actually_written 
                              << ", pending " << (similar_kmers.second - actually_written));
                    
                    return written;  // Buffer full
                }
            }
        }
        
        // 5. End of query -> need to insert sentinel 
        if (current_seq_pos_ >= current_num_positions_ && !has_pending_kmers_) {
            if (written < max_packets) {
                // Room for sentinel
                buffer[written].kmer_idx = KMER_PACKET_SENTINEL;
                buffer[written].bucket_idx = 0;
                buffer[written].query_pos = 0;
                written++;
                stats_.total_sentinels++;
                stats_.queries_completed++;
                last_batch_complete_queries_++;
                
                LOG_TRACE("Streamer: Query " << current_query_idx_  << " complete, wrote sentinel");
                
                // Advance to next query
                if (!advanceToNextQuery()) {
                    break;  // No more queries
                }
            } else {
                // No room for sentinel - defer to next batch
                current_query_sentinel_pending_ = true;
                LOG_TRACE("Streamer: Query " << current_query_idx_ << " complete, sentinel deferred");
                return written;
            }
        }
    }
    
    return written;
}

} // namespace mmseqs::dpu
