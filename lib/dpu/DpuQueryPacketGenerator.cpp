#include "DpuQueryPacketGenerator.h"
#include "Debug.h"

#ifdef DPU_DEBUG_MODE
  #define DPU_DEBUG_LOG Debug(Debug::INFO)
#else
  #define DPU_DEBUG_LOG if (false) Debug(Debug::INFO)
#endif

namespace mmseqs::dpu {

DpuQueryPacketGenerator::Stats DpuQueryPacketGenerator::last_stats_ = {0, 0, 0, 0.0};

std::vector<KmerQueryPacket> DpuQueryPacketGenerator::generate(
    const std::vector<uint8_t>& query_seq,
    int kmer_size,
    KmerGenerator* kmer_gen,
    Indexer* indexer,
    bool use_spaced_kmers,
    const uint8_t* spaced_pattern,
    int pattern_span,
    bool take_only_best_kmer,
    const float* compositionBias,
    int kmerThr
) {
    std::vector<KmerQueryPacket> packets;
    
    int window_size = use_spaced_kmers ? pattern_span : kmer_size;
    
    if (query_seq.size() < (size_t)window_size) {
        Debug(Debug::WARNING) << "[DPU] QueryPacketGen: Query too short (" << query_seq.size() << " < " << window_size << ")\\n";
        return packets;
    }
    
    // Reset stats
    last_stats_ = {0, 0, 0, 0.0};
    
    unsigned char kmer_buf[32];
    size_t num_positions = query_seq.size() - window_size + 1;
    
    // Estimate capacity - much smaller if using exact match only
    if (take_only_best_kmer) {
        packets.reserve(num_positions);  // 1 k-mer per position
    } else {
        packets.reserve(num_positions * 1000);  // ~1000 similar k-mers per position
    }
    
    for (size_t pos = 0; pos < num_positions; ++pos) {
        // Extract k-mer at this position
        const unsigned char* kmer;
        bool contains_x = false; // Check for invalid residues

        if (use_spaced_kmers && spaced_pattern) {
            for (int j = 0; j < kmer_size; ++j) {
                unsigned char aa = query_seq[pos + spaced_pattern[j]];
                if (aa >= 20) { 
                    contains_x = true; 
                    break; 
                } 
                kmer_buf[j] = aa;
            }
            kmer = kmer_buf;
        } else {
            for (int j = 0; j < kmer_size; ++j) {
                if (query_seq[pos + j] >= 20) { 
                    contains_x = true; 
                    break; 
                } 
            }
            kmer = query_seq.data() + pos;
        }

        // Skip packet generation if k-mer contains X
        if (contains_x) continue;

        // ----- BIAS CORRECTION & LOCAL THRESHOLD -----
        if (compositionBias != nullptr) {
            float biasCorrection = 0.0f;
            if (use_spaced_kmers && spaced_pattern) {
                for (int i = 0; i < kmer_size; ++i) {
                    biasCorrection += compositionBias[pos + spaced_pattern[i]];
                }
            } else {
                for (int i = 0; i < kmer_size; ++i) {
                    biasCorrection += compositionBias[pos + i];
                }
            }
            short bias = static_cast<short>((biasCorrection < 0.0f) ? (biasCorrection - 0.5f) : (biasCorrection + 0.5f));
            short localThreshold = std::max(kmerThr - bias, 0);
            kmer_gen->setThreshold(localThreshold);
        }
        
        last_stats_.num_query_positions++;
        
        if (take_only_best_kmer) {
            // Exact k-mer matching only
            // This matches QueryMatcher behavior when takeOnlyBestKmer=true
            size_t exactKmer = indexer->int2index(kmer, 0, kmer_size);
            uint32_t kmer_idx = static_cast<uint32_t>(exactKmer);
            uint16_t hint_idx = static_cast<uint16_t>(
                DpuIndexBuilder::calculateHintIndex(kmer_idx, kmer_size)
            );
            uint16_t query_pos = static_cast<uint16_t>(pos);
            
            KmerQueryPacket packet;
            packet.kmer_idx = kmer_idx;
            packet.hint_idx = hint_idx;
            packet.query_pos = query_pos;
            
            packets.push_back(packet);
            last_stats_.total_similar_kmers++;
        } else {
            // Similar k-mer matching: Generate variants using KmerGenerator
            // Expands to set of similar k-mers based on substitution matrix
            std::pair<size_t*, size_t> similar_kmers = kmer_gen->generateKmerList(kmer);
            
            last_stats_.total_similar_kmers += similar_kmers.second;
            
            // Create packet for each similar k-mer
            for (size_t i = 0; i < similar_kmers.second; ++i) {
                uint32_t kmer_idx = static_cast<uint32_t>(similar_kmers.first[i]);
                uint16_t hint_idx = static_cast<uint16_t>(
                    DpuIndexBuilder::calculateHintIndex(kmer_idx, kmer_size)
                );
                uint16_t query_pos = static_cast<uint16_t>(pos);
                
                KmerQueryPacket packet;
                packet.kmer_idx = kmer_idx;
                packet.hint_idx = hint_idx;
                packet.query_pos = query_pos;
                
                packets.push_back(packet);
            }
        }
    }
    
    last_stats_.total_packets = packets.size();
    if (last_stats_.num_query_positions > 0) {
        last_stats_.avg_similar_per_position = static_cast<double>(last_stats_.total_similar_kmers) / last_stats_.num_query_positions;
    }
    
    return packets;
}

} // namespace mmseqs::dpu
