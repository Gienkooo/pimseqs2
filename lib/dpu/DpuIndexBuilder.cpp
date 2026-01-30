#include "DpuIndexBuilder.h"
#include "DpuLog.h" 
#include "Debug.h"
#include "Indexer.h"
#include "Sequence.h"
#include "Masker.h"
#include "Parameters.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath> 
#include <numeric> 

namespace mmseqs::dpu {

DpuIndexBuffer DpuIndexBuilder::build(
    DBReader<unsigned int>* tdbr,
    const std::vector<uint32_t>& target_ids,
    int kmer_size,
    BaseMatrix* subMat,
    uint32_t global_chunk_id,
    uint32_t dpu_id,
    bool useSpacedKmers,
    const uint8_t* spacedPattern,
    int patternSpan,
    bool maskMode,
    int maskLowerCaseMode,
    float maskProb,
    int maskNrepeats
) {
    DpuIndexBuffer buffer;
    
    if (target_ids.empty() || kmer_size <= 0) return buffer;
    
    Indexer indexer(subMat->alphabetSize - 1, kmer_size);
    
    Sequence s(tdbr->getMaxSeqLen(), Parameters::DBTYPE_AMINO_ACIDS, subMat, kmer_size, useSpacedKmers, false, true, "");
    std::unique_ptr<Masker> masker;
    if (maskMode) {
        masker = std::make_unique<Masker>(*subMat);
    }
    
    std::vector<TempIndexEntry> temp_entries;
    temp_entries.reserve(target_ids.size() * 100);
    
    for (size_t local_id = 0; local_id < target_ids.size(); ++local_id) {
        uint32_t db_key = target_ids[local_id];
        size_t seq_len = tdbr->getSeqLen(db_key);
        
        if (seq_len < (size_t)kmer_size) continue;
        
        // Encode and mask sequence
        s.mapSequence(local_id, db_key, tdbr->getData(db_key, 0), seq_len);
        if (masker) {
            masker->maskSequence(s, maskMode, maskProb, maskLowerCaseMode, maskNrepeats);
        }
        const unsigned char* encoded = s.numSequence;
        
        // Extract k-mers
        int windowSize = useSpacedKmers ? patternSpan : kmer_size;
        if ((int)seq_len < windowSize) continue;
        
        for (size_t j = 0; j <= seq_len - windowSize; ++j) {
            bool contains_x = false;
            uint8_t kmer_buf[32];
            
            if (useSpacedKmers && spacedPattern != nullptr) {
                for (int pos = 0; pos < kmer_size; ++pos) {
                    uint8_t aa = encoded[j + spacedPattern[pos]];
                    if (aa >= 20) { contains_x = true; break; }
                    kmer_buf[pos] = aa;
                }
            } else {
                for (int pos = 0; pos < kmer_size; ++pos) {
                    uint8_t aa = encoded[j + pos];
                    if (aa >= 20) { contains_x = true; break; }
                    kmer_buf[pos] = aa;
                }
            }
            
            if (contains_x) continue;
            
            uint32_t kmer_val = indexer.int2index(kmer_buf, 0, kmer_size);
            temp_entries.push_back({kmer_val, (uint16_t)local_id, (uint16_t)j});
        }
    }
    
    std::sort(temp_entries.begin(), temp_entries.end());
    
    // Build bucket items
    struct BucketData {
        struct Item { uint32_t key; uint32_t offset; uint16_t count; };
        std::vector<Item> items;
    };
    std::vector<BucketData> temp_buckets(NUM_BUCKETS);
    
    uint32_t current_kmer = 0xFFFFFFFF;
    uint32_t current_offset = 0;
    uint16_t current_count = 0;
    
    for (const auto& entry : temp_entries) {
        if (entry.kmer != current_kmer) {
            if (current_kmer != 0xFFFFFFFF) {
                uint32_t bid = computeBucketIndex(current_kmer);
                temp_buckets[bid].items.push_back({current_kmer, current_offset, current_count});
            }
            current_kmer = entry.kmer;
            current_offset = static_cast<uint32_t>(buffer.entries.size());
            current_count = 0;
        }
        buffer.entries.push_back({entry.local_id, entry.pos});
        
        // DEBUG: Log if we are about to overflow uint16_t (65535 -> 0)
        if (current_count == 65535) {
            LOG_INDEX("WARNING: [Overflow] Chunk " << global_chunk_id << " DPU " << dpu_id 
                      << " K-mer " << current_kmer << " reached 65536 hits (wrapping to 0!)");
        }
        
        current_count++;
    }
    if (current_kmer != 0xFFFFFFFF) {
        uint32_t bid = computeBucketIndex(current_kmer);
        temp_buckets[bid].items.push_back({current_kmer, current_offset, current_count});
    }
    
    if (buffer.entries.size() % 2 != 0) buffer.entries.push_back({0xFFFF, 0xFFFF});

    std::vector<KmerBucket> final_buckets(NUM_BUCKETS);
    
    for (auto& b : final_buckets) {
        b.count = 0;
        b.pad1 = 0;
        b.next_idx = CHAIN_END_IDX;
        memset(b.padding, 0, sizeof(b.padding));
    }
    
    uint32_t next_overflow_idx = NUM_BUCKETS;
    
    for (uint32_t i = 0; i < NUM_BUCKETS; ++i) {
        const auto& items = temp_buckets[i].items;
        size_t item_idx = 0;
        uint32_t current_bucket_idx = i;
        
        while (item_idx < items.size()) {
            if (current_bucket_idx >= final_buckets.size()) {
                final_buckets.resize(current_bucket_idx + 1);
                final_buckets[current_bucket_idx].count = 0;
                final_buckets[current_bucket_idx].pad1 = 0;
                final_buckets[current_bucket_idx].next_idx = CHAIN_END_IDX;
                memset(final_buckets[current_bucket_idx].padding, 0, sizeof(final_buckets[current_bucket_idx].padding));
            }
            
            KmerBucket* b = &final_buckets[current_bucket_idx];
            size_t to_write = std::min((size_t)BUCKET_CAPACITY, items.size() - item_idx);
            
            for (size_t k = 0; k < to_write; ++k) {
                b->items[k].key = items[item_idx].key;
                b->items[k].offset = items[item_idx].offset;
                b->items[k].count = items[item_idx].count;
                b->items[k].pad = 0;
                item_idx++;
            }
            b->count = static_cast<uint16_t>(to_write);
            
            if (item_idx < items.size()) {
                uint32_t new_idx = next_overflow_idx++;
                b->next_idx = new_idx;
                current_bucket_idx = new_idx;
            }
        }
    }

    // Statistics Logging
    // 1. Basic Stats (Trace/Index)
     size_t empty_buckets = 0;
     size_t collision_buckets = 0; // Buckets that start a chain (next_idx != END)
     for(uint32_t i = 0; i < NUM_BUCKETS; ++i) {
         if (final_buckets[i].count == 0) empty_buckets++;
         if (final_buckets[i].next_idx != CHAIN_END_IDX) collision_buckets++;
     }
    
     LOG_INDEX("Chunk " << global_chunk_id << " (DPU " << dpu_id << "): Built Index. "
               << "Entries=" << buffer.entries.size() 
               << ", EmptyBuckets=" << empty_buckets << "/" << NUM_BUCKETS << " (" << (empty_buckets*100/NUM_BUCKETS) << "%)"
               << ", Collisions=" << collision_buckets << " (" << (collision_buckets*100/NUM_BUCKETS) << "%)");
    

    
    size_t total_bytes = final_buckets.size() * sizeof(KmerBucket);
    buffer.buckets.resize(total_bytes);
    std::memcpy(buffer.buckets.data(), final_buckets.data(), total_bytes);
    buffer.num_buckets = static_cast<uint32_t>(final_buckets.size());
    
    // 2. Detailed Stats (Index)
    size_t overflow_count = 0;
    size_t total_chain_depth = 0;
    size_t max_chain_depth = 0;
    size_t populated_buckets = 0;

    for (uint32_t i = 0; i < NUM_BUCKETS; ++i) {
        if (final_buckets[i].count > 0) populated_buckets++;
        if (final_buckets[i].next_idx != CHAIN_END_IDX) overflow_count++;
        
        size_t depth = 1;
        uint32_t curr = final_buckets[i].next_idx;
        while (curr != CHAIN_END_IDX) {
            depth++;
            if (curr >= final_buckets.size()) break;
            curr = final_buckets[curr].next_idx;
        }
        total_chain_depth += depth;
        max_chain_depth = std::max(max_chain_depth, depth);
    }

    LOG_INDEX("Chunk " << global_chunk_id << " Stats: Entries=" << buffer.entries.size() 
              << " PrimaryBuckets=" << NUM_BUCKETS
              << " OverflowRate=" << (100.0 * overflow_count / NUM_BUCKETS) << "% (" << overflow_count << " buckets chained)");
    LOG_INDEX("              AvgChainDepth=" << (double)total_chain_depth / NUM_BUCKETS 
              << " MaxChainDepth " << max_chain_depth);

    return buffer;
}

} // namespace mmseqs::dpu
