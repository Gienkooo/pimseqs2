#include "DpuIndexBuilder.h"
#include "Debug.h"
#include "Indexer.h"
#include <algorithm>
#include <cstring>

#ifdef DPU_DEBUG_MODE
  #define DPU_DEBUG_LOG Debug(Debug::INFO)
#else
  #define DPU_DEBUG_LOG if (false) Debug(Debug::INFO)
#endif

namespace mmseqs::dpu {

/**
 * Builds a DPU-optimized k-mer index from database sequences
 * 
 * Creates three data structures:
 * 1. Hint table: Fast lookup to narrow binary search range
 * 2. Keys array: Sorted unique k-mer values
 * 3. Entries array: (target_id, position) pairs for each k-mer occurrence
 * 
 * The hint table uses the top bits of k-mer values to provide O(1) range hints,
 * reducing binary search space. Empty hint slots point to the next valid range.
 * 
 * @param tdbr Database reader
 * @param target_ids List of target sequence IDs to index
 * @param kmer_size K-mer length (5-7 for proteins)
 * @param subMat Substitution matrix for encoding
 * @param useSpacedKmers Whether to use spaced k-mer patterns
 * @param spacedPattern Array of positions for spaced k-mers
 * @param patternSpan Total span of spaced pattern
 * @return DpuIndexBuffer containing the built index
 */
DpuIndexBuffer DpuIndexBuilder::build(
    DBReader<unsigned int>* tdbr,
    const std::vector<uint32_t>& target_ids,
    int kmer_size,
    BaseMatrix* subMat,
    bool useSpacedKmers,
    const uint8_t* spacedPattern,
    int patternSpan
) {
    DpuIndexBuffer buffer;
    
    if (target_ids.empty() || kmer_size <= 0) {
        Debug(Debug::WARNING) << "[DPU] IndexBuilder: Empty input or invalid k-mer size\n";
        return buffer;
    }
    
    if (target_ids.size() > MAX_DPU_SEQS) {
        Debug(Debug::ERROR) << "[DPU] IndexBuilder: Too many sequences (" << target_ids.size() << " > " << MAX_DPU_SEQS << ")\n";
        return buffer;
    }
    
    DPU_DEBUG_LOG << "[DPU] IndexBuilder: Building index for " << target_ids.size() 
                       << " sequences with k=" << kmer_size 
                       << (useSpacedKmers ? " (spaced)" : " (contiguous)") << "\n";
    
    // Create local thread-safe indexer
    Indexer indexer(subMat->alphabetSize - 1, kmer_size);
    
    // Step 1: Extract exact k-mers from all target sequences
    std::vector<TempIndexEntry> temp_entries;
    temp_entries.reserve(target_ids.size() * 500); // Estimate: ~500 AA per protein
    
    for (size_t local_id = 0; local_id < target_ids.size(); ++local_id) {
        uint32_t db_key = target_ids[local_id];
        char* seq_data = tdbr->getData(db_key, 0);
        size_t seq_len = tdbr->getSeqLen(db_key);
        
        if (seq_len < (size_t)kmer_size) continue;
        
        // Encode sequence
        std::vector<uint8_t> encoded(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            unsigned char aa = static_cast<unsigned char>(seq_data[i]);
            encoded[i] = (subMat->aa2num) ? subMat->aa2num[aa] : 20;
            if (encoded[i] >= 21) encoded[i] = 20;
        }
        
        // Extract k-mers (spaced or contiguous)
        int windowSize = useSpacedKmers ? patternSpan : kmer_size;
        if ((int)seq_len < windowSize) continue;
        
        for (size_t j = 0; j <= seq_len - windowSize; ++j) {
            bool contains_x = false; // Flag for invalid residues

            // Extract k-mer using spaced pattern if enabled
            uint8_t kmer_buf[32];
            if (useSpacedKmers && spacedPattern != nullptr) {
                // Spaced k-mer: sample specific positions
                for (int pos = 0; pos < kmer_size; ++pos) {
                    uint8_t aa = encoded[j + spacedPattern[pos]];
                    if (aa >= 20) { contains_x = true; break; } // SKIP X or invalid
                    kmer_buf[pos] = aa;
                }
            } else {
                // Contiguous k-mer: copy sequential positions
                for (int pos = 0; pos < kmer_size; ++pos) {
                    uint8_t aa = encoded[j + pos];
                    if (aa >= 20) { contains_x = true; break; } // SKIP X or invalid
                    kmer_buf[pos] = aa;
                }
            }

            // Skip k-mers with unknown residues
            if (contains_x) continue;

            // Encode k-mer to integer
            uint32_t kmer_val = indexer.int2index(kmer_buf, 0, kmer_size);

            TempIndexEntry entry;
            entry.kmer = kmer_val;
            entry.local_id = static_cast<uint16_t>(local_id);
            entry.pos = static_cast<uint16_t>(j);

            temp_entries.push_back(entry);
        }
    }
    
    DPU_DEBUG_LOG << "[DPU] IndexBuilder: Extracted " << temp_entries.size() << " k-mers\n";
    
    if (temp_entries.empty()) {
        Debug(Debug::WARNING) << "[DPU] IndexBuilder: No k-mers extracted\n";
        return buffer;
    }
    
    // Step 2: Sort by k-mer value
    std::sort(temp_entries.begin(), temp_entries.end());
    
    // Build flat index arrays with hint table
    buffer.hints.resize(HINT_TABLE_SIZE + 1, 0xFFFFFFFF);
    
    uint32_t current_kmer = 0xFFFFFFFF;
    
    for (const auto& entry : temp_entries) {
        if (entry.kmer != current_kmer) {
            // New unique k-mer - add to keys array
            buffer.keys.push_back(entry.kmer);
            buffer.offsets.push_back(static_cast<uint32_t>(buffer.entries.size()));
            current_kmer = entry.kmer;
            
            // Update hint table for this k-mer
            uint32_t hint_idx = calculateHintIndex(current_kmer, kmer_size);
            if (buffer.hints[hint_idx] == 0xFFFFFFFF) {
                buffer.hints[hint_idx] = static_cast<uint32_t>(buffer.keys.size() - 1);
            }
        }
        
        KmerCompactIndexEntry compact;
        compact.local_target_id = entry.local_id;
        compact.pos_j = entry.pos;
        buffer.entries.push_back(compact);
    }
    
    buffer.offsets.push_back(static_cast<uint32_t>(buffer.entries.size()));
    
    // Pad entries array for 8-byte MRAM alignment
    // KmerCompactIndexEntry is 4 bytes, so odd counts need one padding entry
    if (buffer.entries.size() % 2 != 0) {
        KmerCompactIndexEntry padding_entry;
        padding_entry.local_target_id = 0xFFFF;
        padding_entry.pos_j = 0xFFFF;
        buffer.entries.push_back(padding_entry);

    }
    
    // Step 4: Set sentinel for end of table
    buffer.hints[HINT_TABLE_SIZE] = static_cast<uint32_t>(buffer.keys.size());
    
    // Backward fill empty hints to create valid [start, end) ranges
    // Each hint points to the first key with that hint value
    // Empty hints point to the same index as the next hint (zero-size range)
    uint32_t next_valid_idx = buffer.hints[HINT_TABLE_SIZE];
    
    for (int i = HINT_TABLE_SIZE - 1; i >= 0; --i) {
        if (buffer.hints[i] == 0xFFFFFFFF) {
            buffer.hints[i] = next_valid_idx;
        } else {
            next_valid_idx = buffer.hints[i];
        }
    }
    
    DPU_DEBUG_LOG << "[DPU] IndexBuilder: Created index with " << buffer.keys.size() 
                       << " unique k-mers, " << buffer.entries.size() << " total entries ("
                       << (buffer.getTotalBytes() / 1024) << " KB)\n";
    
    return buffer;
}

} // namespace mmseqs::dpu
