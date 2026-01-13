#include "DpuDbSplitter.h"
#include "Debug.h"
#include <algorithm>
#include <cmath>
#include <vector> 

namespace mmseqs::dpu {

    std::vector<std::vector<uint32_t>> DpuDbSplitter::splitDatabase(
        DBReader<unsigned int>* tdbr,
        uint32_t num_dpus,
        size_t mram_limit_bytes,
        uint32_t max_seqs_per_dpu
    ) {
        size_t total_seqs = tdbr->getSize();
        
        // 1. Gather Metadata
        std::vector<SequenceMetadata> seqs;
        seqs.reserve(total_seqs);
        
        for (size_t i = 0; i < total_seqs; ++i) {
            uint32_t key = tdbr->getDbKey(i);
            uint32_t len = tdbr->getSeqLen(i);
            size_t size = estimateSequenceSizeBytes(len);
            
            if (size > mram_limit_bytes) {
                Debug(Debug::ERROR) << "[DPU] Sequence " << key << " is too large (" 
                                    << size/1024/1024 << "MB) for DPU MRAM limit (" 
                                    << mram_limit_bytes/1024/1024 << "MB)\n";
                return {};
            }
            
            seqs.push_back({key, len, size});
        }

        // Sort Descending 
        std::sort(seqs.begin(), seqs.end(), [](const SequenceMetadata& a, const SequenceMetadata& b) {
            return a.estimated_size > b.estimated_size;
        });

        // 2. Greedy Linear Packing
        std::vector<DpuChunk> chunks;
        if (!seqs.empty()) {
            chunks.emplace_back(); 
        }

        for (const auto& seq : seqs) {
            DpuChunk& current = chunks.back();
            
            bool size_ok = (current.current_estimated_bytes + seq.estimated_size) <= mram_limit_bytes;
            bool count_ok = (current.current_seq_count + 1) <= max_seqs_per_dpu;

            if (size_ok && count_ok) {
                // Fits in current chunk
                current.sequence_ids.push_back(seq.db_key);
                current.current_seq_count++;
                current.current_estimated_bytes += seq.estimated_size;
                current.current_total_length += seq.length;
            } else {
                // Must start a new chunk
                chunks.emplace_back();
                DpuChunk& next = chunks.back();
                
                // We already validated that the seq fits in an empty chunk in step 1
                next.sequence_ids.push_back(seq.db_key);
                next.current_seq_count++;
                next.current_estimated_bytes += seq.estimated_size;
                next.current_total_length += seq.length;
            }
        }

        // 3. Convert to output format
        std::vector<std::vector<uint32_t>> result;
        result.reserve(chunks.size());
        
        size_t min_load = 0;
        size_t max_load = 0;
        if (!chunks.empty()) min_load = SIZE_MAX;
        
        for (const auto& chunk : chunks) {
            if (!chunk.sequence_ids.empty()) {
                result.push_back(chunk.sequence_ids);
                size_t bytes = chunk.current_estimated_bytes;
                if (bytes < min_load) min_load = bytes;
                if (bytes > max_load) max_load = bytes;
            }
        }
        
        size_t num_waves = 0;
        if (num_dpus > 0) {
            num_waves = (result.size() + num_dpus - 1) / num_dpus;
        }
        
        Debug(Debug::INFO) << "[DPU] Database split into " << result.size() 
                        << " chunks (" << num_waves << " waves)." 
                        << " Load Balance (Bytes): Min=" << min_load/1024 
                        << "KB Max=" << max_load/1024 << "KB\n";
        return result;
    }

} // namespace mmseqs::dpu