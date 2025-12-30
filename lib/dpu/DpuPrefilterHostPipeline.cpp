#include "DpuPrefilterHostPipeline.h"

// Standard Libraries
#include <cstring>
#include <unistd.h>
#include <limits.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <memory>

// MMseqs2
#include "Debug.h"
#include "StripedSmithWaterman.h"
#include "Matcher.h"
#include "QueryMatcher.h"
#include "QueryMatcherTaxonomyHook.h"
#include "SubstitutionMatrix.h"
#include "KmerGenerator.h"
#include "Indexer.h"
#include "Alignment.h"

// DPU Specific
#include "DpuIndexBuilder.h"
#include "DpuDbSplitter.h"
#include "DpuQueryPacketGenerator.h"
#include "DpuKernelManager.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// comment out to disable DPU debug logs
// #define DPU_DEBUG_MODE

#ifdef DPU_DEBUG_MODE
  #define DPU_DEBUG_LOG Debug(Debug::INFO)
#else
  #define DPU_DEBUG_LOG if (false) Debug(Debug::INFO)
#endif

namespace mmseqs::dpu
{

    // Forward declaration for tasklet calculation helper (defined in HELPERS section)
    static inline uint8_t calculateActiveTasklets(uint32_t wram_per_tasklet_bytes);

    // ============================================================================
    // PIPELINE IMPLEMENTATION
    // ============================================================================

    DpuPrefilterHostPipeline::DpuPrefilterHostPipeline(uint32_t num_dpus)
        : dpu_comm_(num_dpus), workflow_(dpu_comm_), kernel_mgr_(dpu_comm_)
    {
        Debug(Debug::INFO) << "[DPU] Initialized pipeline with " << num_dpus << " DPUs\n";
    }

    DpuPrefilterHostPipeline::~DpuPrefilterHostPipeline() {}

    void DpuPrefilterHostPipeline::runPrefilterOnDpu(
        Parameters &par, BaseMatrix *subMat, int8_t *tinySubMat,
        DBReader<unsigned int> *qdbr, DBReader<unsigned int> *tdbr,
        SequenceLookup *sequenceLookup, bool sameDB, DBWriter &resultWriter,
        EvalueComputation *evaluer, QueryMatcherTaxonomyHook *taxonomyHook,
        int alignmentMode,
        int kmerThr, ScoreMatrix *extMatTwo, ScoreMatrix *extMatThree,
        const std::string &spacedPatternStr, bool takeOnlyBestKmer) 
    {

        Debug(Debug::INFO) << "[DPU] Dispatch: prefMode=" << par.prefMode << " alignMode=" << alignmentMode << "\n";

        if (alignmentMode == 1 || par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE)
        {
            runDpuGappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        }
        else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED)
        {
            runDpuUngappedGappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        }
        else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED)
        {
            runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        }
        else if (par.prefMode == Parameters::PREF_MODE_KMER)
        {
            runDpuKmerBatch(par, subMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter,
                            kmerThr, extMatTwo, extMatThree, spacedPatternStr, takeOnlyBestKmer);
        }
        else
        {
            Debug(Debug::WARNING) << "[DPU] Mode " << par.prefMode << " not supported, falling back to ungapped\n";
            runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        }
    }

    // ============================================================================
    // 1. KMER BATCH
    // ============================================================================

    void DpuPrefilterHostPipeline::runDpuKmerBatch(
        Parameters &par, BaseMatrix *subMat, DBReader<unsigned int> *qdbr,
        DBReader<unsigned int> *tdbr, EvalueComputation *evaluer,
        QueryMatcherTaxonomyHook *taxonomyHook, bool sameDB, DBWriter &resultWriter,
        int kmerThr, ScoreMatrix *extMatTwo, ScoreMatrix *extMatThree,
        const std::string &spacedPatternStr, bool takeOnlyBestKmer) {
        
        const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
        if (num_dpus == 0) {
            Debug(Debug::ERROR) << "[CPU] No active CPUs available\n";
            return;
        }

        DPU_DEBUG_LOG << "loading kernel\n";
        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::KMER);
        DPU_DEBUG_LOG << " done\n";

        // Setup parameters
        int ksize = par.kmerSize;

        int alphabetSize = subMat->alphabetSize;
        // Pass reduced alphabet size for Amino Acids (21 -> 20) to Gnerator to match upstream matrices
        std::unique_ptr<KmerGenerator> kmerGen = std::make_unique<KmerGenerator>(ksize, alphabetSize - 1, (short)kmerThr);
        std::unique_ptr<Indexer> indexer = std::make_unique<Indexer>(alphabetSize, ksize);
        
        // Setup divide strategy for similar k-mer generation
        if (extMatTwo && extMatThree) {
            kmerGen->setDivideStrategy(extMatThree, extMatTwo);
        } else if (!takeOnlyBestKmer && Parameters::isEqualDbtype(qdbr->getDbtype(), Parameters::DBTYPE_AMINO_ACIDS)) {
            Debug(Debug::ERROR) << "[DPU] Similar k-mers requested but matrices missing\n";
            EXIT(EXIT_FAILURE);
        }
        
        // Spaced k-mers (if enabled)
        bool useSpacedKmers = (par.spacedKmer != 0);
        uint8_t spacedPattern[16] = {0};
        int patternSpan = ksize;
        
        if (useSpacedKmers) {
            // Use the passed pattern string directly
            if (!spacedPatternStr.empty()) {
                int pSpan = static_cast<int>(spacedPatternStr.length());
                int patternIdx = 0;
                for (int i = 0; i < pSpan && patternIdx < ksize; ++i) {
                    if (spacedPatternStr[i] == '1') {
                        if (patternIdx < (int)sizeof(spacedPattern)) {
                            spacedPattern[patternIdx++] = static_cast<uint8_t>(i);
                        }
                    }
                }
                // If we didn't collect any positions, disable spaced k-mers
                if (patternIdx == 0) {
                    useSpacedKmers = false;
                } else {
                    patternSpan = pSpan;
                }
            } else {
                // No pattern available
                useSpacedKmers = false;
            }
        }
        
        auto startTime = std::chrono::high_resolution_clock::now();
        uint64_t totalHits = 0;
        uint64_t totalPacketsSent = 0;
        uint64_t totalDoubleHits = 0;  

        // === PRE-CALCULATE STATIC MRAM OFFSETS ===
        // These offsets are constant for all DPUs and all waves because the "Fixed Region" sizes are compile-time constants.
        const uint32_t DESC_SIZE_ALIGNED = DpuCommunicationManager::alignToMram(sizeof(KmerBatchDescriptor));
        const uint32_t RESULTS_HEADER_OFF = DESC_SIZE_ALIGNED;
        const uint32_t CHECKPOINT_OFF = RESULTS_HEADER_OFF + DpuCommunicationManager::alignToMram(sizeof(KmerResultHeader));
        const uint32_t HINTS_OFF = CHECKPOINT_OFF + DpuCommunicationManager::alignToMram(sizeof(KmerCheckpoint));
        
        // Hint table is fixed size (HINT_TABLE_SIZE + 1)
        const uint32_t HINT_BYTES_ALIGNED = DpuCommunicationManager::alignToMram((HINT_TABLE_SIZE + 1) * sizeof(uint32_t));
        const uint32_t STATE_TABLE_OFF = HINTS_OFF + HINT_BYTES_ALIGNED;
        const uint32_t QUERY_PACKETS_OFF = STATE_TABLE_OFF + DpuCommunicationManager::alignToMram(MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
        
        // The Variable Region (Keys, Offsets, Entries) always starts here
        const uint32_t VARIABLE_INDEX_START_OFF = QUERY_PACKETS_OFF + KMER_QUERY_BUFFER_SIZE;

        // Per-query statistics for final summary
        std::vector<uint64_t> perQueryPackets;
        std::vector<uint64_t> perQueryDoubleHits;
        
        auto splits = DpuDbSplitter::splitDatabase(tdbr, num_dpus, MAX_DPU_INDEX_SIZE, MAX_DPU_SEQS);
        
        if (splits.empty()) {
            Debug(Debug::ERROR) << "[CPU] Database splitting failed\n";
            return;
        }
        
        size_t num_waves = splits.size() / num_dpus;
        if (splits.size() % num_dpus != 0) num_waves++;
        
        for (size_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
            size_t wave_start = wave_idx * num_dpus;
            size_t wave_end = std::min(wave_start + num_dpus, splits.size());
            size_t wave_size = wave_end - wave_start;
            
            DPU_DEBUG_LOG << "\n[CPU] ========== Wave " << (wave_idx + 1) << "/" << num_waves << " ========== \n";
            DPU_DEBUG_LOG << "[CPU] Processing chunks " << wave_start << " to "  << (wave_end - 1) << "\n";
            
            // Build indices for all chunks in the wave 
            std::vector<DpuIndexBuffer> wave_indices(wave_size);
            
            #pragma omp parallel for
            for (size_t w = 0; w < wave_size; ++w) {
                size_t chunk_idx = wave_start + w;
                wave_indices[w] = DpuIndexBuilder::build(
                    tdbr, splits[chunk_idx], ksize, subMat,
                    useSpacedKmers, spacedPattern, patternSpan
                );
            }
            
            // Transfer indices to DPUs
            // Prepare parallel transfer buffers (one per DPU slot in the wave)
            // Must be sized to num_dpus for scatterDataParallel, even if some slots are empty
            std::vector<std::vector<uint8_t>> wave_hints(num_dpus);
            std::vector<std::vector<uint8_t>> wave_index_buffers(num_dpus);

            for (size_t w = 0; w < wave_size; ++w) {
                uint32_t dpu_id = static_cast<uint32_t>(w);
                const auto& index = wave_indices[w];

                if (index.keys.empty()) {
                    Debug(Debug::WARNING) << "[CPU] Chunk " << (wave_start + w) << " has empty index, skipping\n";
                    continue;
                }
                
                // Check total size using pre-calculated base offset
                uint32_t keys_size = DpuCommunicationManager::alignToMram(index.keys.size() * sizeof(uint32_t));
                uint32_t offsets_size = DpuCommunicationManager::alignToMram(index.offsets.size() * sizeof(uint32_t));
                uint32_t entries_size = DpuCommunicationManager::alignToMram(index.entries.size() * sizeof(KmerCompactIndexEntry));
                
                uint32_t variable_structures_end = VARIABLE_INDEX_START_OFF + keys_size + offsets_size + entries_size;
                uint32_t fixed_structures_end = VARIABLE_INDEX_START_OFF;
                DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Memory Layout:\n";
                DPU_DEBUG_LOG << "  Fixed region: " << (fixed_structures_end / 1024) << " KB\n";
                DPU_DEBUG_LOG << "    Descriptor:    offset 0x0\n";
                DPU_DEBUG_LOG << "    Result Header: offset " << RESULTS_HEADER_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    Checkpoint:    offset " << CHECKPOINT_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    Hints:         offset " << HINTS_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    State Table:   offset " << STATE_TABLE_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    Query Buffer:  offset " << QUERY_PACKETS_OFF << " (STATIC, 1MB)\n";
                DPU_DEBUG_LOG << "  Variable region: " << ((variable_structures_end - fixed_structures_end) / 1024) << " KB\n";
                DPU_DEBUG_LOG << "    Index Keys:    " << index.keys.size() << " entries\n";
                DPU_DEBUG_LOG << "    Index Offsets: " << index.offsets.size() << " entries\n";
                DPU_DEBUG_LOG << "    Index Entries: " << index.entries.size() << " entries\n";

                // === OUTPUT BUFFER: Uses remaining MRAM ===
                uint32_t results_off = variable_structures_end;
                
                // MRAM Safety Check: Ensure everything fits in 64MB
                uint32_t required_for_buffers = KMER_MIN_OUTPUT_BUFFER_SIZE;
                if (variable_structures_end + required_for_buffers > DPU_MRAM_TOTAL_SIZE) {
                    Debug(Debug::ERROR) << "[CPU] ERROR: DPU " << dpu_id << " MRAM overflow!\n";
                    Debug(Debug::ERROR) << "  Fixed structures: " << fixed_structures_end << " bytes (" 
                                        << (fixed_structures_end / 1024) << " KB)\n";
                    Debug(Debug::ERROR) << "  Variable index: " << (variable_structures_end - fixed_structures_end) 
                                        << " bytes (" << ((variable_structures_end - fixed_structures_end) / 1024) << " KB)\n";
                    Debug(Debug::ERROR) << "  Min output buffer: " << required_for_buffers << " bytes\n";
                    Debug(Debug::ERROR) << "  Total required: " << (variable_structures_end + required_for_buffers) << " bytes\n";
                    Debug(Debug::ERROR) << "  MRAM available: " << DPU_MRAM_TOTAL_SIZE << " bytes\n";
                    Debug(Debug::ERROR) << "  SOLUTION: Reduce MAX_DPU_SEQS or database chunk size\n";
                    EXIT(EXIT_FAILURE);
                }
                
                // Build hint buffer for this DPU
                wave_hints[w].resize(HINT_BYTES_ALIGNED);
                memcpy(wave_hints[w].data(), index.hints.data(), index.hints.size() * sizeof(uint32_t));

                // Build merged index buffer (keys | offsets | entries) in variable region
                uint32_t keys_size_bytes = index.keys.size() * sizeof(uint32_t);
                uint32_t offsets_size_bytes = index.offsets.size() * sizeof(uint32_t);
                uint32_t entries_size_bytes = index.entries.size() * sizeof(KmerCompactIndexEntry);

                uint32_t total_index_size =
                    DpuCommunicationManager::alignToMram(keys_size_bytes) +
                    DpuCommunicationManager::alignToMram(offsets_size_bytes) +
                    DpuCommunicationManager::alignToMram(entries_size_bytes);

                wave_index_buffers[w].resize(total_index_size);
                uint8_t* ptr = wave_index_buffers[w].data();

                // Copy Keys
                if (keys_size_bytes > 0) memcpy(ptr, index.keys.data(), keys_size_bytes);
                ptr += DpuCommunicationManager::alignToMram(keys_size_bytes);

                // Copy Offsets
                if (offsets_size_bytes > 0) memcpy(ptr, index.offsets.data(), offsets_size_bytes);
                ptr += DpuCommunicationManager::alignToMram(offsets_size_bytes);

                // Copy Entries
                if (entries_size_bytes > 0) memcpy(ptr, index.entries.data(), entries_size_bytes);

                DPU_DEBUG_LOG << "[CPU " << dpu_id << "] Prepared index: " << index.keys.size() << " keys, " << index.entries.size() << " entries (" << (index.getTotalBytes() / 1024) << " KB)\n";
            }

            {
                // Transfer Hints (Fixed Region) in one pass
                dpu_comm_.scatterDataParallel(wave_hints, HINTS_OFF);

                // Transfer merged Index Data (Variable Region) in one pass
                dpu_comm_.scatterDataParallel(wave_index_buffers, VARIABLE_INDEX_START_OFF);

                // Log which DPUs received indices
                for (size_t w = 0; w < wave_size; ++w) {
                    if (!wave_index_buffers[w].empty()) {
                        DPU_DEBUG_LOG << "[CPU " << w << "] Loaded index (parallel) size=" << (wave_index_buffers[w].size() / 1024) << " KB, hints=" << (wave_hints[w].size() / 1024) << " KB\n";
                    }
                }
            }
            
            // === QUERY LOOP: Process each query with streaming ===
            std::vector<hit_t> raw_hits;
            std::string resultBuffer;
            for (size_t q = 0; q < qdbr->getSize(); ++q) {
                uint32_t queryKey = qdbr->getDbKey(q);
                uint32_t queryLen = qdbr->getSeqLen(q);
                const char* querySeq = qdbr->getData(q, 0);
                
                DPU_DEBUG_LOG << "[CPU] Generating packets for Query " << q << "...\n";
                
                // Encode query sequence
                std::vector<uint8_t> encoded(queryLen);
                for (size_t i = 0; i < queryLen; ++i) {
                    unsigned char aa = static_cast<unsigned char>(querySeq[i]);
                    encoded[i] = (subMat->aa2num) ? subMat->aa2num[aa] : 20;
                    if (encoded[i] >= 21) encoded[i] = 20;
                }
                
                // Generate packets using KmerGenerator (expands to similar k-mers)
                // If takeOnlyBestKmer=true, only exact k-mers are used (no expansion)
                // Prepare composition bias vector if enabled
                std::vector<float> compositionBiasVec;
                if (par.compBiasCorrection) {
                    compositionBiasVec.resize(queryLen);
                    SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, encoded.data(), (int)queryLen, compositionBiasVec.data(), par.compBiasCorrectionScale);
                }

                std::vector<KmerQueryPacket> packets = DpuQueryPacketGenerator::generate(
                    encoded, ksize, kmerGen.get(), indexer.get(),
                    useSpacedKmers, spacedPattern, patternSpan, takeOnlyBestKmer,
                    compositionBiasVec.empty() ? nullptr : compositionBiasVec.data(),
                    kmerThr
                );
                
                if (packets.empty()) {
                    Debug(Debug::WARNING) << "[CPU] Query " << q << " has no packets, skipping\n";
                    continue;
                }
        
                totalPacketsSent += packets.size();
                
                // Calculate batch size 
                uint32_t packets_per_batch = KMER_QUERY_BUFFER_SIZE / sizeof(KmerQueryPacket);
                
                // Per-DPU result accumulation
                std::vector<std::vector<KmerDoubleHit>> per_dpu_accumulated_results(num_dpus);
                
                // Count consecutive double hits for this query
                uint64_t query_double_hits = 0;
                
                // BATCH LOOP: Process query in multiple batches
                uint32_t global_packet_offset = 0;
                int total_iterations = 0;
                
                while (global_packet_offset < packets.size()) {
                    uint32_t remaining_packets = packets.size() - global_packet_offset;
                    uint32_t batch_packet_count = std::min(remaining_packets, packets_per_batch);
                    bool is_last_batch = (global_packet_offset + batch_packet_count >= packets.size());
                    
                    // Prepare Descriptors for Parallel Scatter
                    std::vector<std::vector<uint8_t>> descriptor_buffers(num_dpus);
                    
                    // Broadcast State Reset (Identical for all DPUs)
                    // Only needed for first batch
                    if (global_packet_offset == 0) {
                        std::vector<uint8_t> reset_state(DpuCommunicationManager::alignToMram(MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry)), 0xFF);
                        dpu_comm_.broadcastData(reset_state.data(), reset_state.size(), STATE_TABLE_OFF);
                    }
                    
                    // Broadcast Query Packets (Identical for all DPUs)
                    uint32_t packets_size = batch_packet_count * sizeof(KmerQueryPacket);
                    const KmerQueryPacket* batch_ptr = packets.data() + global_packet_offset;
                    dpu_comm_.broadcastData(batch_ptr, packets_size, QUERY_PACKETS_OFF);
                    
                    // Broadcast Checkpoint Reset (Identical)
                    KmerCheckpoint zero_checkpoint = {0, 0, 0, 0};
                    dpu_comm_.broadcastData(&zero_checkpoint, sizeof(KmerCheckpoint), CHECKPOINT_OFF);
                    
                    // Prepare all DPUs for this batch
                    for (uint32_t dpu_id = 0; dpu_id < num_dpus; ++dpu_id) {
                        // Handle idle DPUs or empty indices by sending a zeroed descriptor
                        if (dpu_id >= wave_size || wave_indices[dpu_id].keys.empty()) {
                            KmerBatchDescriptor empty_desc;
                            memset(&empty_desc, 0, sizeof(empty_desc));
                            // Set valid offsets so kernel writes 0-hit result to correct place
                            empty_desc.results_header_offset = RESULTS_HEADER_OFF; 
                            empty_desc.checkpoint_offset = CHECKPOINT_OFF;

                            descriptor_buffers[dpu_id].resize(DESC_SIZE_ALIGNED);
                            memcpy(descriptor_buffers[dpu_id].data(), &empty_desc, sizeof(empty_desc));
                            continue;
                        }

                        const auto& index = wave_indices[dpu_id];
                        const auto& chunk = splits[wave_start + dpu_id];
                        
                        // ===== VALIDATE INDEX STRUCTURE =====
                        // Critical: These sizes must match DPU expectations or offsets will be wrong
                        DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Index validation:\n";
                        DPU_DEBUG_LOG << "  hints.size() = " << index.hints.size() << " (expected: " << (HINT_TABLE_SIZE + 1) << ")\n";
                        DPU_DEBUG_LOG << "  keys.size() = " << index.keys.size() << "\n";
                        DPU_DEBUG_LOG << "  offsets.size() = " << index.offsets.size() << " (expected: keys.size() + 1 = " << (index.keys.size() + 1) << ")\n";
                        DPU_DEBUG_LOG << "  entries.size() = " << index.entries.size() << "\n";
                        
                        // ASSERT: hints array must be exactly HINT_TABLE_SIZE + 1
                        if (index.hints.size() != HINT_TABLE_SIZE + 1) {
                            Debug(Debug::ERROR) << "[CPU] CRITICAL: hints.size() mismatch! DPU expects " 
                                            << (HINT_TABLE_SIZE + 1) << " but got " << index.hints.size() << "\n";
                            Debug(Debug::ERROR) << "  This will cause MRAM offset corruption and hangs!\n";
                            // FATAL: DpuIndexBuilder violated the hint table invariant. 
                            // Skipping would result in silent data loss (ignored database chunk).
                            EXIT(EXIT_FAILURE); 
                        }
                        
                        // ASSERT: offsets array must be keys.size() + 1 (sentinel)
                        if (index.offsets.size() != index.keys.size() + 1) {
                            Debug(Debug::ERROR) << "[CPU] CRITICAL: offsets.size() mismatch! Expected " 
                                            << (index.keys.size() + 1) << " but got " << index.offsets.size() << "\n";
                            // FATAL: DpuIndexBuilder violated the offset array invariant.
                            // Skipping would result in silent data loss.
                            EXIT(EXIT_FAILURE); 
                        }
                        
                        // Use precomputed static offsets for fixed regions
                        uint32_t keys_off = VARIABLE_INDEX_START_OFF;
                        uint32_t offsets_off = keys_off + DpuCommunicationManager::alignToMram(index.keys.size() * sizeof(uint32_t));
                        uint32_t entries_off = offsets_off + DpuCommunicationManager::alignToMram(index.offsets.size() * sizeof(uint32_t));
                        uint32_t results_off = entries_off + DpuCommunicationManager::alignToMram(index.entries.size() * sizeof(KmerCompactIndexEntry));
                        
                        // Calculate output buffer size: use remaining MRAM after fixed structures
                        uint32_t remaining_mram = DPU_MRAM_TOTAL_SIZE - results_off;
                        uint32_t results_size = std::max((uint32_t)KMER_MIN_OUTPUT_BUFFER_SIZE, remaining_mram);
                        results_size = DpuCommunicationManager::alignToMram(results_size);
                        
                        // Final MRAM safety check
                        if (results_off + results_size > DPU_MRAM_TOTAL_SIZE) {
                            Debug(Debug::ERROR) << "[CPU] ERROR: DPU " << dpu_id << " output buffer overflow!\n";
                            Debug(Debug::ERROR) << "  Results offset: " << results_off << " bytes\n";
                            Debug(Debug::ERROR) << "  Results size: " << results_size << " bytes\n";
                            Debug(Debug::ERROR) << "  Total: " << (results_off + results_size) << " bytes\n";
                            Debug(Debug::ERROR) << "  MRAM size: " << DPU_MRAM_TOTAL_SIZE << " bytes\n";
                            EXIT(EXIT_FAILURE);
                        }

                        // Setup descriptor for this batch
                        KmerBatchDescriptor desc;
                        memset(&desc, 0, sizeof(desc));
                        desc.num_query_packets = batch_packet_count; // Packets in this batch
                        desc.num_targets = static_cast<uint32_t>(chunk.size());
                        desc.num_index_keys = static_cast<uint32_t>(index.keys.size());
                        desc.num_index_entries = static_cast<uint32_t>(index.entries.size());
                        desc.hint_table_offset = HINTS_OFF;
                        desc.query_packets_offset = QUERY_PACKETS_OFF;
                        desc.index_keys_offset = keys_off;
                        desc.index_offsets_offset = offsets_off;
                        desc.index_entries_offset = entries_off;
                        desc.state_table_offset = STATE_TABLE_OFF;
                        desc.checkpoint_offset = CHECKPOINT_OFF;
                        desc.results_header_offset = RESULTS_HEADER_OFF;
                        desc.results_offset = results_off;
                        desc.results_buffer_size = results_size;
                        desc.packet_start_idx = 0; // Always 0 in test mode (using offsets instead)
                        desc.reserved1 = 0;
                        
                        // Buffer valid descriptor
                        descriptor_buffers[dpu_id].resize(DESC_SIZE_ALIGNED);
                        memcpy(descriptor_buffers[dpu_id].data(), &desc, sizeof(desc));
                    }

                    dpu_comm_.scatterDataParallel(descriptor_buffers, 0);
                    
                    // OVERFLOW RESOLUTION LOOP: Handle output buffer overflow
                    bool all_dpus_complete = false;
                    int overflow_iterations = 0;
                    
                    while (!all_dpus_complete) {
                        overflow_iterations++;
                        // Launch all DPUs together
                        dpu_comm_.executeKernels();
                        
    #ifdef DPU_DEBUG_MODE
                        // Read kernel logs for this iteration
                        dpu_comm_.readAndPrintLog();
    #endif
                        
                        all_dpus_complete = true; // Assume complete unless we find overflow
                        
                        for (size_t w = 0; w < wave_size; ++w) {
                            uint32_t dpu_id = static_cast<uint32_t>(w);
                            const auto& index = wave_indices[w];
                            
                            if (index.keys.empty()) continue;
                            
                            // Use precomputed offsets for this DPU
                            uint32_t keys_off = VARIABLE_INDEX_START_OFF;
                            uint32_t offsets_off = keys_off + DpuCommunicationManager::alignToMram(index.keys.size() * sizeof(uint32_t));
                            uint32_t entries_off = offsets_off + DpuCommunicationManager::alignToMram(index.offsets.size() * sizeof(uint32_t));
                            uint32_t results_off = entries_off + DpuCommunicationManager::alignToMram(index.entries.size() * sizeof(KmerCompactIndexEntry));
                            
                            // Read result header from this DPU
                            KmerResultHeader result_header;
                            dpu_comm_.gatherDataFromDPU(dpu_id, &result_header, sizeof(KmerResultHeader), RESULTS_HEADER_OFF);
                            DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Iteration " << overflow_iterations << ": total_hits=" << result_header.total_hits << " overflow=" << result_header.overflow << "\n";

                            // Read double hits from this DPU
                            if (result_header.total_hits > 0) {
                                uint32_t hits_to_read = result_header.total_hits;
                                // Read even count for alignment (4-byte hits) and skip padding
                                uint32_t read_count_even = (hits_to_read + 1) & ~1U;
                                std::vector<KmerDoubleHit> iteration_results(read_count_even);
                                dpu_comm_.gatherDataFromDPU(dpu_id, iteration_results.data(), read_count_even * sizeof(KmerDoubleHit), results_off);
                                
                                // Append only real hits (skip padding sentinels if present)
                                uint32_t appended = 0;
                                for (uint32_t i = 0; i < read_count_even && appended < hits_to_read; ++i) {
                                    if (iteration_results[i].target_id == KMER_TARGET_ID_PADDING) continue;
                                    per_dpu_accumulated_results[dpu_id].push_back(iteration_results[i]);
                                    appended++;
                                }
                                
                                query_double_hits += appended;
                                DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Read " << appended << " double hits\n";
                                DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Total accumulated: " << per_dpu_accumulated_results[dpu_id].size() << " results\n";
                            }

                            if (result_header.overflow != 0) {
                                all_dpus_complete = false;
                                DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " has overflow, will relaunch\n";

                                // Clear result header only (checkpoint persists for resume)
                                KmerResultHeader zero_header = {0, 0};
                                dpu_comm_.scatterDataToDPU(dpu_id, &zero_header, sizeof(KmerResultHeader), RESULTS_HEADER_OFF);
                            }
                        }
                        
                        if (all_dpus_complete) {
                            DPU_DEBUG_LOG << "[CPU] All DPUs complete (no overflow)\n";
                        } else {
                            DPU_DEBUG_LOG << "[CPU] At least one DPU has overflow, relaunching batch\n";
                        }
                    } // end while overflow loop (!all_dpus_complete)
                    
                    // Batch complete, advance offset for next batch
                    global_packet_offset += batch_packet_count;
                    total_iterations += overflow_iterations;
                    DPU_DEBUG_LOG << "[CPU] Query " << q << " - Batch finished, advanced offset to " << global_packet_offset << " (" << overflow_iterations << " iteration(s))\n";
                    
                } // end while batch loop (global_packet_offset < total_packets_this_query)
                            
                DPU_DEBUG_LOG << "[CPU] Query " << q << " complete! Processed " << global_packet_offset << " packets in " << total_iterations << " total iteration(s)\n";

                raw_hits.clear();
                raw_hits.reserve(query_double_hits);

                size_t total_query_results = 0;
                for (size_t w = 0; w < wave_size; ++w) {
                    size_t dpu_results = per_dpu_accumulated_results[w].size();
                    total_query_results += dpu_results;
                    DPU_DEBUG_LOG << "[CPU]   DPU " << w << ": " << dpu_results << " results\n";

                    const auto& dpu_hits = per_dpu_accumulated_results[w];
                    const auto& chunk_targets = splits[wave_start + w];
                    
                    for (const auto& hit : dpu_hits) {
                        // Safety check: DPU index must be within chunk range
                        if (hit.target_id >= chunk_targets.size()) {
                            Debug(Debug::ERROR) << "[CPU] FATAL: DPU returned invalid TargetID " << hit.target_id 
                                                << ". Max valid ID for this chunk is " << (chunk_targets.size() - 1) << ".\n";
                            Debug(Debug::ERROR) << "  This indicates memory corruption or an index generation bug in the DPU kernel.\n";
                            Debug(Debug::ERROR) << "  Chunk start: " << splits[wave_start + w][0] << ", Chunk size: " << chunk_targets.size() << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        
                        hit_t shortHit;
                        shortHit.seqId = chunk_targets[hit.target_id]; // Map: Local DPU ID -> Global DB ID
                        shortHit.prefScore = 1;                        // "Hits" = 1 (Count/Existence)
                        shortHit.diagonal = hit.diagonal;              // Diagonal
                        
                        raw_hits.push_back(shortHit);
                    }
                }

                DPU_DEBUG_LOG << "[CPU]   Total results: " << total_query_results << "\n";
                DPU_DEBUG_LOG << "[CPU]   Consecutive double hits: " << query_double_hits << "\n";

                // Aggregate hits by (TargetId, Diagonal)
                std::vector<hit_t> final_query_hits;
                if (!raw_hits.empty()) {
                    // Sort by SeqId then Diagonal to group identical hits
                    std::sort(raw_hits.begin(), raw_hits.end(), [](const hit_t& a, const hit_t& b) {
                        if (a.seqId != b.seqId) return a.seqId < b.seqId;
                        return a.diagonal < b.diagonal;
                    });

                    final_query_hits.reserve(raw_hits.size());
                    final_query_hits.push_back(raw_hits[0]);

                    for (size_t i = 1; i < raw_hits.size(); ++i) {
                        hit_t& last = final_query_hits.back();
                        const hit_t& curr = raw_hits[i];
                        
                        if (last.seqId == curr.seqId && last.diagonal == curr.diagonal) {
                            last.prefScore += curr.prefScore;
                        } else {
                            final_query_hits.push_back(curr);
                        }
                    }
                }
                
                if (!final_query_hits.empty()) {
                    // Sort by score (descending) and then ID, required by MMseqs
                    std::sort(final_query_hits.begin(), final_query_hits.end(), hit_t::compareHitsByScoreAndId);
                    
                    // Limit results if configured
                    size_t keep = final_query_hits.size(); //std::min(final_query_hits.size(), (size_t)par.maxResListLen);
                    
                    resultBuffer.clear();
                    resultBuffer.reserve(keep * 16); // Optimization: Pre-reserve buffer
                    
                    for (size_t i = 0; i < keep; ++i) {
                        char outbuf[256];
                        // Formats as binary for DBWriter, which createtsv reads as:
                        // QueryId (Implicit) | TargetId | Score | Diagonal
                        size_t len = QueryMatcher::prefilterHitToBuffer(outbuf, final_query_hits[i]);
                        resultBuffer.append(outbuf, len);
                    }
                    
                    resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
                }
                
                // Track per-query stats for final summary
                perQueryPackets.push_back(packets.size());
                perQueryDoubleHits.push_back(query_double_hits);
                totalDoubleHits += query_double_hits;
                
            } // end for q (query loop)
            
            DPU_DEBUG_LOG << "\n[CPU] Wave " << (wave_idx + 1) << " complete! Processed " << qdbr->getSize() << " queries\n"; 
        } // end for wave_idx
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();
        
        // Per-query summary table
        for (size_t q = 0; q < perQueryPackets.size(); ++q) {
            double hitRate = (perQueryPackets[q] > 0) ? (100.0 * perQueryDoubleHits[q] / perQueryPackets[q]) : 0.0;
        }
    }

    // ============================================================================
    // 2. GAPPED BATCH (Standard SW / X-Drop)
    // ============================================================================
    void DpuPrefilterHostPipeline::runDpuGappedBatch(
        Parameters &par, BaseMatrix *subMat, int8_t *tinySubMat,
        DBReader<unsigned int> *qdbr, DBReader<unsigned int> *tdbr,
        EvalueComputation *evaluer, QueryMatcherTaxonomyHook *taxonomyHook,
        bool sameDB, DBWriter &resultWriter)
    {
        const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
        if (num_dpus == 0)
            return;

        std::string kernelName = "gapped_prefilter";
        std::string kPathInstalled = "lib/mmseqs/dpu/" + kernelName;
        std::string kPathBuild = "build/lib/dpu/kernels/" + kernelName;

        std::string kPath = kPathInstalled;
        if (access(kPathBuild.c_str(), F_OK) != -1)
        {
            kPath = kPathBuild;
        }

        auto perDpuTargetIndices = buildLoadBalancedDistribution(tdbr, num_dpus);
        std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
        std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

#pragma omp parallel for schedule(dynamic)
        for (uint32_t d = 0; d < num_dpus; ++d)
        {
            if (!perDpuTargetIndices[d].empty())
            {
                assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[d],
                                             perDpuTargetData[d], perDpuTargetMeta[d], subMat);
            }
        }

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);

        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::GAPPED);

        // DYNAMIC TASKLET CALCULATION
        // Each tasklet needs ~55KB WRAM for MAX_TARGET_LEN=5000
        // WRAM is 64KB. So only 1 tasklet can run safely by default.
        uint32_t wram_per_tasklet = (5000 + 1) * 2 * 5 + 5000 + 2048; // conservative estimate
        uint8_t active_tasklets = calculateActiveTasklets(wram_per_tasklet);

        bool prev_active = false;
        std::vector<DpuWorkflow::MramLayout> prev_layouts;
        uint32_t prev_queryKey = 0;
        uint32_t prev_queryLen = 0;

        for (size_t qId = 0; qId < qdbr->getSize(); ++qId)
        {
            kernel_mgr_.loadKernel(DpuKernelManager::KernelType::GAPPED);

            uint32_t queryKey = qdbr->getDbKey(qId);
            uint32_t queryLen = qdbr->getSeqLen(qId);
            const char *querySeq = qdbr->getData(qId, 0);

            int16_t minScoreThr = static_cast<int16_t>(par.minDiagScoreThr);
            if (evaluer && par.evalThr < std::numeric_limits<double>::max())
            {
                int rawMin = evaluer->minScore(par.evalThr, queryLen);
                if (rawMin > minScoreThr)
                    minScoreThr = static_cast<int16_t>(rawMin);
            }

            std::vector<int8_t> pssm = buildPSSMFromSequence(
                querySeq, queryLen, subMat,
                par.compBiasCorrection, par.compBiasCorrectionScale, compBias);

            QueryMetadata qmeta = {queryKey, queryLen, 0, 0};
            uint32_t qmeta_size = sizeof(QueryMetadata);
            uint32_t pssm_size = pssm.size();
            uint32_t common_size = DpuCommunicationManager::alignToMram(qmeta_size) +
                                   DpuCommunicationManager::alignToMram(pssm_size);
            std::vector<uint8_t> commonData(common_size, 0);
            memcpy(commonData.data(), &qmeta, qmeta_size);
            memcpy(commonData.data() + DpuCommunicationManager::alignToMram(qmeta_size), pssm.data(), pssm_size);

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);
            std::vector<GappedBatchDescriptor> bds(num_dpus);
            uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(GappedBatchDescriptor));

            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                layouts[d] = workflow_.calculateLayout(
                    sizeof(GappedBatchDescriptor), common_size, perDpuTargetMeta[d].size(), perDpuTargetData[d].size(), sizeof(GappedHit));

                GappedBatchDescriptor bd = {};
                bd.header.num_queries = 1;
                bd.header.num_targets = perDpuTargetMeta[d].size();
                bd.header.query_len = queryLen;
                bd.header.queries_metadata_offset = layouts[d].common_data_offset;
                bd.header.pssm_data_offset = bd.header.queries_metadata_offset + DpuCommunicationManager::alignToMram(qmeta_size);
                bd.header.targets_metadata_offset = layouts[d].target_meta_offset;
                bd.header.targets_data_offset = layouts[d].target_data_offset;
                bd.header.results_offset = layouts[d].results_offset;
                bd.header.results_buffer_size = layouts[d].results_capacity;
                bd.header.num_active_tasklets = active_tasklets;

                bd.header.pssm_total_size = pssm_size;
                bd.header.targets_total_size = perDpuTargetData[d].size();
                bd.header.flags = (par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE) ? 1 : 0;
                bd.min_score = minScoreThr;
                bd.cov_mode = (uint8_t)par.covMode;
                bd.cov_thr_pct = (uint8_t)(par.covThr * 100.0f);
                bd.min_aln_len = (uint8_t)std::min(par.alnLenThr, 255);
                bd.seq_id_thr_pct = (uint8_t)(par.seqIdThr * 100.0f);

                bd.gap_open_cost = static_cast<int16_t>(par.gapOpen.values.aminoacid());
                bd.gap_extend_cost = static_cast<int16_t>(par.gapExtend.values.aminoacid());
                bd.xdrop_threshold = static_cast<int16_t>(par.zdrop);
                bd.pssm_bias = 0;

                bds[d] = bd;
            }

            if (prev_active)
            {
                dpu_comm_.waitForKernels();

                std::vector<Matcher::result_t> resultsAln;
                for (uint32_t d = 0; d < num_dpus; ++d)
                {
                    std::vector<GappedHit> hits = workflow_.gatherResults<GappedHit>(d, prev_layouts[d].results_offset);
                    for (const auto &hit : hits)
                    {
                        unsigned int targetKey = tdbr->getDbKey(hit.target_id);
                        bool isIdentity = (prev_queryKey == targetKey && (par.includeIdentity || sameDB));
                        double evalue = 0.0;
                        if (par.evalThr < std::numeric_limits<double>::max())
                        {
                            evalue = evaluer->computeEvalue(hit.score, prev_queryLen);
                        }
                        if (!isIdentity && evalue > par.evalThr)
                            continue;

                        Matcher::result_t res;
                        res.dbKey = targetKey;
                        res.eval = evalue;
                        res.dbEndPos = hit.t_end;
                        res.dbLen = tdbr->getSeqLen(hit.target_id);
                        res.qEndPos = hit.q_end;
                        res.qLen = prev_queryLen;
                        if (evaluer)
                            res.score = static_cast<int>(evaluer->computeBitScore(hit.score) + 0.5);
                        else
                            res.score = hit.score;
                        res.qStartPos = 0;
                        res.dbStartPos = 0;
                        res.alnLength = Matcher::computeAlnLength(0, res.qEndPos, 0, res.dbEndPos);
                        res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen);
                        res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen);
                        unsigned int qAlnLen = std::max((unsigned int)res.qEndPos, 1u);
                        unsigned int dbAlnLen = std::max((unsigned int)res.dbEndPos, 1u);
                        res.seqId = Matcher::estimateSeqIdByScorePerCol(hit.score, qAlnLen, dbAlnLen);

                        if (Alignment::checkCriteria(res, isIdentity, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr))
                        {
                            resultsAln.push_back(res);
                        }
                    }
                }

                if (!resultsAln.empty())
                {
                    SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);
                    size_t maxSeqs = std::min((size_t)par.maxResListLen, resultsAln.size());
                    std::string resultBuffer;
                    for (size_t i = 0; i < maxSeqs; ++i)
                    {
                        char outbuf[4096];
                        size_t len = Matcher::resultToBuffer(outbuf, resultsAln[i], false);
                        resultBuffer.append(outbuf, len);
                    }
                    resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), prev_queryKey, 0);
                }

                prev_active = false;
            }

            // Now perform actual broadcast & scatter for the current query and launch asynchronously
            workflow_.broadcastCommon(commonData.data(), common_size, bd_size);
            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                workflow_.scatterBatch(d, bds[d], perDpuTargetMeta[d], perDpuTargetData[d], layouts[d]);
            }

            dpu_comm_.executeKernelsAsync();

            // Move current to prev for next iteration
            prev_active = true;
            prev_layouts.swap(layouts);
            prev_queryKey = queryKey;
            prev_queryLen = queryLen;
        }

        // Drain: wait for any remaining active batch and gather
        if (prev_active)
        {
            dpu_comm_.waitForKernels();
            std::vector<Matcher::result_t> resultsAln;
            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                std::vector<GappedHit> hits = workflow_.gatherResults<GappedHit>(d, prev_layouts[d].results_offset);
                for (const auto &hit : hits)
                {
                    unsigned int targetKey = tdbr->getDbKey(hit.target_id);
                    bool isIdentity = (prev_queryKey == targetKey && (par.includeIdentity || sameDB));
                    double evalue = 0.0;
                    if (par.evalThr < std::numeric_limits<double>::max())
                    {
                        evalue = evaluer->computeEvalue(hit.score, prev_queryLen);
                    }
                    if (!isIdentity && evalue > par.evalThr)
                        continue;

                    Matcher::result_t res;
                    res.dbKey = targetKey;
                    res.eval = evalue;
                    res.dbEndPos = hit.t_end;
                    res.dbLen = tdbr->getSeqLen(hit.target_id);
                    res.qEndPos = hit.q_end;
                    res.qLen = prev_queryLen;
                    if (evaluer)
                        res.score = static_cast<int>(evaluer->computeBitScore(hit.score) + 0.5);
                    else
                        res.score = hit.score;
                    res.qStartPos = 0;
                    res.dbStartPos = 0;
                    res.alnLength = Matcher::computeAlnLength(0, res.qEndPos, 0, res.dbEndPos);
                    res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen);
                    res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen);
                    unsigned int qAlnLen = std::max((unsigned int)res.qEndPos, 1u);
                    unsigned int dbAlnLen = std::max((unsigned int)res.dbEndPos, 1u);
                    res.seqId = Matcher::estimateSeqIdByScorePerCol(hit.score, qAlnLen, dbAlnLen);

                    if (Alignment::checkCriteria(res, isIdentity, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr))
                    {
                        resultsAln.push_back(res);
                    }
                }
            }

            if (!resultsAln.empty())
            {
                SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);
                size_t maxSeqs = std::min((size_t)par.maxResListLen, resultsAln.size());
                std::string resultBuffer;
                for (size_t i = 0; i < maxSeqs; ++i)
                {
                    char outbuf[4096];
                    size_t len = Matcher::resultToBuffer(outbuf, resultsAln[i], false);
                    resultBuffer.append(outbuf, len);
                }
                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), prev_queryKey, 0);
            }
        }
    }

    // ============================================================================
    // 3. UNGAPPED BATCH (Diagonal Only)
    // ============================================================================
    void DpuPrefilterHostPipeline::runDpuUngappedBatch(
        Parameters &par, BaseMatrix *subMat, int8_t *tinySubMat,
        DBReader<unsigned int> *qdbr, DBReader<unsigned int> *tdbr,
        EvalueComputation *evaluer, QueryMatcherTaxonomyHook *taxonomyHook,
        bool sameDB, DBWriter &resultWriter)
    {
        const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
        if (num_dpus == 0)
            return;

        auto perDpuTargetIndices = buildLoadBalancedDistribution(tdbr, num_dpus);
        std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
        std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

#pragma omp parallel for schedule(dynamic)
        for (uint32_t d = 0; d < num_dpus; ++d)
        {
            if (!perDpuTargetIndices[d].empty())
                assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[d],
                                             perDpuTargetData[d], perDpuTargetMeta[d], subMat);
        }

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);
        int16_t minScoreThr = static_cast<int16_t>(par.minDiagScoreThr);

        for (size_t qId = 0; qId < qdbr->getSize(); ++qId)
        {
            kernel_mgr_.loadKernel(DpuKernelManager::KernelType::UNGAPPED);

            uint32_t queryKey = qdbr->getDbKey(qId);
            uint32_t queryLen = qdbr->getSeqLen(qId);
            const char *querySeq = qdbr->getData(qId, 0);

            auto pssm = buildPSSMFromSequence(querySeq, queryLen, subMat, par.compBiasCorrection, par.compBiasCorrectionScale, compBias);

            QueryMetadata qmeta = {queryKey, queryLen, 0, 0};
            uint32_t qmeta_size = sizeof(QueryMetadata);
            uint32_t common_size = DpuCommunicationManager::alignToMram(qmeta_size) + DpuCommunicationManager::alignToMram(pssm.size());
            std::vector<uint8_t> commonData(common_size, 0);
            memcpy(commonData.data(), &qmeta, qmeta_size);
            memcpy(commonData.data() + DpuCommunicationManager::alignToMram(qmeta_size), pssm.data(), pssm.size());

            uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(UngappedBatchDescriptor));
            workflow_.broadcastCommon(commonData.data(), common_size, bd_size);

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);

            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                layouts[d] = workflow_.calculateLayout(
                    sizeof(UngappedBatchDescriptor), common_size, perDpuTargetMeta[d].size(), perDpuTargetData[d].size(), sizeof(Hit));

                UngappedBatchDescriptor bd = {};
                bd.header.num_queries = 1;
                bd.header.num_targets = perDpuTargetMeta[d].size();
                bd.header.query_len = queryLen;
                bd.header.queries_metadata_offset = layouts[d].common_data_offset;
                bd.header.pssm_data_offset = bd.header.queries_metadata_offset + DpuCommunicationManager::alignToMram(qmeta_size);
                bd.header.targets_metadata_offset = layouts[d].target_meta_offset;
                bd.header.targets_data_offset = layouts[d].target_data_offset;
                bd.header.results_offset = layouts[d].results_offset;
                bd.min_score = minScoreThr;
                bd.header.results_buffer_size = layouts[d].results_capacity;

                // Dynamic tasklets: conservatively estimate ~22KB WRAM per tasklet for UNGAPPED
                bd.header.num_active_tasklets = calculateActiveTasklets(22000);

                bd.gap_open_cost = static_cast<int16_t>(par.gapOpen.values.aminoacid());
                bd.gap_extend_cost = static_cast<int16_t>(par.gapExtend.values.aminoacid());
                bd.pssm_bias = 0;

                workflow_.scatterBatch(d, bd, perDpuTargetMeta[d], perDpuTargetData[d], layouts[d]);
            }

            dpu_comm_.executeKernels();
            // dpu_comm_.readAndPrintLog();

            std::vector<hit_t> queryResults;
            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                auto hits = workflow_.gatherResults<Hit>(d, layouts[d].results_offset);
                for (const auto &hit : hits)
                {
                    hit_t res;
                    res.seqId = tdbr->getDbKey(hit.target_id);
                    res.prefScore = hit.score;
                    res.diagonal = hit.diagonal;
                    queryResults.push_back(res);
                }
            }

            if (!queryResults.empty())
            {
                std::sort(queryResults.begin(), queryResults.end(), hit_t::compareHitsByScoreAndId);
                std::string resultBuffer;
                size_t keep = std::min(queryResults.size(), (size_t)par.maxResListLen);
                for (size_t i = 0; i < keep; ++i)
                {
                    char buffer[256];
                    size_t len = QueryMatcher::prefilterHitToBuffer(buffer, queryResults[i]);
                    resultBuffer.append(buffer, len);
                }
                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
            }
        }
    }

    // ============================================================================
    // 4. COMBINED BATCH (Ungapped + Gapped)
    // ============================================================================
    void DpuPrefilterHostPipeline::runDpuUngappedGappedBatch(
        Parameters &par, BaseMatrix *subMat, int8_t * /*tinySubMat*/,
        DBReader<unsigned int> *qdbr, DBReader<unsigned int> *tdbr,
        EvalueComputation *evaluer, QueryMatcherTaxonomyHook *taxonomyHook,
        bool sameDB, DBWriter &resultWriter)
    {
        const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
        if (num_dpus == 0)
            return;

        const int query_seq_type = qdbr->getDbtype();
        if (Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_HMM_PROFILE)) {
            Debug(Debug::ERROR) << "DPU ungapped+gapped does not support HMM profile queries\n";
            EXIT(EXIT_FAILURE);
        }

        // Use kernel manager to load combined (ungapped+gapped) kernel
        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::COMBINED);

        // DYNAMIC TASKLET CALCULATION
        // Based on ungapped_gapped_prefilter.c WRAM usage (~23KB per tasklet)
        uint32_t wram_per_tasklet = 23000u;
        uint8_t active_tasklets = calculateActiveTasklets(wram_per_tasklet); 

        // Target prep
        auto perDpuTargetIndices = buildLoadBalancedDistribution(tdbr, num_dpus);
        std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
        std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

        #pragma omp parallel for schedule(dynamic)
        for (uint32_t d = 0; d < num_dpus; ++d) {
            if (!perDpuTargetIndices[d].empty()) {
                assembleTargetBatchByIndices(tdbr,
                                            perDpuTargetIndices[d],
                                            perDpuTargetData[d],
                                            perDpuTargetMeta[d],
                                            subMat);
            }
        }

        Sequence qSeq(par.maxSeqLen, query_seq_type, subMat, 0, false, par.compBiasCorrection);
        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);

        // Kernel already loaded via kernel manager above

        for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
            const uint32_t queryKey = qdbr->getDbKey(qId);
            const unsigned int rawLen = qdbr->getSeqLen(qId);
            const char *rawSeq = qdbr->getData(qId, 0);

            qSeq.mapSequence(qId, queryKey, rawSeq, rawLen);
            const uint32_t L = static_cast<uint32_t>(qSeq.L);
            const uint8_t *qidx = reinterpret_cast<const uint8_t*>(qSeq.numSequence);

            int16_t minUngappedThr = static_cast<int16_t>(par.minDiagScoreThr);
            int16_t minGappedThr = minUngappedThr;
            if (evaluer && par.evalThr != std::numeric_limits<double>::max()) {
                const int rawMin = evaluer->minScore(par.evalThr, L);
                if (rawMin > minGappedThr) minGappedThr = static_cast<int16_t>(rawMin);
            }

            const uint32_t A = static_cast<uint32_t>(subMat->alphabetSize);
            constexpr uint32_t KERNEL_AA_SLOTS = 21;

            std::vector<int8_t> pssm = buildPSSMFromSequence(
                rawSeq, L, subMat,
                par.compBiasCorrection,
                par.compBiasCorrectionScale,
                compBias
            );

            QueryMetadata qmeta { queryKey, L, 0, 0 };

            const uint32_t qmeta_size = sizeof(QueryMetadata);
            const uint32_t pssm_size  = static_cast<uint32_t>(pssm.size());

            const uint32_t common_size =
                DpuCommunicationManager::alignToMram(qmeta_size) +
                DpuCommunicationManager::alignToMram(pssm_size);

            std::vector<uint8_t> commonData(common_size, 0);
            std::memcpy(commonData.data(), &qmeta, qmeta_size);
            std::memcpy(commonData.data() + DpuCommunicationManager::alignToMram(qmeta_size),
                        pssm.data(), pssm_size);

            const uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(CombinedBatchDescriptor));
            workflow_.broadcastCommon(commonData.data(), common_size, bd_size);

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);

            for (uint32_t d = 0; d < num_dpus; ++d) {
                layouts[d] = workflow_.calculateLayout(sizeof(CombinedBatchDescriptor), common_size,
                                                      perDpuTargetMeta[d].size(),
                                                      perDpuTargetData[d].size(),
                                                      sizeof(GappedHit));
                CombinedBatchDescriptor bd{};

                bd.header.num_queries = 1;
                bd.header.num_targets = static_cast<uint32_t>(perDpuTargetMeta[d].size());
                bd.header.query_len   = L;

                const uint32_t hits_area =
                    DpuCommunicationManager::alignToMram(bd.header.num_targets * sizeof(GappedHit) + 64);

                const uint32_t vec_bytes =
                    DpuCommunicationManager::alignToMram((bd.header.query_len + 1) * sizeof(int16_t));

                const uint32_t scratch_bytes = static_cast<uint32_t>(active_tasklets) * 2 * vec_bytes;

                // Make the reserved results region big enough for [count+hits+padding] + scratch vectors
                layouts[d].results_capacity =
                    DpuCommunicationManager::alignToMram(hits_area + scratch_bytes);

                layouts[d].total_mram_used = layouts[d].results_offset + layouts[d].results_capacity;   

                bd.header.queries_metadata_offset = layouts[d].common_data_offset;
                bd.header.pssm_data_offset        = bd.header.queries_metadata_offset + DpuCommunicationManager::alignToMram(qmeta_size);

                bd.header.targets_metadata_offset = layouts[d].target_meta_offset;
                bd.header.targets_data_offset     = layouts[d].target_data_offset;

                bd.header.results_offset         = layouts[d].results_offset;
                bd.header.results_buffer_size    = layouts[d].results_capacity;

                bd.min_ungapped_score     = minUngappedThr;
                bd.min_score              = minGappedThr;

                bd.cov_mode               = static_cast<uint8_t>(par.covMode);
                bd.cov_thr_pct            = static_cast<uint8_t>(par.covThr * 100.0f);

                bd.gap_open_cost          = static_cast<int16_t>(par.gapOpen.values.aminoacid());
                bd.gap_extend_cost        = static_cast<int16_t>(par.gapExtend.values.aminoacid());
                bd.xdrop_threshold        = static_cast<int16_t>(par.zdrop);
                bd.pssm_bias              = 0;

                // Dynamic tasklets: set based on computed active_tasklets
                bd.header.num_active_tasklets = active_tasklets;

                workflow_.scatterBatch(d, bd, perDpuTargetMeta[d], perDpuTargetData[d], layouts[d]);
            }

            // Ensure results header is zeroed on each DPU before launch (safety for early exits)
            uint32_t zeroHdr[2] = {0, 0};
            for (uint32_t d = 0; d < num_dpus; ++d) {
                dpu_comm_.scatterDataToDPU(d, zeroHdr, 8, layouts[d].results_offset);
            }

            dpu_comm_.executeKernels();
            dpu_comm_.readAndPrintLog();

            std::vector<Matcher::result_t> resultsAln;
            for (uint32_t d = 0; d < num_dpus; ++d) {
                // Read count header and clamp to what fits in allocated capacity and number of targets
                uint64_t count_buf = 0;
                dpu_comm_.gatherDataFromDPU(d, &count_buf, 8, layouts[d].results_offset);
                uint32_t hit_count = static_cast<uint32_t>(count_buf);

                uint32_t maxHitsByCapacity = 0;
                if (layouts[d].results_capacity >= 8) {
                    maxHitsByCapacity = (layouts[d].results_capacity - 8) / sizeof(GappedHit);
                }
                uint32_t maxHitsByTargets = static_cast<uint32_t>(perDpuTargetMeta[d].size());

                uint32_t clamped = std::min({hit_count, maxHitsByCapacity, maxHitsByTargets});
                if (clamped == 0) continue;

                uint32_t data_size = clamped * static_cast<uint32_t>(sizeof(GappedHit));
                uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);
                std::vector<uint8_t> buf(aligned_size);
                dpu_comm_.gatherDataFromDPU(d, buf.data(), aligned_size, layouts[d].results_offset + 8);

                std::vector<GappedHit> hits(clamped);
                memcpy(hits.data(), buf.data(), data_size);

                for (const auto &hit : hits) {
                    if (hit.target_id >= static_cast<uint32_t>(tdbr->getSize())) {
                        Debug(Debug::WARNING) << "[DPU] Dropping invalid hit.targetid=" << hit.target_id
                                              << " (tdb size=" << tdbr->getSize() << ")\n";
                        continue;
                    }

                    const unsigned int targetKey = tdbr->getDbKey(hit.target_id);

                    if (taxonomyHook != nullptr) {
                        TaxID currTax = taxonomyHook->taxonomyMapping->lookup(targetKey);
                        if (taxonomyHook->expression[0]->isAncestor(currTax) == false) continue;
                    }

                    const bool isIdentity = (queryKey == targetKey && par.includeIdentity && sameDB);

                    double evalue = 0.0;
                    if (par.evalThr != std::numeric_limits<double>::max()) {
                        evalue = evaluer->computeEvalue(hit.score, L);
                        if (!isIdentity && evalue > par.evalThr) continue;
                    }

                    Matcher::result_t res{};
                    res.dbKey = targetKey;
                    res.eval  = evalue;
                    res.dbEndPos = hit.t_end;
                    res.dbLen    = tdbr->getSeqLen(hit.target_id);
                    res.qEndPos   = hit.q_end;
                    res.qLen      = L;
                    res.score = evaluer ? evaluer->computeBitScore(hit.score) : hit.score;

                    res.qStartPos = 0;
                    res.dbStartPos = 0;
                    res.alnLength = Matcher::computeAlnLength(0, res.qEndPos, 0, res.dbEndPos);
                    res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen);
                    res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen);

                    const unsigned int qAlnLen  = std::max<unsigned int>(res.qEndPos, 1u);
                    const unsigned int dbAlnLen = std::max<unsigned int>(res.dbEndPos, 1u);
                    res.seqId = Matcher::estimateSeqIdByScorePerCol(hit.score, qAlnLen, dbAlnLen);

                    if (Alignment::checkCriteria(res, isIdentity, par.evalThr, par.seqIdThr,
                                                 par.alnLenThr, par.covMode, par.covThr)) {
                        resultsAln.push_back(res);
                    }
                }
            }

            if (!resultsAln.empty()) {
                SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);

                const size_t maxSeqs = std::min<size_t>(par.maxResListLen, resultsAln.size());
                std::string resultBuffer;
                resultBuffer.reserve(262144);

                for (size_t i = 0; i < maxSeqs; i++) {
                    char outbuf[4096];
                    const size_t len = Matcher::resultToBuffer(outbuf, resultsAln[i], false);
                    resultBuffer.append(outbuf, len);
                }

                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
            }
        }
    }

    // ============================================================================
    // HELPERS
    // ============================================================================

    static inline uint8_t calculateActiveTasklets(uint32_t wram_per_tasklet_bytes) {
        const uint32_t SAFE_WRAM = 62000u; 
        if (wram_per_tasklet_bytes == 0) return 1;
        uint32_t count = SAFE_WRAM / wram_per_tasklet_bytes;
        if (count < 1) count = 1;
        if (count > 16) count = 16;
        return (uint8_t)count;
    }

    std::vector<int8_t> DpuPrefilterHostPipeline::buildPSSMFromSequence(
        const char *sequence, uint32_t seqlen, BaseMatrix *subMat,
        bool compBiasCorrection, float compBiasCorrectionScale, std::vector<float> &compositionBias)
    {
        const int KERNEL_AA_SLOTS = 21;
        const int alphSize = subMat->alphabetSize;
        std::vector<int8_t> pssm(seqlen * KERNEL_AA_SLOTS, -128);

        std::vector<unsigned char> qidx(seqlen);
        for (uint32_t i = 0; i < seqlen; ++i)
        {
            unsigned char aa = static_cast<unsigned char>(sequence[i]);
            int idx = subMat->aa2num ? subMat->aa2num[aa] : 20;
            if (idx < 0 || idx >= 21)
                idx = 20;
            qidx[i] = static_cast<unsigned char>(idx);
        }

        if (compBiasCorrection)
        {
            if (compositionBias.size() < seqlen)
                compositionBias.resize(seqlen, 0.0f);
            SubstitutionMatrix::calcLocalAaBiasCorrection(
                subMat, qidx.data(), seqlen, compositionBias.data(), compBiasCorrectionScale);
        }

        for (uint32_t pos = 0; pos < seqlen; ++pos)
        {
            int q = static_cast<int>(qidx[pos]);
            if (q < 0 || q >= alphSize)
                continue;

            int bias = 0;
            if (compBiasCorrection)
            {
                float v = compositionBias[pos];
                bias = (v < 0.0f) ? (int)(v - 0.5f) : (int)(v + 0.5f);
            }

            for (int aa = 0; aa < KERNEL_AA_SLOTS; ++aa)
            {
                if (aa >= alphSize)
                    continue;
                int s = subMat->subMatrix[aa][q] + bias;
                if (s > 127)
                    s = 127;
                else if (s < -128)
                    s = -128;
                pssm[pos * KERNEL_AA_SLOTS + aa] = static_cast<int8_t>(s);
            }
        }

        /* Defensive check: some host codepaths may produce a biased PSSM (e.g., +128 offset
           used for unsigned SIMD kernels). Detect and remove +128 bias if present. */
        bool biased = false;
        for (size_t i = 0; i < pssm.size(); ++i)
        {
            if ((int)pssm[i] > 100)
            {
                biased = true;
                break;
            }
        }
        if (biased)
        {
            for (size_t i = 0; i < pssm.size(); ++i)
            {
                int v = (int)pssm[i];
                v -= 128;
                if (v > 127)
                    v = 127;
                else if (v < -128)
                    v = -128;
                pssm[i] = static_cast<int8_t>(v);
            }
        }
        return pssm;
    }

    void DpuPrefilterHostPipeline::assembleTargetBatchByIndices(
        DBReader<unsigned int> *tdbr,
        const std::vector<uint32_t> &target_indices,
        std::vector<uint8_t> &packed_sequences,
        std::vector<TargetMetadata> &metadata,
        BaseMatrix *subMat)
    {

        metadata.clear();
        packed_sequences.clear();
        uint32_t offset = 0;

        for (uint32_t target_id : target_indices)
        {
            if (target_id >= tdbr->getSize())
                continue;

            size_t seq_len = 0;
            const char *seq = tdbr->getData(target_id, 0);
            while (seq[seq_len] != '\0')
                seq_len++;

            TargetMetadata meta;
            meta.target_id = target_id;
            meta.target_len = seq_len;
            meta.offset_in_data = offset;
            meta.padding = 0;
            metadata.push_back(meta);

            for (size_t j = 0; j < seq_len; j++)
            {
                unsigned char aa = static_cast<unsigned char>(seq[j]);
                int num_aa = subMat->aa2num ? subMat->aa2num[aa] : 20;
                if (num_aa >= 21)
                    num_aa = 20;
                packed_sequences.push_back((uint8_t)num_aa);
            }
            while (packed_sequences.size() % 8 != 0)
                packed_sequences.push_back(0);
            offset = packed_sequences.size();
        }
    }

    std::vector<std::vector<uint32_t>> DpuPrefilterHostPipeline::buildLoadBalancedDistribution(
        DBReader<unsigned int> *tdbr, uint32_t num_dpus)
    {

        uint32_t totalTargets = tdbr->getSize();
        std::vector<std::pair<uint32_t, uint32_t>> lengthIndex(totalTargets);
        for (uint32_t i = 0; i < totalTargets; i++)
        {
            lengthIndex[i] = {tdbr->getSeqLen(i), i};
        }
        std::sort(lengthIndex.begin(), lengthIndex.end(),
                  [](const auto &a, const auto &b)
                  { return a.first > b.first; });

        std::vector<std::vector<uint32_t>> perDpuTargets(num_dpus);
        for (uint32_t i = 0; i < totalTargets; i++)
        {
            uint32_t dpu_idx = i % num_dpus;
            perDpuTargets[dpu_idx].push_back(lengthIndex[i].second);
        }
        return perDpuTargets;
    }

} // namespace mmseqs::dpu