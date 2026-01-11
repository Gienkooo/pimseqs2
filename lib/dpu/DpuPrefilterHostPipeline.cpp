#include "DpuPrefilterHostPipeline.h"

// Standard Libraries
#include <cstring>
#include <unistd.h>
#include <limits.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <memory>
#include <utility>

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
#include "DpuGroupManager.h"
#include "Sequence.h"

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

    // --------------------------------------------------------------------------
    // Shared helper: build a batch of queries (metadata + packed PSSMs)
    // --------------------------------------------------------------------------
    DpuPrefilterHostPipeline::BatchData DpuPrefilterHostPipeline::buildQueryBatch(
        size_t start_q_idx,
        DBReader<unsigned int>* qdbr,
        BaseMatrix* subMat,
        const Parameters& par,
        std::vector<float>& compBias,
        const BatchLimits& limits,
        EvalueComputation* evaluer,
        int16_t minScoreThr,
        Sequence* seqMapper)
    {
        BatchData batch;
        batch.next_q_idx = start_q_idx;
        batch.empty = true;

        const bool is_profile = Parameters::isEqualDbtype(qdbr->getDbtype(), Parameters::DBTYPE_HMM_PROFILE);

        uint32_t current_pssm_offset = 0;
        uint32_t total_residues = 0;
        const uint32_t MAX_RESIDUES = 20000; // soft cap to avoid oversized batches

        while (batch.next_q_idx < qdbr->getSize() && batch.meta.size() < limits.max_queries) {
            const size_t qId = batch.next_q_idx;
            const uint32_t queryKey = qdbr->getDbKey(qId);
            const uint32_t queryLen = qdbr->getSeqLen(qId);
            const char* querySeq = qdbr->getData(qId, 0);

            uint32_t L = queryLen;
            if (seqMapper && !is_profile) {
                seqMapper->mapSequence(static_cast<unsigned int>(qId), queryKey, querySeq, queryLen);
                L = static_cast<uint32_t>(seqMapper->L);
            }

            // Soft residue guard
            if (!batch.meta.empty() && (total_residues + L > MAX_RESIDUES)) break;

            // Build PSSM
            if (compBias.size() < L) compBias.resize(L);
            auto pssm = is_profile
                ? extractPSSMFromProfile(querySeq, L, subMat)
                : buildPSSMFromSequence(querySeq, L, subMat, par.compBiasCorrection, par.compBiasCorrectionScale, compBias);

            // Hard PSSM byte guard
            if (batch.pssm.size() + pssm.size() > limits.max_pssm_bytes) break;

            // 32-byte padding for PSSM block
            const uint32_t pssm_size = static_cast<uint32_t>(pssm.size());
            const uint32_t padding32 = static_cast<uint32_t>((32u - (pssm_size % 32u)) % 32u);

            // Common buffer size check
            const uint32_t new_qmeta_bytes = static_cast<uint32_t>((batch.meta.size() + 1) * sizeof(QueryMetadata));
            const uint32_t new_pssm_bytes = static_cast<uint32_t>(batch.pssm.size()) + pssm_size + padding32;
            const uint32_t total_common = DpuCommunicationManager::alignToMram(new_qmeta_bytes) +
                                          DpuCommunicationManager::alignToMram(new_pssm_bytes);
            if (total_common > limits.max_common_bytes) break;

            QueryMetadata qmeta{};
            qmeta.query_id = static_cast<uint32_t>(batch.meta.size());
            qmeta.query_len = L;
            qmeta.pssm_offset_in_batch = current_pssm_offset;
            qmeta.bias = 0;
            qmeta.pad[0] = qmeta.pad[1] = qmeta.pad[2] = 0;

            int16_t q_min = minScoreThr;
            if (evaluer && par.evalThr != std::numeric_limits<double>::max()) {
                const int rawMin = evaluer->minScore(par.evalThr, L);
                if (rawMin > q_min) q_min = static_cast<int16_t>(rawMin);
            }
            if (q_min < batch.min_score) batch.min_score = q_min;

            batch.meta.push_back(qmeta);
            batch.keys.push_back(queryKey);
            batch.lens.push_back(L);
            batch.qids.push_back(qId);
            batch.pssm.insert(batch.pssm.end(), pssm.begin(), pssm.end());
            batch.pssm.insert(batch.pssm.end(), padding32, 0);

            current_pssm_offset += pssm_size + padding32;
            batch.max_q_len = std::max(batch.max_q_len, L);
            total_residues += L;
            batch.next_q_idx++;
            batch.empty = false;
        }

        if (!batch.empty) {
            const uint32_t qmeta_size = static_cast<uint32_t>(batch.meta.size() * sizeof(QueryMetadata));
            const uint32_t pssm_size = static_cast<uint32_t>(batch.pssm.size());
            const uint32_t meta_aligned = DpuCommunicationManager::alignToMram(qmeta_size);
            const uint32_t total_size = meta_aligned + DpuCommunicationManager::alignToMram(pssm_size);
            batch.common_buffer = std::make_shared<std::vector<uint8_t>>(total_size, 0);
            memcpy(batch.common_buffer->data(), batch.meta.data(), qmeta_size);
            memcpy(batch.common_buffer->data() + meta_aligned, batch.pssm.data(), pssm_size);
            batch.common_size = total_size;
        }

        return batch;
    }

    // --------------------------------------------------------------------------
    // Shared helper: convert DPU hits to Matcher::result_t
    // --------------------------------------------------------------------------
    void DpuPrefilterHostPipeline::processDpuHits(
        const std::vector<std::vector<GappedHit>>& dpu_hits,
        const BatchData& batch,
        DBReader<unsigned int>* tdbr,
        const Parameters& par,
        EvalueComputation* evaluer,
        bool sameDB,
        QueryMatcherTaxonomyHook* taxonomyHook,
        std::vector<std::vector<Matcher::result_t>>& out_results)
    {
        if (batch.keys.empty()) return;
        if (out_results.size() < batch.keys.size()) {
            out_results.resize(batch.keys.size());
        }

        const uint32_t num_dpus = static_cast<uint32_t>(dpu_hits.size());

        for (uint32_t d = 0; d < num_dpus; ++d) {
            for (const auto& hit : dpu_hits[d]) {
                uint16_t q_idx = hit.padding[0];
                if (q_idx >= batch.keys.size()) continue;

                const unsigned int targetKey = tdbr->getDbKey(hit.target_id);
                const uint32_t queryKey = batch.keys[q_idx];
                const uint32_t qLen = batch.lens[q_idx];

                if (taxonomyHook) {
                    TaxID currTax = taxonomyHook->taxonomyMapping->lookup(targetKey);
                    if (!taxonomyHook->expression[0]->isAncestor(currTax)) continue;
                }

                bool isIdentity = (queryKey == targetKey && (par.includeIdentity || sameDB));
                double evalue = 0.0;
                if (par.evalThr != std::numeric_limits<double>::max()) {
                    evalue = evaluer ? evaluer->computeEvalue(hit.score, qLen) : 0.0;
                    if (!isIdentity && evalue > par.evalThr) continue;
                }

                Matcher::result_t res;
                res.dbKey = targetKey;
                res.eval = evalue;
                res.dbEndPos = hit.t_end;
                res.dbLen = tdbr->getSeqLen(hit.target_id);
                res.qEndPos = hit.q_end;
                res.qLen = qLen;
                res.score = evaluer ? static_cast<int>(evaluer->computeBitScore(hit.score) + 0.5) : hit.score;
                res.qStartPos = 0;
                res.dbStartPos = 0;

                res.alnLength = Matcher::computeAlnLength(0, res.qEndPos, 0, res.dbEndPos);
                res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen);
                res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen);

                unsigned int qAlnLen = std::max(static_cast<unsigned int>(res.qEndPos), 1u);
                unsigned int dbAlnLen = std::max(static_cast<unsigned int>(res.dbEndPos), 1u);
                res.seqId = Matcher::estimateSeqIdByScorePerCol(hit.score, qAlnLen, dbAlnLen);

                if (Alignment::checkCriteria(res, isIdentity, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr)) {
                    out_results[q_idx].push_back(res);
                }
            }
        }
    }

    // Forward declaration for tasklet calculation helper (defined in HELPERS section)
    static inline uint8_t calculateActiveTasklets(uint32_t wram_per_tasklet_bytes);

    // ============================================================================
    // HELPER: Partition Targets into MRAM-sized chunks
    // ============================================================================
    static std::vector<std::vector<std::vector<uint32_t>>> partitionTargetsIntoBatches(
        DBReader<unsigned int> *tdbr, uint32_t num_dpus)
    {
        // Rank-aware packing: use bin-packing splitter to even out work per DPU.
        constexpr size_t TARGET_BUDGET_BYTES = 40u * 1024u * 1024u; // leave headroom for descriptor/common/results
        constexpr uint32_t MAX_SEQS_PER_DPU = 16384u;

        auto chunks = DpuDbSplitter::splitDatabase(tdbr, num_dpus, TARGET_BUDGET_BYTES, MAX_SEQS_PER_DPU);
        if (chunks.empty()) {
            return {};
        }

        const size_t num_waves = (chunks.size() + num_dpus - 1) / num_dpus;
        std::vector<std::vector<std::vector<uint32_t>>> targetBatches(num_waves, std::vector<std::vector<uint32_t>>(num_dpus));

        for (size_t idx = 0; idx < chunks.size(); ++idx) {
            const size_t wave = idx / num_dpus;
            const size_t slot = idx % num_dpus;
            targetBatches[wave][slot] = std::move(chunks[idx]);
        }

        return targetBatches;
    }

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

        const int query_seq_type = qdbr->getDbtype();
        const bool is_profile = Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_HMM_PROFILE);
        if (!is_profile && !Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_AMINO_ACIDS)) {
            Debug(Debug::ERROR) << "DPU gapped prefilter supports amino acid or HMM profile queries only\n";
            EXIT(EXIT_FAILURE);
        }

        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::GAPPED);

        const auto rank_sets = dpu_comm_.getRankSets();
        DpuGroupManager group_mgr(rank_sets);
        const uint32_t num_groups = group_mgr.getNumGroups();

        std::vector<std::vector<uint32_t>> group_to_dpu_ids(num_groups);
        uint32_t global_dpu_idx = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            uint32_t nr_dpus_in_rank = 0;
            dpu_error_t status = dpu_get_nr_dpus(rank_sets[g], &nr_dpus_in_rank);
            if (status != DPU_OK) {
                Debug(Debug::ERROR) << "[DPU] Failed to query DPUs in rank " << g << ": " << dpu_error_to_string(status);
                EXIT(EXIT_FAILURE);
            }
            for (uint32_t i = 0; i < nr_dpus_in_rank && global_dpu_idx < num_dpus; ++i) {
                group_to_dpu_ids[g].push_back(global_dpu_idx);
                global_dpu_idx++;
            }
        }

        for (uint32_t d = 0; d < num_dpus; ++d) {
            bool mapped = false;
            for (const auto &v : group_to_dpu_ids) {
                mapped = mapped || std::find(v.begin(), v.end(), d) != v.end();
                if (mapped) break;
            }
            if (!mapped) {
                Debug(Debug::ERROR) << "[DPU] DPU " << d << " not mapped to any rank group";
                EXIT(EXIT_FAILURE);
            }
        }

        auto targetBatches = partitionTargetsIntoBatches(tdbr, num_dpus);
        if (targetBatches.empty()) {
            Debug(Debug::ERROR) << "[DPU] Database splitting failed for gapped path\n";
            return;
        }

        // Storage for accumulated results: [queryIdx] -> list of hits
        std::vector<std::vector<Matcher::result_t>> allResults(qdbr->getSize());

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);

        // DYNAMIC TASKLET CALCULATION
        // Tiled implementation uses fixed WRAM per tasklet regardless of sequence length
        // Stack (~2KB) + Heap (3 * 128 * 4 + 128 + 128 = 1792 bytes) -> ~4KB
        uint32_t wram_per_tasklet = 4096; 
        uint8_t allowed_tasklets = calculateActiveTasklets(wram_per_tasklet);
        const uint8_t DESIRED_TASKLETS = 14;
        uint8_t active_tasklets = std::min<uint8_t>(DESIRED_TASKLETS, allowed_tasklets);

        // Pre-calculate max query sizes for layout
        const uint32_t max_query_len = qdbr->getMaxSeqLen();
        const uint32_t max_pssm_per_query = max_query_len * 32 + 1024;
        const uint32_t max_queries_per_batch = 64u;

        BatchLimits q_limits{};
        q_limits.max_queries = max_queries_per_batch;
        q_limits.max_pssm_bytes = max_pssm_per_query * max_queries_per_batch;
        q_limits.max_common_bytes = DpuCommunicationManager::alignToMram(q_limits.max_queries * static_cast<uint32_t>(sizeof(QueryMetadata))) +
                        DpuCommunicationManager::alignToMram(max_pssm_per_query * q_limits.max_queries);
        
        const uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(GappedBatchDescriptor));

        // Calculate scratch memory needed for tiled implementation
        const uint32_t vec_bytes = DpuCommunicationManager::alignToMram((max_query_len + 1) * sizeof(int16_t));
        const uint32_t scratch_size = (2 * vec_bytes) * active_tasklets;

        Debug(Debug::INFO) << "DPU Gapped Batch Debug:\n"
                           << "  sizeof(GappedBatchDescriptor) = " << sizeof(GappedBatchDescriptor) << "\n"
                           << "  bd_size (aligned) = " << bd_size << "\n"
                           << "  max_query_len = " << max_query_len << "\n"
                           << "  max_common_size = " << q_limits.max_common_bytes << "\n"
                           << "  scratch_size = " << scratch_size << "\n";

        // 2. Loop over Target Batches
        for (size_t bIdx = 0; bIdx < targetBatches.size(); ++bIdx) {
            const auto& perDpuTargetIndices = targetBatches[bIdx];
            std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
            std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

            // Assemble data for this batch
            #pragma omp parallel for schedule(dynamic)
            for (uint32_t d = 0; d < num_dpus; ++d) {
                if (!perDpuTargetIndices[d].empty()) {
                    assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[d],
                                                 perDpuTargetData[d], perDpuTargetMeta[d], subMat);
                }
            }

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);
            std::vector<GappedBatchDescriptor> bds(num_dpus);
            std::vector<uint32_t> maxHitsPerDpu(num_dpus, 0);

            // Calculate fixed layout for parallel transfer
            // MAX_SEQS_PER_DPU = 16384, TARGET_BUDGET_BYTES = 40MB
            uint32_t max_tdata_size = DpuCommunicationManager::alignToMram(40 * 1024 * 1024);
            DpuWorkflow::MramLayout max_layout = workflow_.calculateLayout(
                sizeof(GappedBatchDescriptor), q_limits.max_common_bytes, 16384, q_limits.max_queries, max_tdata_size, sizeof(GappedHit), scratch_size);

            // Prepare descriptors with fixed layout
            for (uint32_t d = 0; d < num_dpus; ++d) {
                layouts[d] = max_layout; // Use fixed layout for everyone

                const uint32_t usable_results = (max_layout.results_capacity > 8u)
                    ? (max_layout.results_capacity - 8u)
                    : 0u;
                maxHitsPerDpu[d] = usable_results / static_cast<uint32_t>(sizeof(GappedHit));
                
                // Initialize descriptor placeholder so initial scatter writes a valid struct.
                // Query-dependent fields (e.g., PSSM offsets) are rewritten in the per-query loop below.
                GappedBatchDescriptor init_bd{};
                init_bd.header.num_targets = perDpuTargetMeta[d].size();
                init_bd.header.num_active_tasklets = active_tasklets;

                // Use maximum-aligned query metadata size for a stable common-data offset.
                const uint32_t qmeta_aligned = DpuCommunicationManager::alignToMram(sizeof(QueryMetadata));
                init_bd.header.queries_metadata_offset = max_layout.common_data_offset;
                init_bd.header.pssm_data_offset = init_bd.header.queries_metadata_offset + qmeta_aligned;
                init_bd.header.targets_metadata_offset = max_layout.target_meta_offset;
                init_bd.header.targets_data_offset = max_layout.target_data_offset;
                init_bd.header.results_offset = max_layout.results_offset;
                init_bd.header.results_buffer_size = max_layout.results_capacity;
                bds[d] = init_bd;
            }

            // Parallel Scatter Targets
            workflow_.scatterBatchParallel(bds, perDpuTargetMeta, perDpuTargetData, max_layout);

            RankDispatcher dispatcher(group_mgr, group_to_dpu_ids);

            std::vector<std::vector<GappedHit>> dpu_hits(num_dpus);
            std::vector<bool> dpu_active(num_dpus, false);
            std::vector<bool> dpu_ready(num_dpus, false);

            size_t q_cursor = 0;
            while (q_cursor < qdbr->getSize()) {
                auto batch = buildQueryBatch(q_cursor, qdbr, subMat, par, compBias, q_limits, evaluer, static_cast<int16_t>(par.minDiagScoreThr), nullptr);
                if (batch.empty) {
                    Debug(Debug::ERROR) << "[DPU] Gapped batch could not fit any query at cursor " << q_cursor;
                    break;
                }
                q_cursor = batch.next_q_idx;

                for (auto &v : dpu_hits) v.clear();

                workflow_.broadcastCommon(batch.common_buffer->data(), batch.common_size, max_layout.common_data_offset);

                std::vector<std::vector<uint8_t>> bd_bufs(num_dpus);
                const uint32_t qmeta_size = static_cast<uint32_t>(batch.meta.size() * sizeof(QueryMetadata));
                const uint32_t pssm_size = static_cast<uint32_t>(batch.pssm.size());
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (perDpuTargetMeta[d].empty()) continue;

                    const uint32_t qmeta_off = max_layout.common_data_offset;
                    const uint32_t pssm_off = qmeta_off + DpuCommunicationManager::alignToMram(qmeta_size);
                    const uint32_t flags = (par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE) ? 1u : 0u;

                    DpuBatchHeader hdr(
                        static_cast<uint32_t>(batch.meta.size()),
                        static_cast<uint32_t>(perDpuTargetMeta[d].size()),
                        batch.max_q_len,
                        qmeta_off,
                        pssm_off,
                        max_layout.target_meta_offset,
                        max_layout.target_data_offset,
                        max_layout.results_offset,
                        max_layout.results_capacity,
                        pssm_size,
                        static_cast<uint32_t>(perDpuTargetData[d].size()),
                        static_cast<uint16_t>(flags),
                        active_tasklets);
                    hdr.batch_id = 0;

                    GappedBatchDescriptor bd(
                        hdr,
                        batch.min_score,
                        static_cast<int16_t>(par.gapOpen.values.aminoacid()),
                        static_cast<int16_t>(par.gapExtend.values.aminoacid()),
                        static_cast<int16_t>(par.zdrop),
                        0,
                        static_cast<uint8_t>(par.covMode),
                        static_cast<uint8_t>(par.covThr * 100.0f),
                        static_cast<uint8_t>(std::min(par.alnLenThr, 255)),
                        static_cast<uint8_t>(par.seqIdThr * 100.0f));

                    bd_bufs[d].resize(sizeof(GappedBatchDescriptor));
                    memcpy(bd_bufs[d].data(), &bd, sizeof(GappedBatchDescriptor));
                }

                dpu_comm_.scatterDataParallel(bd_bufs, 0);

                uint64_t zero_hdr = 0;
                dpu_comm_.broadcastData(&zero_hdr, 8, max_layout.results_offset);

                std::fill(dpu_active.begin(), dpu_active.end(), false);
                std::fill(dpu_ready.begin(), dpu_ready.end(), false);
                size_t inflight_groups = 0;
                for (uint32_t gid = 0; gid < num_groups; ++gid) {
                    bool launched = dispatcher.launchGroup(gid, [&](uint32_t d) {
                        if (d >= perDpuTargetMeta.size()) return false;
                        if (perDpuTargetMeta[d].empty()) return false;
                        dpu_active[d] = true;
                        return true;
                    });
                    if (launched) inflight_groups++;
                }

                kernel_mgr_.loadKernel(DpuKernelManager::KernelType::GAPPED);

                while (inflight_groups > 0) {
                    size_t drained_groups = dispatcher.drainCompleted([&](uint32_t d) {
                        if (!dpu_active[d]) return;
                        dpu_ready[d] = true;
                    });
                    if (drained_groups == 0) {
                        dispatcher.poll();
                        usleep(100);
                    } else {
                        inflight_groups -= drained_groups;

                        std::vector<std::vector<uint8_t>> header_bufs(num_dpus);
                        for (uint32_t d = 0; d < num_dpus; ++d) {
                            if (dpu_ready[d]) header_bufs[d].resize(8);
                        }
                        dpu_comm_.gatherDataParallel(header_bufs, 8, layouts[0].results_offset);

                        for (uint32_t d = 0; d < num_dpus; ++d) {
                            if (!dpu_ready[d]) continue;

                            uint64_t hdr = 0;
                            memcpy(&hdr, header_bufs[d].data(), 8);
                            const uint32_t hit_count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
                            const uint32_t max_hits = maxHitsPerDpu[d];
                            if (hit_count > max_hits) {
                                Debug(Debug::ERROR) << "[DPU] Gapped batch overflow detected for DPU " << d
                                                    << " (hits=" << hit_count << ", max=" << max_hits << ")";
                                EXIT(EXIT_FAILURE);
                            }
                            if (hit_count > 0) {
                                const uint32_t data_size = hit_count * static_cast<uint32_t>(sizeof(GappedHit));
                                const uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);
                                std::vector<GappedHit> hits(hit_count);
                                if (aligned_size != data_size) {
                                    std::vector<uint8_t> buf(aligned_size);
                                    dpu_comm_.gatherDataFromDPU(d, buf.data(), aligned_size, layouts[d].results_offset + 8);
                                    memcpy(hits.data(), buf.data(), data_size);
                                } else {
                                    dpu_comm_.gatherDataFromDPU(d, hits.data(), aligned_size, layouts[d].results_offset + 8);
                                }
                                dpu_hits[d] = std::move(hits);
                            } else {
                                dpu_hits[d].clear();
                            }

                            dpu_active[d] = false;
                            dpu_ready[d] = false;
                        }
                    }
                }

                std::vector<std::vector<Matcher::result_t>> batch_results(batch.meta.size());
                processDpuHits(dpu_hits, batch, tdbr, par, evaluer, sameDB, taxonomyHook, batch_results);

                for (size_t i = 0; i < batch_results.size(); ++i) {
                    if (batch_results[i].empty()) continue;
                    const size_t q_global = batch.qids[i];
                    allResults[q_global].insert(allResults[q_global].end(), batch_results[i].begin(), batch_results[i].end());
                }
            }

        } // End Target Batch Loop

        // 3. Write Results
        for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
            auto& resultsAln = allResults[qId];
            if (!resultsAln.empty()) {
                SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);
                size_t maxSeqs = std::min((size_t)par.maxResListLen, resultsAln.size());
                std::string resultBuffer;
                for (size_t i = 0; i < maxSeqs; ++i) {
                    char outbuf[4096];
                    size_t len = Matcher::resultToBuffer(outbuf, resultsAln[i], false);
                    resultBuffer.append(outbuf, len);
                }
                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), qdbr->getDbKey(qId), 0);
            }
        }

        if (dpu_comm_.isProfilingEnabled()) {
            dpu_comm_.dumpProfile("gapped_prefilter");
            dpu_comm_.resetProfile();
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

        const int query_seq_type = qdbr->getDbtype();
        const bool is_profile = Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_HMM_PROFILE);
        if (!is_profile && !Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_AMINO_ACIDS)) {
            Debug(Debug::ERROR) << "DPU ungapped prefilter supports amino acid or HMM profile queries only\n";
            EXIT(EXIT_FAILURE);
        }

        auto perDpuTargetIndices = buildLoadBalancedDistribution(tdbr, num_dpus);
        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);
        int16_t minScoreThr = static_cast<int16_t>(par.minDiagScoreThr);

        const auto rank_sets = dpu_comm_.getRankSets();
        DpuGroupManager group_mgr(rank_sets);
        const uint32_t num_groups = group_mgr.getNumGroups();

        std::vector<std::vector<uint32_t>> group_to_dpu_ids(num_groups);
        std::vector<uint32_t> dpu_to_group(num_dpus, UINT32_MAX);
        uint32_t global_dpu_idx = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            uint32_t nr_dpus_in_rank = 0;
            dpu_error_t status = dpu_get_nr_dpus(rank_sets[g], &nr_dpus_in_rank);
            if (status != DPU_OK) {
                Debug(Debug::ERROR) << "[DPU] Failed to query DPUs in rank " << g << ": " << dpu_error_to_string(status);
                EXIT(EXIT_FAILURE);
            }
            for (uint32_t i = 0; i < nr_dpus_in_rank && global_dpu_idx < num_dpus; ++i) {
                group_to_dpu_ids[g].push_back(global_dpu_idx);
                dpu_to_group[global_dpu_idx] = g;
                global_dpu_idx++;
            }
        }

        for (uint32_t d = 0; d < num_dpus; ++d) {
            if (dpu_to_group[d] == UINT32_MAX) {
                Debug(Debug::ERROR) << "[DPU] DPU " << d << " not mapped to any rank group";
                EXIT(EXIT_FAILURE);
            }
        }

        RankDispatcher dispatcher(group_mgr, group_to_dpu_ids);

        struct TargetChunk {
            bool valid = false;
            size_t count = 0;
            std::vector<uint32_t> indices;
            std::vector<uint8_t> data;
            std::vector<TargetMetadata> meta;
            DpuWorkflow::MramLayout layout{};
        };

        const uint32_t UNGAPPED_RESULTS_BYTES = 8 * 1024 * 1024; // fixed results buffer

        auto buildChunkForDpu = [&](uint32_t dpu_id, size_t cursor, uint32_t query_count, uint32_t common_size) {
            TargetChunk chunk;
            const auto &indices = perDpuTargetIndices[dpu_id];
            if (cursor >= indices.size()) return chunk;

            uint64_t tdata_bytes = 0;
            uint32_t num_targets = 0;
            DpuWorkflow::MramLayout last_layout{};

            while (cursor + num_targets < indices.size()) {
                uint32_t tid = indices[cursor + num_targets];
                uint32_t tlen = tdbr->getSeqLen(tid);
                uint32_t padded = DpuCommunicationManager::alignToMram(tlen);
                uint64_t candidate_data = tdata_bytes + padded;
                uint32_t candidate_targets = num_targets + 1;

                // Manual layout to force fixed results capacity and keep MRAM accounting consistent.
                const uint32_t bd_size = DpuCommunicationManager::alignToMram(static_cast<uint32_t>(sizeof(UngappedBatchDescriptor)));
                DpuWorkflow::MramLayout layout{};
                layout.common_data_offset = bd_size;
                layout.target_meta_offset = layout.common_data_offset + DpuCommunicationManager::alignToMram(common_size);
                layout.target_data_offset = layout.target_meta_offset + DpuCommunicationManager::alignToMram(candidate_targets * static_cast<uint32_t>(sizeof(TargetMetadata)));
                layout.results_offset     = layout.target_data_offset + DpuCommunicationManager::alignToMram(static_cast<uint32_t>(candidate_data));
                layout.results_capacity   = DpuCommunicationManager::alignToMram(UNGAPPED_RESULTS_BYTES);
                layout.total_mram_used    = layout.results_offset + layout.results_capacity;

                if (layout.total_mram_used > DPU_MRAM_TOTAL_SIZE) {
                    break;
                }

                last_layout = layout;
                tdata_bytes = candidate_data;
                num_targets = candidate_targets;
            }

            if (num_targets == 0) {
                return chunk;
            }

            chunk.indices.insert(chunk.indices.end(), indices.begin() + cursor, indices.begin() + cursor + num_targets);
            assembleTargetBatchByIndices(tdbr, chunk.indices, chunk.data, chunk.meta, subMat);
            chunk.layout = last_layout;
            chunk.count = num_targets;
            chunk.valid = true;
            return chunk;
        };

        // Load kernel once; optionally reload before each launch when simulator requires a full reset.
        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::UNGAPPED);

        std::vector<uint32_t> active_dpus;
        for (uint32_t d = 0; d < num_dpus; ++d) {
            if (!perDpuTargetIndices[d].empty()) active_dpus.push_back(d);
        }

        size_t qId = 0;
        while (qId < qdbr->getSize())
        {
            // -----------------
            // Build Current Batch (single-query for ungapped path)
            // -----------------
            std::vector<QueryMetadata> batchMeta;
            std::vector<uint32_t> batchQueryKeys;
            uint32_t qmeta_block_size = 0;
            uint32_t common_size = 0;
            uint32_t max_q_len = 0;

            {
                const uint32_t queryKey = qdbr->getDbKey(qId);
                const uint32_t queryLen = qdbr->getSeqLen(qId);
                const char *querySeq = qdbr->getData(qId, 0);

                auto pssm = is_profile
                    ? extractPSSMFromProfile(querySeq, queryLen, subMat)
                    : buildPSSMFromSequence(querySeq, queryLen, subMat, par.compBiasCorrection, par.compBiasCorrectionScale, compBias);

                int8_t min_val = 127;
                for (int8_t v : pssm) {
                    if (v < min_val) min_val = v;
                }
                uint8_t bias = (min_val < 0) ? static_cast<uint8_t>(-min_val) : 0;

                std::vector<uint8_t> biased_pssm;
                biased_pssm.reserve(pssm.size());
                for (int8_t v : pssm) {
                    biased_pssm.push_back(static_cast<uint8_t>(v + bias));
                }

                QueryMetadata qmeta{};
                qmeta.query_id = 0;
                qmeta.query_len = queryLen;
                qmeta.pssm_offset_in_batch = 0;
                qmeta.bias = bias;
                qmeta.pad[0] = qmeta.pad[1] = qmeta.pad[2] = 0;

                batchMeta.push_back(qmeta);
                batchQueryKeys.push_back(queryKey);

                qmeta_block_size = static_cast<uint32_t>(batchMeta.size() * sizeof(QueryMetadata));
                const uint32_t pssm_block_size = static_cast<uint32_t>(biased_pssm.size());
                common_size = DpuCommunicationManager::alignToMram(qmeta_block_size) +
                              DpuCommunicationManager::alignToMram(pssm_block_size);
                max_q_len = queryLen;

                std::vector<uint8_t> commonData(common_size, 0);
                memcpy(commonData.data(), batchMeta.data(), qmeta_block_size);
                memcpy(commonData.data() + DpuCommunicationManager::alignToMram(qmeta_block_size), biased_pssm.data(), pssm_block_size);

                const uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(UngappedBatchDescriptor));
                workflow_.broadcastCommon(commonData.data(), common_size, bd_size);
            }

            qId++;

            // -----------------
            // Per-DPU target chunk streaming with group-managed async scheduling
            // -----------------
            if (!active_dpus.empty()) {
                std::vector<size_t> target_cursor(num_dpus, 0);
                std::vector<bool> dpu_done(num_dpus, false);
                std::vector<bool> dpu_active(num_dpus, false);
                std::vector<DpuWorkflow::MramLayout> active_layouts(num_dpus);
                std::vector<uint32_t> active_chunk_counts(num_dpus, 0);
                std::vector<std::vector<hit_t>> resultsByQuery(batchQueryKeys.size());
                auto gatherUngappedChecked = [&](uint32_t dpu_id, const DpuWorkflow::MramLayout& layout) {
                    uint64_t hdr = 0;
                    dpu_comm_.gatherDataFromDPU(dpu_id, &hdr, 8, layout.results_offset);
                    const uint32_t hit_count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
                    const uint32_t max_hits = (layout.results_capacity > 8u)
                                                  ? (layout.results_capacity - 8u) / static_cast<uint32_t>(sizeof(Hit))
                                                  : 0u;
                    if (hit_count > max_hits) {
                        Debug(Debug::ERROR) << "[DPU] Ungapped batch overflow detected for DPU " << dpu_id
                                            << " (hits=" << hit_count << ", max=" << max_hits << ")";
                        EXIT(EXIT_FAILURE);
                    }
                    if (hit_count == 0) return std::vector<Hit>{};

                    const uint32_t data_size = hit_count * static_cast<uint32_t>(sizeof(Hit));
                    const uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);
                    std::vector<Hit> hits(hit_count);
                    if (aligned_size != data_size) {
                        std::vector<uint8_t> buf(aligned_size);
                        dpu_comm_.gatherDataFromDPU(dpu_id, buf.data(), aligned_size, layout.results_offset + 8);
                        memcpy(hits.data(), buf.data(), data_size);
                    } else {
                        dpu_comm_.gatherDataFromDPU(dpu_id, hits.data(), aligned_size, layout.results_offset + 8);
                    }
                    return hits;
                };

                size_t completed = 0;
                const size_t active_total = active_dpus.size();

                while (completed < active_total) {
                    size_t drained = dispatcher.drainCompleted([&](uint32_t d) {
                        if (!dpu_active[d]) return;

                        auto hits = gatherUngappedChecked(d, active_layouts[d]);
                        for (const auto &hit : hits) {
                            if (hit.query_id >= batchQueryKeys.size()) continue;
                            hit_t res;
                            res.seqId = tdbr->getDbKey(hit.target_id);
                            res.prefScore = hit.score;
                            res.diagonal = hit.diagonal;
                            resultsByQuery[hit.query_id].push_back(res);
                        }

                        target_cursor[d] += active_chunk_counts[d];
                        dpu_active[d] = false;
                        if (target_cursor[d] >= perDpuTargetIndices[d].size() && !dpu_done[d]) {
                            dpu_done[d] = true;
                            completed++;
                        }
                    });

                    bool launched = false;
                    for (uint32_t gid = 0; gid < num_groups; ++gid) {
                        bool launched_group = dispatcher.launchGroup(gid, [&](uint32_t d) {
                            if (perDpuTargetIndices[d].empty()) {
                                dpu_done[d] = true;
                                return false;
                            }

                            if (dpu_done[d] || dpu_active[d]) return false;
                            if (target_cursor[d] >= perDpuTargetIndices[d].size()) {
                                if (!dpu_done[d]) {
                                    dpu_done[d] = true;
                                    completed++;
                                }
                                return false;
                            }

                            auto chunk = buildChunkForDpu(d, target_cursor[d], static_cast<uint32_t>(batchMeta.size()), common_size);
                            if (!chunk.valid) {
                                Debug(Debug::ERROR) << "[DPU] Unable to fit any targets for DPU " << d << " in current batch";
                                dpu_done[d] = true;
                                completed++;
                                return false;
                            }

                            UngappedBatchDescriptor bd = {};
                            bd.header.num_queries = batchMeta.size();
                            bd.header.num_targets = static_cast<uint32_t>(chunk.count);
                            bd.header.query_len = max_q_len;
                            bd.header.queries_metadata_offset = chunk.layout.common_data_offset;
                            bd.header.pssm_data_offset = bd.header.queries_metadata_offset + DpuCommunicationManager::alignToMram(qmeta_block_size);
                            bd.header.targets_metadata_offset = chunk.layout.target_meta_offset;
                            bd.header.targets_data_offset = chunk.layout.target_data_offset;
                            bd.header.results_offset = chunk.layout.results_offset;
                            bd.header.results_buffer_size = chunk.layout.results_capacity;
                            bd.min_score = minScoreThr;
                            {
                                uint32_t batch_max_q = max_q_len;
                                const uint32_t MAX_TARGET_WRAM_LEN_HOST = 6144;
                                uint32_t diag_bytes = 2 * (MAX_TARGET_WRAM_LEN_HOST + batch_max_q);
                                uint32_t wramPerTasklet = MAX_TARGET_WRAM_LEN_HOST + diag_bytes + 2048;
                                bd.header.num_active_tasklets = std::min<uint8_t>(14, calculateActiveTasklets(wramPerTasklet));
                            }
                            bd.gap_open_cost = static_cast<int16_t>(par.gapOpen.values.aminoacid());
                            bd.gap_extend_cost = static_cast<int16_t>(par.gapExtend.values.aminoacid());
                            bd.pssm_bias = 0;

                            workflow_.scatterBatch(d, bd, chunk.meta, chunk.data, chunk.layout);

                            uint64_t zero_hdr = 0;
                            dpu_comm_.scatterDataToDPU(d, &zero_hdr, sizeof(uint64_t), chunk.layout.results_offset);

                            dpu_active[d] = true;
                            active_layouts[d] = chunk.layout;
                            active_chunk_counts[d] = static_cast<uint32_t>(chunk.count);
                            return true;
                        });

                        if (launched_group) {
                            launched = true;
                        }
                    }

                    if (!launched && drained == 0) {
                        dispatcher.poll();
                        usleep(500);
                    }
                }

                // -----------------
                // Emit results for this batch
                // -----------------
                for (size_t i = 0; i < batchQueryKeys.size(); ++i)
                {
                    if (!resultsByQuery[i].empty())
                    {
                        std::sort(resultsByQuery[i].begin(), resultsByQuery[i].end(), hit_t::compareHitsByScoreAndId);
                        std::string resultBuffer;
                        size_t keep = std::min(resultsByQuery[i].size(), (size_t)par.maxResListLen);
                        for (size_t k = 0; k < keep; ++k)
                        {
                            char buffer[256];
                            size_t len = QueryMatcher::prefilterHitToBuffer(buffer, resultsByQuery[i][k]);
                            resultBuffer.append(buffer, len);
                        }
                        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), batchQueryKeys[i], 0);
                    }
                }
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
        const bool is_profile = Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_HMM_PROFILE);
        if (!is_profile && !Parameters::isEqualDbtype(query_seq_type, Parameters::DBTYPE_AMINO_ACIDS)) {
            Debug(Debug::ERROR) << "DPU ungapped+gapped supports amino acid or HMM profile queries only\n";
            EXIT(EXIT_FAILURE);
        }

        // DYNAMIC TASKLET CALCULATION
        // Reduced WRAM usage allows for more tasklets (approx 4KB per tasklet)
        uint32_t wram_per_tasklet = 4000u;
        uint8_t allowed_tasklets_comb = calculateActiveTasklets(wram_per_tasklet); 
        const uint8_t DESIRED_TASKLETS = 14;
        uint8_t tasklet_limit = std::min<uint8_t>(DESIRED_TASKLETS, allowed_tasklets_comb);

        struct CombinedLimits {
            uint32_t max_batch_queries;
            uint32_t max_pssm_bytes;
            uint32_t combined_results_bytes;
            uint32_t result_header_bytes;
            uint32_t reserved_qmeta;
            uint32_t reserved_pssm;
            uint32_t reserved_common;
            uint8_t tasklet_limit;
        };

        auto align32 = [](uint32_t v) { return (v + 31u) & ~31u; };

        auto makeLimits = [&](uint32_t max_queries, uint32_t max_pssm) {
            CombinedLimits lim{};
            lim.max_batch_queries = max_queries;
            lim.max_pssm_bytes = max_pssm;
            lim.combined_results_bytes = 16u * 1024u * 1024u; // 16 MB fixed results buffer per DPU
            lim.result_header_bytes = 8u; // count + overflow flag
            lim.reserved_qmeta = DpuCommunicationManager::alignToMram(max_queries * static_cast<uint32_t>(sizeof(QueryMetadata)));
            lim.reserved_pssm = align32(max_pssm);
            lim.reserved_common = DpuCommunicationManager::alignToMram(lim.reserved_qmeta + lim.reserved_pssm);
            lim.tasklet_limit = tasklet_limit;
            return lim;
        };

        const CombinedLimits limits = makeLimits(32u, 4u * 1024u * 1024u);

        auto targetBatches = partitionTargetsIntoBatches(tdbr, num_dpus);
        bool accumulate_results = (targetBatches.size() > 1);
        std::vector<std::vector<Matcher::result_t>> allResults;
        if (accumulate_results) {
            allResults.resize(qdbr->getSize());
        }

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);
        int16_t minUngappedThr = static_cast<int16_t>(par.minDiagScoreThr);
        // Sequence qSeq(par.maxSeqLen, query_seq_type, subMat, 0, false, par.compBiasCorrection);

        const auto rank_sets = dpu_comm_.getRankSets();
        DpuGroupManager group_mgr(rank_sets);
        const uint32_t num_groups = group_mgr.getNumGroups();

        std::vector<std::vector<uint32_t>> group_to_dpu_ids(num_groups);
        std::vector<uint32_t> dpu_to_group(num_dpus, UINT32_MAX);
        uint32_t global_dpu_idx = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            uint32_t nr_dpus_in_rank = 0;
            dpu_error_t status = dpu_get_nr_dpus(rank_sets[g], &nr_dpus_in_rank);
            if (status != DPU_OK) {
                Debug(Debug::ERROR) << "[DPU] Failed to query DPUs in rank " << g << ": " << dpu_error_to_string(status);
                EXIT(EXIT_FAILURE);
            }
            for (uint32_t i = 0; i < nr_dpus_in_rank && global_dpu_idx < num_dpus; ++i) {
                group_to_dpu_ids[g].push_back(global_dpu_idx);
                dpu_to_group[global_dpu_idx] = g;
                global_dpu_idx++;
            }
        }

        for (uint32_t d = 0; d < num_dpus; ++d) {
            if (dpu_to_group[d] == UINT32_MAX) {
                Debug(Debug::ERROR) << "[DPU] DPU " << d << " not mapped to any rank group";
                EXIT(EXIT_FAILURE);
            }
        }

        RankDispatcher dispatcher(group_mgr, group_to_dpu_ids);

        for (size_t bIdx = 0; bIdx < targetBatches.size(); ++bIdx) {
            const auto& perDpuTargetIndices = targetBatches[bIdx];
            std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
            std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

            // Ensure kernel is loaded before sending any data to DPUs in this target batch
            kernel_mgr_.loadKernel(DpuKernelManager::KernelType::COMBINED);

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

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);
            std::vector<uint32_t> maxHitsPerDpu(num_dpus, 0);
            std::vector<uint32_t> active_dpus;
            for (uint32_t d = 0; d < num_dpus; ++d) {
                const uint32_t bd_size = DpuCommunicationManager::alignToMram(static_cast<uint32_t>(sizeof(CombinedBatchDescriptor)));
                const uint32_t tmeta_size = DpuCommunicationManager::alignToMram(static_cast<uint32_t>(perDpuTargetMeta[d].size() * sizeof(TargetMetadata)));
                const uint32_t tdata_size = DpuCommunicationManager::alignToMram(static_cast<uint32_t>(perDpuTargetData[d].size()));

                layouts[d].common_data_offset = bd_size;
                layouts[d].target_meta_offset = layouts[d].common_data_offset + limits.reserved_common;
                layouts[d].target_data_offset = layouts[d].target_meta_offset + tmeta_size;
                layouts[d].results_offset     = layouts[d].target_data_offset + tdata_size;
                layouts[d].results_capacity   = DpuCommunicationManager::alignToMram(limits.combined_results_bytes);
                layouts[d].total_mram_used    = layouts[d].results_offset + layouts[d].results_capacity;

                const uint32_t usable_results = (layouts[d].results_capacity > limits.result_header_bytes)
                    ? (layouts[d].results_capacity - limits.result_header_bytes)
                    : 0u;
                maxHitsPerDpu[d] = usable_results / static_cast<uint32_t>(sizeof(GappedHit));

                if (!perDpuTargetMeta[d].empty()) {
                    active_dpus.push_back(d);
                }

                if (layouts[d].total_mram_used > DPU_MRAM_TOTAL_SIZE) {
                    Debug(Debug::ERROR) << "[DPU] Combined layout exceeds MRAM for DPU " << d << " (" << layouts[d].total_mram_used << " > " << DPU_MRAM_TOTAL_SIZE << ")";
                    EXIT(EXIT_FAILURE);
                }
            }

            size_t max_batch_queries_limit = limits.max_batch_queries;
            {
                uint32_t cap_by_results = limits.max_batch_queries;
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (perDpuTargetMeta[d].empty()) continue;
                    const uint32_t targets = static_cast<uint32_t>(perDpuTargetMeta[d].size());
                    const uint32_t max_hits = (layouts[d].results_capacity > limits.result_header_bytes)
                        ? (layouts[d].results_capacity - limits.result_header_bytes) / static_cast<uint32_t>(sizeof(GappedHit))
                        : 0u;
                    const uint32_t qcap = (targets == 0 || max_hits == 0) ? 1u : std::max<uint32_t>(1u, max_hits / targets);
                    cap_by_results = std::min(cap_by_results, qcap);
                }
                max_batch_queries_limit = std::max<size_t>(1u, std::min<size_t>(max_batch_queries_limit, cap_by_results));
            }

            // Upload target metadata and data once per target batch (descriptor will be scattered per query batch)
            for (uint32_t d = 0; d < num_dpus; ++d) {
                if (perDpuTargetMeta[d].empty()) continue;

                uint32_t tmeta_size_bytes = static_cast<uint32_t>(perDpuTargetMeta[d].size() * sizeof(TargetMetadata));
                uint32_t tmeta_aligned = DpuCommunicationManager::alignToMram(tmeta_size_bytes);
                if (tmeta_size_bytes == tmeta_aligned) {
                    dpu_comm_.scatterDataToDPU(d, perDpuTargetMeta[d].data(), tmeta_aligned, layouts[d].target_meta_offset);
                } else {
                    std::vector<uint8_t> buf(tmeta_aligned, 0);
                    memcpy(buf.data(), perDpuTargetMeta[d].data(), tmeta_size_bytes);
                    dpu_comm_.scatterDataToDPU(d, buf.data(), tmeta_aligned, layouts[d].target_meta_offset);
                }

                uint32_t tdata_size_bytes = static_cast<uint32_t>(perDpuTargetData[d].size());
                uint32_t tdata_aligned = DpuCommunicationManager::alignToMram(tdata_size_bytes);
                if (tdata_size_bytes == tdata_aligned) {
                    dpu_comm_.scatterDataToDPU(d, perDpuTargetData[d].data(), tdata_aligned, layouts[d].target_data_offset);
                } else {
                    std::vector<uint8_t> buf(tdata_aligned, 0);
                    memcpy(buf.data(), perDpuTargetData[d].data(), tdata_size_bytes);
                    dpu_comm_.scatterDataToDPU(d, buf.data(), tdata_aligned, layouts[d].target_data_offset);
                }
            }

            auto gatherChecked = [&](uint32_t dpu_id, const DpuWorkflow::MramLayout& layout) {
                uint64_t hdr = 0;
                dpu_comm_.gatherDataFromDPU(dpu_id, &hdr, 8, layout.results_offset);
                const uint32_t hit_count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
                const uint32_t overflow_flag = static_cast<uint32_t>(hdr >> 32);
                const uint32_t max_hits = maxHitsPerDpu[dpu_id];
                if (overflow_flag != 0) {
                    Debug(Debug::ERROR) << "[DPU] Combined batch overflow flag set on DPU " << dpu_id
                                        << " (hits=" << hit_count << ", capacity=" << max_hits << ")";
                }
                if (hit_count > max_hits) {
                    Debug(Debug::ERROR) << "[DPU] Combined batch overflow detected for DPU " << dpu_id
                                        << " (hits=" << hit_count << ", max=" << max_hits << ")";
                    EXIT(EXIT_FAILURE);
                }
                if (hit_count == 0) return std::vector<GappedHit>{};

                const uint32_t data_size = hit_count * static_cast<uint32_t>(sizeof(GappedHit));
                const uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);
                std::vector<GappedHit> hits(hit_count);
                if (aligned_size != data_size) {
                    std::vector<uint8_t> buf(aligned_size);
                    dpu_comm_.gatherDataFromDPU(dpu_id, buf.data(), aligned_size, layout.results_offset + 8);
                    memcpy(hits.data(), buf.data(), data_size);
                } else {
                    dpu_comm_.gatherDataFromDPU(dpu_id, hits.data(), aligned_size, layout.results_offset + 8);
                }
                return hits;
            };

            BatchLimits batch_limits{};
            batch_limits.max_queries = static_cast<uint32_t>(max_batch_queries_limit);
            batch_limits.max_pssm_bytes = limits.max_pssm_bytes;
            batch_limits.max_common_bytes = limits.reserved_common;

            std::vector<std::vector<GappedHit>> dpu_hits(num_dpus);
            std::vector<bool> dpu_active(num_dpus, false);
            std::vector<bool> dpu_ready(num_dpus, false);

            size_t q_cursor = 0;
            while (q_cursor < qdbr->getSize()) {
                auto batch = buildQueryBatch(q_cursor, qdbr, subMat, par, compBias, batch_limits, evaluer, minUngappedThr, nullptr);
                if (batch.empty) {
                    Debug(Debug::ERROR) << "[DPU] Combined batch could not fit any query at cursor " << q_cursor;
                    break;
                }
                q_cursor = batch.next_q_idx;

                for (auto &v : dpu_hits) v.clear();

                workflow_.broadcastCommon(batch.common_buffer->data(), batch.common_size, layouts[0].common_data_offset);

                std::vector<std::vector<uint8_t>> bd_bufs(num_dpus);
                const uint32_t qmeta_size = static_cast<uint32_t>(batch.meta.size() * sizeof(QueryMetadata));
                const uint32_t pssm_size = static_cast<uint32_t>(batch.pssm.size());

                for (uint32_t d : active_dpus) {
                    const uint32_t qmeta_off = layouts[d].common_data_offset;
                    const uint32_t pssm_off = qmeta_off + DpuCommunicationManager::alignToMram(qmeta_size);

                    DpuBatchHeader hdr(
                        static_cast<uint32_t>(batch.meta.size()),
                        static_cast<uint32_t>(perDpuTargetMeta[d].size()),
                        batch.max_q_len,
                        qmeta_off,
                        pssm_off,
                        layouts[d].target_meta_offset,
                        layouts[d].target_data_offset,
                        layouts[d].results_offset,
                        layouts[d].results_capacity,
                        pssm_size,
                        static_cast<uint32_t>(perDpuTargetData[d].size()),
                        0,
                        limits.tasklet_limit);
                    hdr.batch_id = 0;

                    CombinedBatchDescriptor bd(
                        hdr,
                        minUngappedThr,
                        batch.min_score,
                        static_cast<int16_t>(par.gapOpen.values.aminoacid()),
                        static_cast<int16_t>(par.gapExtend.values.aminoacid()),
                        static_cast<int16_t>(par.zdrop),
                        0,
                        static_cast<uint8_t>(par.covMode),
                        static_cast<uint8_t>(par.covThr * 100.0f),
                        static_cast<uint8_t>(std::min(par.alnLenThr, 255)),
                        static_cast<uint8_t>(par.seqIdThr * 100.0f));

                    bd_bufs[d].resize(sizeof(CombinedBatchDescriptor));
                    memcpy(bd_bufs[d].data(), &bd, sizeof(CombinedBatchDescriptor));
                }

                dpu_comm_.scatterDataParallel(bd_bufs, 0);

                uint64_t zero_hdr = 0;
                dpu_comm_.broadcastData(&zero_hdr, 8, layouts[0].results_offset);

                std::fill(dpu_active.begin(), dpu_active.end(), false);
                std::fill(dpu_ready.begin(), dpu_ready.end(), false);
                size_t inflight_groups = 0;
                for (uint32_t gid = 0; gid < num_groups; ++gid) {
                    bool launched = dispatcher.launchGroup(gid, [&](uint32_t d) {
                        if (d >= perDpuTargetMeta.size()) return false;
                        if (perDpuTargetMeta[d].empty()) return false;
                        dpu_active[d] = true;
                        return true;
                    });
                    if (launched) inflight_groups++;
                }

                while (inflight_groups > 0) {
                    size_t drained = dispatcher.drainCompleted([&](uint32_t d) {
                        if (!dpu_active[d]) return;
                        dpu_ready[d] = true;
                    });

                    if (drained == 0) {
                        dispatcher.poll();
                        usleep(100);
                    } else {
                        inflight_groups -= drained;
                        for (uint32_t d = 0; d < num_dpus; ++d) {
                            if (!dpu_ready[d]) continue;
                            dpu_hits[d] = gatherChecked(d, layouts[d]);
                            dpu_ready[d] = false;
                            dpu_active[d] = false;
                        }
                    }
                }

                std::vector<std::vector<Matcher::result_t>> batch_results(batch.meta.size());
                processDpuHits(dpu_hits, batch, tdbr, par, evaluer, sameDB, taxonomyHook, batch_results);

                for (size_t i = 0; i < batch_results.size(); ++i) {
                    if (batch_results[i].empty()) continue;

                    if (accumulate_results) {
                        const size_t q_global = batch.qids[i];
                        allResults[q_global].insert(allResults[q_global].end(), batch_results[i].begin(), batch_results[i].end());
                    } else {
                        SORT_PARALLEL(batch_results[i].begin(), batch_results[i].end(), Matcher::compareHits);
                        const size_t maxSeqs = std::min<size_t>(par.maxResListLen, batch_results[i].size());
                        std::string resultBuffer;
                        resultBuffer.reserve(262144);
                        for (size_t k = 0; k < maxSeqs; k++) {
                            char outbuf[4096];
                            const size_t len = Matcher::resultToBuffer(outbuf, batch_results[i][k], false);
                            resultBuffer.append(outbuf, len);
                        }
                        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), batch.keys[i], 0);
                    }
                }
            }
        }

        if (accumulate_results) {
            for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
                auto& resultsAln = allResults[qId];
                if (!resultsAln.empty()) {
                    SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);
                    size_t maxSeqs = std::min((size_t)par.maxResListLen, resultsAln.size());
                    std::string resultBuffer;
                    resultBuffer.reserve(262144);
                    for (size_t i = 0; i < maxSeqs; ++i) {
                        char outbuf[4096];
                        size_t len = Matcher::resultToBuffer(outbuf, resultsAln[i], false);
                        resultBuffer.append(outbuf, len);
                    }
                    resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), qdbr->getDbKey(qId), 0);
                }
            }
        }

        if (dpu_comm_.isProfilingEnabled()) {
            dpu_comm_.dumpProfile("ungapped_prefilter");
            dpu_comm_.resetProfile();
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

    std::vector<int8_t> DpuPrefilterHostPipeline::extractPSSMFromProfile(
        const char *profileData, uint32_t seqlen, BaseMatrix *subMat)
    {
        const int KERNEL_AA_SLOTS = 21; // DPU kernels expect 20 AAs + sentinel slot
        const int alphSize = subMat ? subMat->alphabetSize : KERNEL_AA_SLOTS;
        const int stride = KERNEL_AA_SLOTS; // assume interleaved row-major layout

        std::vector<int8_t> pssm(seqlen * KERNEL_AA_SLOTS, -128);
        const int8_t *src = reinterpret_cast<const int8_t*>(profileData);

        for (uint32_t pos = 0; pos < seqlen; ++pos) {
            for (int aa = 0; aa < KERNEL_AA_SLOTS && aa < alphSize; ++aa) {
                int idx = static_cast<int>(pos * stride + aa);
                int v = static_cast<int>(src[idx]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                pssm[pos * KERNEL_AA_SLOTS + aa] = static_cast<int8_t>(v);
            }
        }

        return pssm;
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