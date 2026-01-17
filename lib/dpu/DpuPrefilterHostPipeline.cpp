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
#define DPU_DEBUG_MODE

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
    
    struct DpuPrefilterHostPipeline::KmerRunContext {
        // MRAM Offsets 
        uint32_t STATE_TABLE_OFF;
        uint32_t QUERY_PACKETS_OFF;
        uint32_t VARIABLE_INDEX_START;
        uint32_t RESULTS_HEADER_OFF;
        uint32_t CHECKPOINT_OFF;

        // Configuration
        uint32_t num_dpus;

        static KmerRunContext create(uint32_t num_dpus) {
            using CM = DpuCommunicationManager;
            KmerRunContext ctx;
            ctx.num_dpus = num_dpus;

            // Define Fixed Region Layout
            uint32_t desc_size = CM::alignToMram(sizeof(KmerBatchDescriptor));
            ctx.RESULTS_HEADER_OFF = desc_size;
            ctx.CHECKPOINT_OFF     = ctx.RESULTS_HEADER_OFF + CM::alignToMram(sizeof(KmerResultHeader));
            ctx.STATE_TABLE_OFF    = ctx.CHECKPOINT_OFF + CM::alignToMram(sizeof(KmerCheckpoint));
            
            uint32_t state_bytes   = CM::alignToMram(MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
            ctx.QUERY_PACKETS_OFF  = ctx.STATE_TABLE_OFF + state_bytes;
            
            ctx.VARIABLE_INDEX_START = ctx.QUERY_PACKETS_OFF + KMER_QUERY_BUFFER_SIZE;
            
            return ctx;
        }
    };

    // ============================================================================
    // HELPER: Partition Targets into MRAM-sized chunks
    // ============================================================================
    static std::vector<std::vector<std::vector<uint32_t>>> partitionTargetsIntoBatches(
        DBReader<unsigned int> *tdbr, uint32_t num_dpus)
    {
        // Rank-aware packing: use bin-packing splitter to even out work per DPU.

        auto chunks = DpuDbSplitter::splitDatabase(tdbr, num_dpus, MAX_DPU_INDEX_SIZE, MAX_DPU_SEQS);
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

        // Detailed batching diagnostics
        DPU_DEBUG_LOG << "[DPU] === Batching Diagnostics ==="  << "\n";
        DPU_DEBUG_LOG << "[DPU]   Total DB sequences: " << tdbr->getSize() << "\n";
        DPU_DEBUG_LOG << "[DPU]   Available DPUs: " << num_dpus << "\n";
        DPU_DEBUG_LOG << "[DPU]   Generated chunks: " << chunks.size() << "\n";
        DPU_DEBUG_LOG << "[DPU]   Target waves: " << num_waves << "\n";
        
        // Per-wave statistics
        for (size_t wave = 0; wave < num_waves; ++wave) {
            size_t active_dpus = 0;
            size_t total_seqs = 0;
            size_t min_seqs = SIZE_MAX;
            size_t max_seqs = 0;
            
            for (uint32_t d = 0; d < num_dpus; ++d) {
                size_t n = targetBatches[wave][d].size();
                if (n > 0) {
                    active_dpus++;
                    total_seqs += n;
                    if (n < min_seqs) min_seqs = n;
                    if (n > max_seqs) max_seqs = n;
                }
            }
            
            if (active_dpus == 0) min_seqs = 0;
            
            DPU_DEBUG_LOG << "[DPU]   Wave " << (wave + 1) << "/" << num_waves 
                          << ": active_dpus=" << active_dpus
                          << ", total_seqs=" << total_seqs
                          << ", seqs_per_dpu=[" << min_seqs << ".." << max_seqs << "]\n";
        }
        DPU_DEBUG_LOG << "[DPU] ==============================\n";

        return targetBatches;
    }

    // ============================================================================
    // HELPER: Partition Targets for MAXIMUM PARALLELISM (ungapped/gapped modes)
    // Uses LPT algorithm to spread work across ALL DPUs
    // ============================================================================
    static std::vector<std::vector<std::vector<uint32_t>>> partitionTargetsForParallelism(
        DBReader<unsigned int> *tdbr, uint32_t num_dpus)
    {
        // Use LPT distribution to maximize parallelism across all DPUs
        // Simpler MRAM limit (no k-mer index overhead)
        constexpr size_t MRAM_LIMIT_PARALLEL = 40 * 1024 * 1024; // 40MB for sequences
        
        auto chunks = DpuDbSplitter::distributeForParallelism(tdbr, num_dpus, MRAM_LIMIT_PARALLEL, MAX_DPU_SEQS);
        if (chunks.empty()) {
            return {};
        }

        // For parallel distribution, we typically get num_dpus chunks (one wave)
        // But if DB is huge, we might get multiple waves
        const size_t num_waves = (chunks.size() + num_dpus - 1) / num_dpus;
        std::vector<std::vector<std::vector<uint32_t>>> targetBatches(num_waves, std::vector<std::vector<uint32_t>>(num_dpus));

        for (size_t idx = 0; idx < chunks.size(); ++idx) {
            const size_t wave = idx / num_dpus;
            const size_t slot = idx % num_dpus;
            targetBatches[wave][slot] = std::move(chunks[idx]);
        }

        // Detailed batching diagnostics
        DPU_DEBUG_LOG << "[DPU] === Parallel Batching Diagnostics ===\n";
        DPU_DEBUG_LOG << "[DPU]   Total DB sequences: " << tdbr->getSize() << "\n";
        DPU_DEBUG_LOG << "[DPU]   Available DPUs: " << num_dpus << "\n";
        DPU_DEBUG_LOG << "[DPU]   Generated chunks: " << chunks.size() << "\n";
        DPU_DEBUG_LOG << "[DPU]   Target waves: " << num_waves << "\n";
        
        for (size_t wave = 0; wave < num_waves; ++wave) {
            size_t active_dpus = 0;
            size_t total_seqs = 0;
            size_t min_seqs = SIZE_MAX;
            size_t max_seqs = 0;
            
            for (uint32_t d = 0; d < num_dpus; ++d) {
                size_t n = targetBatches[wave][d].size();
                if (n > 0) {
                    active_dpus++;
                    total_seqs += n;
                    if (n < min_seqs) min_seqs = n;
                    if (n > max_seqs) max_seqs = n;
                }
            }
            
            if (active_dpus == 0) min_seqs = 0;
            
            DPU_DEBUG_LOG << "[DPU]   Wave " << (wave + 1) << "/" << num_waves 
                          << ": active_dpus=" << active_dpus
                          << ", total_seqs=" << total_seqs
                          << ", seqs_per_dpu=[" << min_seqs << ".." << max_seqs << "]\n";
        }
        DPU_DEBUG_LOG << "[DPU] ==========================================\n";

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

        // Reduced alphabet size for Amino Acids (21 -> 20) - excluded 'X'
        int alphabetSize = subMat->alphabetSize - 1;
        std::unique_ptr<KmerGenerator> kmerGen = std::make_unique<KmerGenerator>(ksize, alphabetSize, (short)kmerThr);
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
        
        // Statistics counters
        uint64_t totalBatchTransfers = 0;    // Number of query packet batch transfers to DPUs
        uint64_t totalOverflowEvents = 0;    // Number of result overflow events across all DPUs  

        auto ctx = KmerRunContext::create(num_dpus);

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
            std::vector<std::vector<uint8_t>> wave_index_buffers(num_dpus);

            for (size_t w = 0; w < wave_size; ++w) {
                uint32_t dpu_id = static_cast<uint32_t>(w);
                const auto& index = wave_indices[w];

                if (index.buckets.empty()) {
                    Debug(Debug::WARNING) << "[CPU] Chunk " << (wave_start + w) << " has empty index, skipping\n";
                    continue;
                }
                
                // Check total size using pre-calculated base offset (Buckets + Entries)
                uint32_t buckets_size = DpuCommunicationManager::alignToMram(index.buckets.size());
                uint32_t entries_size = DpuCommunicationManager::alignToMram(index.entries.size() * sizeof(KmerCompactIndexEntry));
                
                uint32_t variable_structures_end = ctx.VARIABLE_INDEX_START + buckets_size + entries_size;
                uint32_t fixed_structures_end = ctx.VARIABLE_INDEX_START;
                DPU_DEBUG_LOG << "[CPU] DPU " << dpu_id << " Memory Layout:\n";
                DPU_DEBUG_LOG << "  Fixed region: " << (fixed_structures_end / 1024) << " KB\n";
                DPU_DEBUG_LOG << "    Descriptor:    offset 0x0\n";
                DPU_DEBUG_LOG << "    Result Header: offset " << ctx.RESULTS_HEADER_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    Checkpoint:    offset " << ctx.CHECKPOINT_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    State Table:   offset " << ctx.STATE_TABLE_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "    Query Buffer:  offset " << ctx.QUERY_PACKETS_OFF << " (STATIC)\n";
                DPU_DEBUG_LOG << "  Variable region: " << ((variable_structures_end - fixed_structures_end) / 1024) << " KB\n";
                DPU_DEBUG_LOG << "    Buckets:       " << index.num_buckets << " buckets (" << (buckets_size / 1024) << " KB)\n";
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

                // Build merged index buffer (buckets | entries) in variable region
                uint32_t total_index_size = buckets_size + entries_size;

                wave_index_buffers[w].resize(total_index_size);
                uint8_t* ptr = wave_index_buffers[w].data();

                // Copy Buckets
                if (!index.buckets.empty()) memcpy(ptr, index.buckets.data(), index.buckets.size());
                ptr += buckets_size;

                // Copy Entries
                if (!index.entries.empty()) memcpy(ptr, index.entries.data(), index.entries.size() * sizeof(KmerCompactIndexEntry));

                DPU_DEBUG_LOG << "[CPU " << dpu_id << "] Prepared index: " << index.num_buckets << " buckets, " << index.entries.size() << " entries (" << (index.getTotalBytes() / 1024) << " KB)\n";
            }

            {
                // Transfer merged Index Data (Variable Region) in one pass
                dpu_comm_.scatterDataParallel(wave_index_buffers, ctx.VARIABLE_INDEX_START);

                // Log which DPUs received indices
                for (size_t w = 0; w < wave_size; ++w) {
                    if (!wave_index_buffers[w].empty()) {
                        DPU_DEBUG_LOG << "[CPU " << w << "] Loaded index (parallel) size=" << (wave_index_buffers[w].size() / 1024) << " KB\n";
                    }
                }
            }
            
            // === MULTI-QUERY BATCHING: Streaming packet generation with async double buffering ===
            // Uses DpuQueryPacketGenerator as a stateful iterator that generates packets on-demand.
            // 3-stage pipeline: CPU fills batch N+1 while DPU executes batch N
            
            std::string resultBuffer;
            
            DpuQueryPacketGenerator streamer(
                qdbr, kmerGen.get(), indexer.get(), subMat,
                ksize, useSpacedKmers, spacedPattern, patternSpan,
                takeOnlyBestKmer, par.compBiasCorrection != 0,
                par.compBiasCorrectionScale, kmerThr
            );
            
            // Reserve ~1% buffer for sentinel packets and alignment padding
            uint32_t max_packets_per_batch = (KMER_QUERY_BUFFER_SIZE / sizeof(KmerQueryPacket)) - 1000;
            
            // Double-buffered batch data
            KmerBatchData batches[2];
            for (int i = 0; i < 2; ++i) {
                batches[i].packets.resize(max_packets_per_batch + 100);
            }
            
            std::vector<std::vector<hit_t>> perQueryRawHits(qdbr->getSize());
            std::vector<uint64_t> perQueryPacketCount(qdbr->getSize(), 0);
            
            // Broadcast State Reset only once before the batch loop starts.
            // It is the DPU kernel's responsibility to handle per-query state resets via sentinels.
            std::vector<uint8_t> reset_state(DpuCommunicationManager::alignToMram(MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry)), 0xFF);
            dpu_comm_.broadcastData(reset_state.data(), reset_state.size(), ctx.STATE_TABLE_OFF);
            
            // Lambda to parse results from one DPU batch
            auto parseResults = [&](const std::vector<std::vector<KmerDoubleHit>>& per_dpu_results,
                                    const std::vector<size_t>& batchQueryIndices) {
                for (size_t w = 0; w < wave_size; ++w) {
                    const auto& dpu_hits = per_dpu_results[w];
                    const auto& chunk_targets = splits[wave_start + w];
                    
                    if (dpu_hits.empty()) continue;
                    
                    DPU_DEBUG_LOG << "[CPU] Wave " << w << " DPU " << (wave_start + w) 
                                  << ": " << dpu_hits.size() << " hits, chunk has " 
                                  << chunk_targets.size() << " targets\n";
                    
                    size_t queryIdxInBatch = 0;  // Index into batchQueryIndices
                    size_t hitIdx = 0;
                    
                    for (const auto& hit : dpu_hits) {
                        hitIdx++;
                        // Check sentinel (32-bit) to detect query delimiters
                        if (hit.target_id == (uint32_t)KMER_RESULT_SENTINEL_TARGET) {
                            queryIdxInBatch++;
                            if (queryIdxInBatch > batchQueryIndices.size()) {
                                Debug(Debug::ERROR) << "[CPU] ERROR: Received more delimiters than queries sent!\n";
                            }
                            continue;
                        }

                        // Ignore alignment padding hits
                        if (hit.target_id == (uint32_t)KMER_TARGET_ID_PADDING) {
                            continue;
                        }
                        
                        if (queryIdxInBatch >= batchQueryIndices.size()) {
                            Debug(Debug::ERROR) << "[CPU] ERROR: Hit received after all query delimiters!\n";
                            continue;
                        }
                        
                        if (hit.target_id >= chunk_targets.size()) {
                            Debug(Debug::ERROR) << "[CPU] FATAL: Invalid TargetID " << hit.target_id 
                                                << " (max: " << (chunk_targets.size() - 1) << ")\n";
                            EXIT(EXIT_FAILURE);
                        }
                        
                        size_t actualQueryIdx = batchQueryIndices[queryIdxInBatch];
                        
                        hit_t shortHit;
                        shortHit.seqId = chunk_targets[hit.target_id];
                        shortHit.prefScore = 1;
                        shortHit.diagonal = hit.diagonal;
                        
                        perQueryRawHits[actualQueryIdx].push_back(shortHit);
                    }
                }
            };
            
            // Lambda to fill a batch from the streamer
            auto fillBatch = [&](KmerBatchData& batch) {
                auto t_start_gen = std::chrono::high_resolution_clock::now();
                
                batch.packet_count = streamer.fillNextBatch(batch.packets.data(), max_packets_per_batch);
                
                auto t_end_gen = std::chrono::high_resolution_clock::now();
                double gen_time = std::chrono::duration<double>(t_end_gen - t_start_gen).count();
                
                if (batch.packet_count > 0) {
                    batch.query_indices = streamer.getLastBatchQueryIndices();
                    batch.valid = true;
                    
                    // Update per-query packet counts
                    for (size_t qi : batch.query_indices) {
                        perQueryPacketCount[qi] = 1;  // Mark as processed
                    }
                    
                    auto stats = streamer.getStats();
                    totalPacketsSent = stats.total_packets;
                    
                    Debug(Debug::INFO) << "[BENCH] Generation: " << batch.packet_count << " packets in "
                                       << gen_time << "s (" << (batch.packet_count / gen_time) / 1e6 << " Mpps)\n";
                    
                    DPU_DEBUG_LOG << "[CPU] Filled batch: " << batch.packet_count << " packets, "
                                  << batch.query_indices.size() << " queries\n";
                } else {
                    batch.valid = false;
                }
            };
            
            // === ASYNC DOUBLE-BUFFERED BATCH LOOP ===
            int current = 0;
            
            // Prime the pipeline: fill first batch on CPU
            fillBatch(batches[current]);
            
            std::future<std::vector<std::vector<KmerDoubleHit>>> pendingDpuResult;
            std::vector<size_t> pendingQueryIndices;
            
            while (batches[current].valid) {
                // 1. Start DPU execution asynchronously
                pendingQueryIndices = batches[current].query_indices;
                pendingDpuResult = std::async(std::launch::async, [&, current]() {
                    return processBatchOnDpu(ctx, batches[current], wave_indices, splits, wave_start, wave_size);
                });
                totalBatchTransfers++;
                
                // 2. While DPU is running, fill the next batch on CPU
                int next = 1 - current;
                if (!streamer.isFinished()) {
                    fillBatch(batches[next]);
                } else {
                    batches[next].valid = false;
                }
                
                // 3. Wait for DPU results
                auto per_dpu_batch_results = pendingDpuResult.get();
                
                // 4. Parse results while next batch is ready to go
                parseResults(per_dpu_batch_results, pendingQueryIndices);
                
                // 5. Swap buffers
                current = next;
            }  // End async streaming batch loop
            
            // === WRITE RESULTS FOR ALL QUERIES ===
            for (size_t q = 0; q < qdbr->getSize(); ++q) {
                const auto& raw_hits = perQueryRawHits[q];
                uint32_t queryKey = qdbr->getDbKey(q);
                
                perQueryPackets.push_back(perQueryPacketCount[q]);
                
                if (raw_hits.empty()) {
                    perQueryDoubleHits.push_back(0);
                    continue;
                }
                
                perQueryDoubleHits.push_back(raw_hits.size());
                totalDoubleHits += raw_hits.size();

                // Aggregate hits by (TargetId, Diagonal)
                std::vector<hit_t> sorted_hits = raw_hits;
                std::sort(sorted_hits.begin(), sorted_hits.end(), [](const hit_t& a, const hit_t& b) {
                    if (a.seqId != b.seqId) return a.seqId < b.seqId;
                    return a.diagonal < b.diagonal;
                });

                std::vector<hit_t> final_query_hits;
                final_query_hits.reserve(sorted_hits.size());
                final_query_hits.push_back(sorted_hits[0]);

                for (size_t i = 1; i < sorted_hits.size(); ++i) {
                    hit_t& last = final_query_hits.back();
                    const hit_t& curr = sorted_hits[i];
                    
                    if (last.seqId == curr.seqId && last.diagonal == curr.diagonal) {
                        last.prefScore += curr.prefScore;
                    } else {
                        final_query_hits.push_back(curr);
                    }
                }
                
                // Sort by score (descending) and then ID
                std::sort(final_query_hits.begin(), final_query_hits.end(), hit_t::compareHitsByScoreAndId);
                
                resultBuffer.clear();
                resultBuffer.reserve(final_query_hits.size() * 16);
                
                for (size_t i = 0; i < final_query_hits.size(); ++i) {
                    char outbuf[256];
                    size_t len = QueryMatcher::prefilterHitToBuffer(outbuf, final_query_hits[i]);
                    resultBuffer.append(outbuf, len);
                }
                
                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
                
                // DPU_DEBUG_LOG << "[CPU] Query " << q << " (key=" << queryKey << "): " << final_query_hits.size() << " aggregated hits\n";
            }
            
            DPU_DEBUG_LOG << "\n[CPU] Wave " << (wave_idx + 1) << " complete! Processed " << qdbr->getSize() << " queries\n"; 
        } // end for wave_idx
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();
        
        
        // TODO: change to only print when debug
        Debug(Debug::INFO) << "\n[DPU K-mer Prefilter Statistics]\n";
        Debug(Debug::INFO) << "  Total queries processed:      " << qdbr->getSize() << "\n";
        Debug(Debug::INFO) << "  Total query packets sent:     " << totalPacketsSent << "\n";
        Debug(Debug::INFO) << "  Total batch transfers:        " << totalBatchTransfers << "\n";
        Debug(Debug::INFO) << "  Total overflow events:        " << totalOverflowEvents << "\n";
        Debug(Debug::INFO) << "  Total double hits detected:   " << totalDoubleHits << "\n";
        Debug(Debug::INFO) << "  Processing time:              " << seconds << " seconds\n";
        if (totalBatchTransfers > 0) {
            Debug(Debug::INFO) << "  Avg packets per batch:        " << (totalPacketsSent / totalBatchTransfers) << "\n";
            Debug(Debug::INFO) << "  Overflow rate:                " << (100.0 * totalOverflowEvents / totalBatchTransfers) << "%\n";
        }
        Debug(Debug::INFO) << "\n";
    }

    // ============================================================================
    // K-MER BATCH HELPERS
    // ============================================================================
    
    std::vector<std::vector<uint8_t>> DpuPrefilterHostPipeline::prepareKmerDescriptors(
        const KmerRunContext& ctx,
        const std::vector<DpuIndexBuffer>& wave_indices,
        const std::vector<std::vector<uint32_t>>& splits,
        uint32_t num_packets,
        size_t wave_start,
        size_t wave_size) 
    {
        std::vector<std::vector<uint8_t>> descriptors(ctx.num_dpus);
        const uint32_t desc_size = DpuCommunicationManager::alignToMram(sizeof(KmerBatchDescriptor));

        for (uint32_t d = 0; d < ctx.num_dpus; ++d) {
            descriptors[d].resize(desc_size, 0); // Zero initialize
            
            if (d >= wave_size || wave_indices[d].buckets.empty()) {
                // Empty descriptor for idle DPUs
                KmerBatchDescriptor empty_desc = {};
                memcpy(descriptors[d].data(), &empty_desc, sizeof(empty_desc));
                continue;
            }

            const auto& index = wave_indices[d];
            const auto& chunk = splits[wave_start + d];
            
            // Calculate Variable Offsets (Buckets | Entries)
            uint32_t buckets_off = ctx.VARIABLE_INDEX_START;
            uint32_t buckets_size = DpuCommunicationManager::alignToMram(index.buckets.size());
            uint32_t entries_off = buckets_off + buckets_size;
            uint32_t entries_size = DpuCommunicationManager::alignToMram(index.entries.size() * sizeof(KmerCompactIndexEntry));
            uint32_t results_off = entries_off + entries_size;
            
            uint32_t remaining_mram = DPU_MRAM_TOTAL_SIZE - results_off;
            uint32_t result_buffer_size = std::max((uint32_t)KMER_MIN_OUTPUT_BUFFER_SIZE, remaining_mram);
            result_buffer_size = DpuCommunicationManager::alignToMram(result_buffer_size);

            KmerBatchDescriptor desc = {};
            desc.num_query_packets = num_packets;
            desc.num_targets = chunk.size();
            desc.num_buckets = index.num_buckets;
            desc.num_index_entries = index.entries.size();
            
            desc.state_table_offset = ctx.STATE_TABLE_OFF;
            desc.query_packets_offset = ctx.QUERY_PACKETS_OFF;
            desc.buckets_offset = buckets_off;
            desc.index_entries_offset = entries_off;
            
            desc.results_offset = results_off;
            desc.results_buffer_size = result_buffer_size;
            
            memcpy(descriptors[d].data(), &desc, sizeof(desc));
        }
        return descriptors;
    }
    
    std::vector<std::vector<KmerDoubleHit>> DpuPrefilterHostPipeline::executeKmerBatchWithOverflow(
        const KmerRunContext& ctx,
        const std::vector<std::vector<uint8_t>>& descriptors)
    {
        std::vector<std::vector<KmerDoubleHit>> accumulated_results(ctx.num_dpus);
        bool all_dpus_complete = false;
        int overflow_retries = 0;

        // 1. Initial Launch - Reset checkpoint and send descriptors
        KmerCheckpoint zero_ckpt = {0, 0, 0, 0};
        dpu_comm_.broadcastData(&zero_ckpt, sizeof(KmerCheckpoint), ctx.CHECKPOINT_OFF);
        dpu_comm_.scatterDataParallel(descriptors, 0);

        while (!all_dpus_complete) {
            dpu_comm_.executeKernels(); // Block until done
            all_dpus_complete = true;   // Assume done until overflow seen

            for (uint32_t d = 0; d < ctx.num_dpus; ++d) {
                // Extract Results Offset from the descriptor we just sent
                const KmerBatchDescriptor* desc = (const KmerBatchDescriptor*)descriptors[d].data();
                if (desc->num_targets == 0) continue;

                uint32_t overflow = 0;
                
                // === REUSE: DpuWorkflow::gatherResultsClamped ===
                // This handles reading header, checking count, handling alignment, and reading hits
                auto iteration_hits = workflow_.gatherResultsClamped<KmerDoubleHit>(
                    d, 
                    desc->results_offset, 
                    desc->results_buffer_size, 
                    &overflow 
                );

                // Filter padding and accumulate
                accumulated_results[d].reserve(accumulated_results[d].size() + iteration_hits.size());
                for (const auto& hit : iteration_hits) {
                    if (hit.target_id != KMER_TARGET_ID_PADDING) {
                        accumulated_results[d].push_back(hit);
                    }
                }

                if (overflow) {
                    all_dpus_complete = false; // Must relaunch this DPU
                    
                    // Reset Result Header (Count=0, Overflow=0) at the START of results buffer
                    // Note: We must write to results_offset now (where header is), NOT RESULTS_HEADER_OFF
                    KmerResultHeader zero_hdr = {0, 0};
                    dpu_comm_.scatterDataToDPU(d, &zero_hdr, sizeof(KmerResultHeader), desc->results_offset);
                }
            }
            overflow_retries++;
        }
        
        DPU_DEBUG_LOG << "[CPU] Batch complete after " << overflow_retries << " iterations\n";
        
        return accumulated_results;
    }
    
    std::vector<std::vector<KmerDoubleHit>> DpuPrefilterHostPipeline::processBatchOnDpu(
        const KmerRunContext& ctx,
        const KmerBatchData& batch,
        const std::vector<DpuIndexBuffer>& wave_indices,
        const std::vector<std::vector<uint32_t>>& splits,
        size_t wave_start,
        size_t wave_size)
    {
        if (!batch.valid || batch.packet_count == 0) {
            return std::vector<std::vector<KmerDoubleHit>>(ctx.num_dpus);
        }
        
        // 1. Prepare descriptors for this batch
        auto descriptors = prepareKmerDescriptors(ctx, wave_indices, splits, 
                                                   batch.packet_count, wave_start, wave_size);
        
        // 2. Transfer: Broadcast Query Packets to all DPUs
        auto t_start_xfer = std::chrono::high_resolution_clock::now();
        
        uint32_t packets_size = batch.packet_count * sizeof(KmerQueryPacket);
        dpu_comm_.broadcastData(batch.packets.data(), packets_size, ctx.QUERY_PACKETS_OFF);
        
        auto t_end_xfer = std::chrono::high_resolution_clock::now();
        
        // 3. Execute kernel and gather results (handles overflow internally, includes descriptor scatter)
        auto t_start_exec = std::chrono::high_resolution_clock::now();
        
        auto results = executeKmerBatchWithOverflow(ctx, descriptors);
        
        auto t_end_exec = std::chrono::high_resolution_clock::now();
        
        double xfer_time = std::chrono::duration<double>(t_end_xfer - t_start_xfer).count();
        double exec_time = std::chrono::duration<double>(t_end_exec - t_start_exec).count();
        double xfer_mb = packets_size / (1024.0 * 1024.0);
        
        Debug(Debug::INFO) << "[BENCH] DPU: Xfer " << xfer_time << "s (" << (xfer_mb / xfer_time) << " MB/s), "
                           << "Exec " << exec_time << "s\n";
        
        return results;
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

        // Use parallel distribution for maximum DPU utilization (not dense packing)
        auto targetBatches = partitionTargetsForParallelism(tdbr, num_dpus);
        if (targetBatches.empty()) {
            Debug(Debug::ERROR) << "[DPU] Database splitting failed for gapped path\n";
            return;
        }

        // Storage for accumulated results: [queryIdx] -> list of hits
        std::vector<std::vector<Matcher::result_t>> allResults(qdbr->getSize());

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);

        // DYNAMIC TASKLET CALCULATION
        // Kernel MAX_SAFE_TASKLETS=11 due to WRAM constraints (stack + SW buffers + scratch)
        // Host matches kernel limit for optimal resource usage
        const uint8_t MAX_KERNEL_TASKLETS = 11;
        uint8_t active_tasklets = MAX_KERNEL_TASKLETS;

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

        // Profiling alias
        using PS = DpuCommunicationManager::ProfileSlot;

        // 2. Loop over Target Batches
        for (size_t bIdx = 0; bIdx < targetBatches.size(); ++bIdx) {
            const auto& perDpuTargetIndices = targetBatches[bIdx];
            std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
            std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

            // Assemble data for this batch
            {
                auto timer = dpu_comm_.timeSlot(PS::HostBuildTargetBatch);
                #pragma omp parallel for schedule(dynamic)
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (!perDpuTargetIndices[d].empty()) {
                        assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[d],
                                                     perDpuTargetData[d], perDpuTargetMeta[d], subMat);
                    }
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
                // Time the entire batch processing
                auto batch_timer = dpu_comm_.timeSlot(PS::HostTotalBatch);
                
                BatchData batch;
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostBuildQueryBatch);
                    batch = buildQueryBatch(q_cursor, qdbr, subMat, par, compBias, q_limits, evaluer, static_cast<int16_t>(par.minDiagScoreThr), nullptr);
                }
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
                        static_cast<uint8_t>(par.covThr * 100.0f), // TODO this should be wrapped inside the object itself
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

                // DPU execution + result gathering with timing
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostDispatcherWait);
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

                            // Parse headers and find max hits for parallel gather
                            std::vector<uint32_t> hit_counts(num_dpus, 0);
                            uint32_t max_hit_count = 0;
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
                                hit_counts[d] = hit_count;
                                if (hit_count > max_hit_count) max_hit_count = hit_count;
                            }

                            // Parallel gather all hits using max size (reference pattern: single bulk transfer)
                            if (max_hit_count > 0) {
                                const uint32_t transfer_size = DpuCommunicationManager::alignToMram(
                                    max_hit_count * static_cast<uint32_t>(sizeof(GappedHit)));
                                std::vector<std::vector<uint8_t>> hit_bufs;
                                dpu_comm_.gatherDataParallel(hit_bufs, transfer_size, layouts[0].results_offset + 8);

                                // Parse hits from buffers
                                for (uint32_t d = 0; d < num_dpus; ++d) {
                                    if (!dpu_ready[d] || hit_counts[d] == 0) {
                                        dpu_hits[d].clear();
                                    } else {
                                        dpu_hits[d].resize(hit_counts[d]);
                                        memcpy(dpu_hits[d].data(), hit_bufs[d].data(), 
                                               hit_counts[d] * sizeof(GappedHit));
                                    }
                                    dpu_active[d] = false;
                                    dpu_ready[d] = false;
                                }
                            } else {
                                for (uint32_t d = 0; d < num_dpus; ++d) {
                                    if (dpu_ready[d]) {
                                        dpu_hits[d].clear();
                                        dpu_active[d] = false;
                                        dpu_ready[d] = false;
                                    }
                                }
                            }
                        }
                    }
                }

                // Count total hits for profiling
                uint64_t total_hits = 0;
                for (const auto& hits : dpu_hits) total_hits += hits.size();

                std::vector<std::vector<Matcher::result_t>> batch_results(batch.meta.size());
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostProcessHits);
                    processDpuHits(dpu_hits, batch, tdbr, par, evaluer, sameDB, taxonomyHook, batch_results);
                    dpu_comm_.recordSlotMetrics(PS::HostProcessHits, 0, total_hits);
                }

                for (size_t i = 0; i < batch_results.size(); ++i) {
                    if (batch_results[i].empty()) continue;
                    const size_t q_global = batch.qids[i];
                    allResults[q_global].insert(allResults[q_global].end(), batch_results[i].begin(), batch_results[i].end());
                }
            }

        } // End Target Batch Loop

        // 3. Write Results
        {
            auto timer = dpu_comm_.timeSlot(PS::HostResultWrite);
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

        kernel_mgr_.loadKernel(DpuKernelManager::KernelType::UNGAPPED);

        // Use parallel distribution (same as gapped path)
        auto targetBatches = partitionTargetsForParallelism(tdbr, num_dpus);
        if (targetBatches.empty()) {
            Debug(Debug::ERROR) << "[DPU] Database splitting failed for ungapped path\n";
            return;
        }

        std::vector<float> compBias(qdbr->getMaxSeqLen() + 1, 0.0f);
        int16_t minScoreThr = static_cast<int16_t>(par.minDiagScoreThr);

        // Pre-calculate max query sizes for layout (same pattern as gapped)
        const uint32_t max_query_len = qdbr->getMaxSeqLen();
        const uint32_t max_pssm_per_query = max_query_len * 32 + 1024;
        const uint32_t max_queries_per_batch = 64u;

        BatchLimits q_limits{};
        q_limits.max_queries = max_queries_per_batch;
        q_limits.max_pssm_bytes = max_pssm_per_query * max_queries_per_batch;
        q_limits.max_common_bytes = DpuCommunicationManager::alignToMram(q_limits.max_queries * static_cast<uint32_t>(sizeof(QueryMetadata))) +
                        DpuCommunicationManager::alignToMram(max_pssm_per_query * q_limits.max_queries);

        const uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(UngappedBatchDescriptor));
        const uint32_t UNGAPPED_RESULTS_BYTES = 8 * 1024 * 1024;

        // Tasklet calculation for ungapped kernel
        // Must match kernel constants: TARGET_TILE_SIZE=1024, PSSM_CACHE_SIZE=512, MAX_DIAG_ENTRIES=4096
        const uint32_t TARGET_TILE_SIZE = 1024;
        const uint32_t PSSM_CACHE_SIZE = 512;
        const uint32_t MAX_DIAG_ENTRIES = 4096;
        uint32_t diag_bytes = MAX_DIAG_ENTRIES * 2; // int16_t per diagonal (8KB)
        uint32_t wramPerTasklet = TARGET_TILE_SIZE + PSSM_CACHE_SIZE + diag_bytes + 1024; // +1024 for stack/misc
        // ~10.5KB per tasklet -> 6 tasklets fit in 64KB WRAM
        uint8_t active_tasklets = std::min<uint8_t>(6, calculateActiveTasklets(wramPerTasklet));

        // Guard: Check sequence length constraints
        // Diagonal buffer size limits: max_target_len = MAX_DIAG_ENTRIES - max_query_len
        // Sequences exceeding this will be silently skipped by the kernel
        const uint32_t max_target_for_diag = (MAX_DIAG_ENTRIES > max_query_len) 
                                              ? (MAX_DIAG_ENTRIES - max_query_len) 
                                              : 256;
        const uint32_t db_max_target_len = tdbr->getMaxSeqLen();
        if (db_max_target_len > max_target_for_diag) {
            Debug(Debug::WARNING) << "[DPU] Ungapped kernel constraint: max_query=" << max_query_len 
                                  << " + max_target=" << db_max_target_len << " = " << (max_query_len + db_max_target_len)
                                  << " exceeds diagonal buffer (" << MAX_DIAG_ENTRIES << ").\n"
                                  << "[DPU] Targets longer than " << max_target_for_diag 
                                  << " residues will be SKIPPED. Consider using gapped mode for long sequences.\n";
        }

        // Storage for accumulated results: [queryIdx] -> list of hits
        std::vector<std::vector<hit_t>> allResults(qdbr->getSize());

        // Profiling alias
        using PS = DpuCommunicationManager::ProfileSlot;

        Debug(Debug::INFO) << "DPU Ungapped Batch (Parallel):\n"
                           << "  num_dpus = " << num_dpus << "\n"
                           << "  target_batches = " << targetBatches.size() << "\n"
                           << "  max_query_len = " << max_query_len << "\n"
                           << "  max_target_for_diag = " << max_target_for_diag << "\n"
                           << "  active_tasklets = " << (int)active_tasklets << "\n";

        // Loop over Target Batches
        for (size_t bIdx = 0; bIdx < targetBatches.size(); ++bIdx) {
            const auto& perDpuTargetIndices = targetBatches[bIdx];
            std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
            std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

            // Assemble target data in parallel (same as gapped)
            {
                auto timer = dpu_comm_.timeSlot(PS::HostBuildTargetBatch);
                #pragma omp parallel for schedule(dynamic)
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (!perDpuTargetIndices[d].empty()) {
                        assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[d],
                                                     perDpuTargetData[d], perDpuTargetMeta[d], subMat);
                    }
                }
            }

            // Calculate fixed layout for parallel transfer
            uint32_t max_tdata_size = DpuCommunicationManager::alignToMram(40 * 1024 * 1024);
            DpuWorkflow::MramLayout max_layout = workflow_.calculateLayout(
                sizeof(UngappedBatchDescriptor), q_limits.max_common_bytes, 16384, q_limits.max_queries, max_tdata_size, sizeof(Hit), 0);
            max_layout.results_capacity = DpuCommunicationManager::alignToMram(UNGAPPED_RESULTS_BYTES);

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus, max_layout);
            std::vector<UngappedBatchDescriptor> bds(num_dpus);
            std::vector<uint32_t> maxHitsPerDpu(num_dpus, 0);

            // Prepare initial descriptors
            for (uint32_t d = 0; d < num_dpus; ++d) {
                const uint32_t usable_results = (max_layout.results_capacity > 8u)
                    ? (max_layout.results_capacity - 8u)
                    : 0u;
                maxHitsPerDpu[d] = usable_results / static_cast<uint32_t>(sizeof(Hit));

                UngappedBatchDescriptor init_bd{};
                init_bd.header.num_targets = perDpuTargetMeta[d].size();
                init_bd.header.num_active_tasklets = active_tasklets;
                init_bd.header.queries_metadata_offset = max_layout.common_data_offset;
                init_bd.header.pssm_data_offset = init_bd.header.queries_metadata_offset + 
                    DpuCommunicationManager::alignToMram(sizeof(QueryMetadata));
                init_bd.header.targets_metadata_offset = max_layout.target_meta_offset;
                init_bd.header.targets_data_offset = max_layout.target_data_offset;
                init_bd.header.results_offset = max_layout.results_offset;
                init_bd.header.results_buffer_size = max_layout.results_capacity;
                bds[d] = init_bd;
            }

            // Parallel Scatter Targets (key optimization!)
            workflow_.scatterBatchParallel(bds, perDpuTargetMeta, perDpuTargetData, max_layout);

            // Process queries in batches
            size_t q_cursor = 0;
            while (q_cursor < qdbr->getSize()) {
                auto batch_timer = dpu_comm_.timeSlot(PS::HostTotalBatch);

                BatchData batch;
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostBuildQueryBatch);
                    batch = buildQueryBatch(q_cursor, qdbr, subMat, par, compBias, q_limits, evaluer, minScoreThr, nullptr);
                }
                if (batch.empty) {
                    Debug(Debug::ERROR) << "[DPU] Ungapped batch could not fit any query at cursor " << q_cursor;
                    break;
                }
                q_cursor = batch.next_q_idx;

                // Broadcast common data (queries + PSSMs)
                workflow_.broadcastCommon(batch.common_buffer->data(), batch.common_size, max_layout.common_data_offset);

                // Prepare per-DPU descriptors
                std::vector<std::vector<uint8_t>> bd_bufs(num_dpus);
                const uint32_t qmeta_size = static_cast<uint32_t>(batch.meta.size() * sizeof(QueryMetadata));
                const uint32_t pssm_size = static_cast<uint32_t>(batch.pssm.size());

                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (perDpuTargetMeta[d].empty()) continue;

                    const uint32_t qmeta_off = max_layout.common_data_offset;
                    const uint32_t pssm_off = qmeta_off + DpuCommunicationManager::alignToMram(qmeta_size);

                    UngappedBatchDescriptor bd{};
                    bd.header.num_queries = batch.meta.size();
                    bd.header.num_targets = static_cast<uint32_t>(perDpuTargetMeta[d].size());
                    bd.header.query_len = batch.max_q_len;
                    bd.header.queries_metadata_offset = qmeta_off;
                    bd.header.pssm_data_offset = pssm_off;
                    bd.header.targets_metadata_offset = max_layout.target_meta_offset;
                    bd.header.targets_data_offset = max_layout.target_data_offset;
                    bd.header.results_offset = max_layout.results_offset;
                    bd.header.results_buffer_size = max_layout.results_capacity;
                    bd.header.num_active_tasklets = active_tasklets;
                    bd.min_score = minScoreThr;
                    bd.gap_open_cost = static_cast<int16_t>(par.gapOpen.values.aminoacid());
                    bd.gap_extend_cost = static_cast<int16_t>(par.gapExtend.values.aminoacid());
                    bd.pssm_bias = 0;

                    bd_bufs[d].resize(sizeof(UngappedBatchDescriptor));
                    memcpy(bd_bufs[d].data(), &bd, sizeof(UngappedBatchDescriptor));
                }

                // Parallel scatter descriptors
                dpu_comm_.scatterDataParallel(bd_bufs, 0);

                // Clear results headers
                uint64_t zero_hdr = 0;
                dpu_comm_.broadcastData(&zero_hdr, 8, max_layout.results_offset);

                // Launch all DPUs synchronously
                dpu_comm_.executeKernels();

                // Parallel gather results
                std::vector<std::vector<Hit>> dpu_hits(num_dpus);
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostDispatcherWait);

                    // Gather headers in parallel
                    std::vector<std::vector<uint8_t>> header_bufs;
                    dpu_comm_.gatherDataParallel(header_bufs, 8, max_layout.results_offset);

                    // Parse headers
                    std::vector<uint32_t> hit_counts(num_dpus, 0);
                    uint32_t max_hit_count = 0;
                    for (uint32_t d = 0; d < num_dpus; ++d) {
                        if (perDpuTargetMeta[d].empty() || header_bufs[d].size() < 8) continue;
                        uint64_t hdr = 0;
                        memcpy(&hdr, header_bufs[d].data(), 8);
                        uint32_t hit_count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
                        if (hit_count > maxHitsPerDpu[d]) {
                            Debug(Debug::ERROR) << "[DPU] Ungapped overflow: DPU " << d 
                                << " hits=" << hit_count << " max=" << maxHitsPerDpu[d];
                            EXIT(EXIT_FAILURE);
                        }
                        hit_counts[d] = hit_count;
                        if (hit_count > max_hit_count) max_hit_count = hit_count;
                    }

                    // Parallel gather hits
                    if (max_hit_count > 0) {
                        const uint32_t transfer_size = DpuCommunicationManager::alignToMram(
                            max_hit_count * static_cast<uint32_t>(sizeof(Hit)));
                        std::vector<std::vector<uint8_t>> hit_bufs;
                        dpu_comm_.gatherDataParallel(hit_bufs, transfer_size, max_layout.results_offset + 8);

                        for (uint32_t d = 0; d < num_dpus; ++d) {
                            if (hit_counts[d] == 0) continue;
                            dpu_hits[d].resize(hit_counts[d]);
                            memcpy(dpu_hits[d].data(), hit_bufs[d].data(), hit_counts[d] * sizeof(Hit));
                        }
                    }
                }

                // Process hits
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostProcessHits);
                    uint64_t total_hits = 0;
                    for (uint32_t d = 0; d < num_dpus; ++d) {
                        for (const auto& hit : dpu_hits[d]) {
                            if (hit.query_id >= batch.meta.size()) continue;
                            size_t q_global = batch.qids[hit.query_id];
                            hit_t res;
                            res.seqId = tdbr->getDbKey(hit.target_id);
                            res.prefScore = hit.score;
                            res.diagonal = hit.diagonal;
                            allResults[q_global].push_back(res);
                            total_hits++;
                        }
                    }
                    dpu_comm_.recordSlotMetrics(PS::HostProcessHits, 0, total_hits);
                }
            }
        } // End Target Batch Loop

        // Write Results
        {
            auto timer = dpu_comm_.timeSlot(PS::HostResultWrite);
            for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
                auto& results = allResults[qId];
                if (!results.empty()) {
                    std::sort(results.begin(), results.end(), hit_t::compareHitsByScoreAndId);
                    size_t keep = std::min(results.size(), (size_t)par.maxResListLen);
                    std::string resultBuffer;
                    for (size_t k = 0; k < keep; ++k) {
                        char buffer[256];
                        size_t len = QueryMatcher::prefilterHitToBuffer(buffer, results[k]);
                        resultBuffer.append(buffer, len);
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
        // Kernel MAX_SAFE_TASKLETS=11 due to WRAM constraints (stack + SW buffers + scratch)
        // Host matches kernel limit for optimal resource usage
        const uint8_t MAX_KERNEL_TASKLETS = 11;
        uint8_t tasklet_limit = MAX_KERNEL_TASKLETS;

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

        // Use parallel distribution for maximum DPU utilization (not dense packing)
        auto targetBatches = partitionTargetsForParallelism(tdbr, num_dpus);
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

        // Profiling alias
        using PS = DpuCommunicationManager::ProfileSlot;

        for (size_t bIdx = 0; bIdx < targetBatches.size(); ++bIdx) {
            const auto& perDpuTargetIndices = targetBatches[bIdx];
            std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
            std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

            // Ensure kernel is loaded before sending any data to DPUs in this target batch
            kernel_mgr_.loadKernel(DpuKernelManager::KernelType::COMBINED);

            {
                auto timer = dpu_comm_.timeSlot(PS::HostBuildTargetBatch);
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
            }

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);
            std::vector<CombinedBatchDescriptor> bds(num_dpus);
            std::vector<uint32_t> maxHitsPerDpu(num_dpus, 0);
            std::vector<uint32_t> active_dpus;

            // Calculate fixed layout for parallel transfer (same pattern as gapped path)
            // MAX_SEQS_PER_DPU = 16384, TARGET_BUDGET_BYTES = 40MB
            uint32_t max_tdata_size = DpuCommunicationManager::alignToMram(40 * 1024 * 1024);
            DpuWorkflow::MramLayout max_layout = workflow_.calculateLayout(
                sizeof(CombinedBatchDescriptor), limits.reserved_common, 16384, limits.max_batch_queries, max_tdata_size, sizeof(GappedHit), 0);

            for (uint32_t d = 0; d < num_dpus; ++d) {
                layouts[d] = max_layout; // Use fixed layout for everyone

                const uint32_t usable_results = (max_layout.results_capacity > limits.result_header_bytes)
                    ? (max_layout.results_capacity - limits.result_header_bytes)
                    : 0u;
                maxHitsPerDpu[d] = usable_results / static_cast<uint32_t>(sizeof(GappedHit));

                if (!perDpuTargetMeta[d].empty()) {
                    active_dpus.push_back(d);
                }

                // Initialize descriptor placeholder
                CombinedBatchDescriptor init_bd{};
                init_bd.header.num_targets = static_cast<uint32_t>(perDpuTargetMeta[d].size());
                init_bd.header.num_active_tasklets = limits.tasklet_limit;

                const uint32_t qmeta_aligned = DpuCommunicationManager::alignToMram(sizeof(QueryMetadata));
                init_bd.header.queries_metadata_offset = max_layout.common_data_offset;
                init_bd.header.pssm_data_offset = init_bd.header.queries_metadata_offset + qmeta_aligned;
                init_bd.header.targets_metadata_offset = max_layout.target_meta_offset;
                init_bd.header.targets_data_offset = max_layout.target_data_offset;
                init_bd.header.results_offset = max_layout.results_offset;
                init_bd.header.results_buffer_size = max_layout.results_capacity;
                bds[d] = init_bd;

                if (max_layout.total_mram_used > DPU_MRAM_TOTAL_SIZE) {
                    Debug(Debug::ERROR) << "[DPU] Combined layout exceeds MRAM for DPU " << d << " (" << max_layout.total_mram_used << " > " << DPU_MRAM_TOTAL_SIZE << ")";
                    EXIT(EXIT_FAILURE);
                }
            }

            size_t max_batch_queries_limit = limits.max_batch_queries;
            {
                uint32_t cap_by_results = limits.max_batch_queries;
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (perDpuTargetMeta[d].empty()) continue;
                    const uint32_t targets = static_cast<uint32_t>(perDpuTargetMeta[d].size());
                    const uint32_t max_hits = (max_layout.results_capacity > limits.result_header_bytes)
                        ? (max_layout.results_capacity - limits.result_header_bytes) / static_cast<uint32_t>(sizeof(GappedHit))
                        : 0u;
                    const uint32_t qcap = (targets == 0 || max_hits == 0) ? 1u : std::max<uint32_t>(1u, max_hits / targets);
                    cap_by_results = std::min(cap_by_results, qcap);
                }
                max_batch_queries_limit = std::max<size_t>(1u, std::min<size_t>(max_batch_queries_limit, cap_by_results));
            }

            // DATA-RESIDENT PATTERN: Scatter targets ONCE at start of wave
            // Targets stay resident in DPU MRAM while we broadcast multiple query batches
            DPU_DEBUG_LOG << "[DPU] Wave " << (bIdx + 1) << "/" << targetBatches.size() 
                          << ": Scattering " << active_dpus.size() << " target batches (data-resident)\\n";
            workflow_.scatterTargetsOnly(perDpuTargetMeta, perDpuTargetData, max_layout);

            BatchLimits batch_limits{};
            batch_limits.max_queries = static_cast<uint32_t>(max_batch_queries_limit);
            batch_limits.max_pssm_bytes = limits.max_pssm_bytes;
            batch_limits.max_common_bytes = limits.reserved_common;

            std::vector<std::vector<GappedHit>> dpu_hits(num_dpus);
            std::vector<bool> dpu_active(num_dpus, false);
            std::vector<bool> dpu_ready(num_dpus, false);

            // Parallel gather helper - collects results from all ready DPUs in one bulk transfer
            // NOTE: Must be defined after dpu_hits so lambda capture works
            auto parallelGatherCombinedResults = [&](const std::vector<bool>& ready_flags) {
                // Step 1: Parallel gather all headers
                std::vector<std::vector<uint8_t>> header_bufs(num_dpus);
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (ready_flags[d]) header_bufs[d].resize(8);
                }
                dpu_comm_.gatherDataParallel(header_bufs, 8, max_layout.results_offset);

                // Step 2: Parse headers and find max hit count
                std::vector<uint32_t> hit_counts(num_dpus, 0);
                uint32_t max_hit_count = 0;
                for (uint32_t d = 0; d < num_dpus; ++d) {
                    if (!ready_flags[d]) continue;

                    uint64_t hdr = 0;
                    memcpy(&hdr, header_bufs[d].data(), 8);
                    const uint32_t hit_count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
                    const uint32_t overflow_flag = static_cast<uint32_t>(hdr >> 32);
                    const uint32_t max_hits = maxHitsPerDpu[d];
                    if (overflow_flag != 0) {
                        Debug(Debug::ERROR) << "[DPU] Combined batch overflow flag set on DPU " << d
                                            << " (hits=" << hit_count << ", capacity=" << max_hits << ")";
                    }
                    if (hit_count > max_hits) {
                        Debug(Debug::ERROR) << "[DPU] Combined batch overflow detected for DPU " << d
                                            << " (hits=" << hit_count << ", max=" << max_hits << ")";
                        EXIT(EXIT_FAILURE);
                    }
                    hit_counts[d] = hit_count;
                    if (hit_count > max_hit_count) max_hit_count = hit_count;
                }

                // Step 3: Parallel gather all hits using max size (reference pattern: single bulk transfer)
                if (max_hit_count > 0) {
                    const uint32_t transfer_size = DpuCommunicationManager::alignToMram(
                        max_hit_count * static_cast<uint32_t>(sizeof(GappedHit)));
                    std::vector<std::vector<uint8_t>> hit_bufs(num_dpus);
                    for (uint32_t d = 0; d < num_dpus; ++d) {
                        if (ready_flags[d]) hit_bufs[d].resize(transfer_size);
                    }
                    dpu_comm_.gatherDataParallel(hit_bufs, transfer_size, max_layout.results_offset + 8);

                    // Parse hits from buffers
                    for (uint32_t d = 0; d < num_dpus; ++d) {
                        if (!ready_flags[d] || hit_counts[d] == 0) {
                            dpu_hits[d].clear();
                        } else {
                            dpu_hits[d].resize(hit_counts[d]);
                            memcpy(dpu_hits[d].data(), hit_bufs[d].data(),
                                   hit_counts[d] * sizeof(GappedHit));
                        }
                    }
                } else {
                    for (uint32_t d = 0; d < num_dpus; ++d) {
                        if (ready_flags[d]) {
                            dpu_hits[d].clear();
                        }
                    }
                }
            };

            // Profiling alias for cleaner code
            using PS = DpuCommunicationManager::ProfileSlot;

            size_t q_cursor = 0;
            size_t query_batch_num = 0;
            while (q_cursor < qdbr->getSize()) {
                // Time the entire batch processing
                auto batch_timer = dpu_comm_.timeSlot(PS::HostTotalBatch);
                
                BatchData batch;
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostBuildQueryBatch);
                    batch = buildQueryBatch(q_cursor, qdbr, subMat, par, compBias, batch_limits, evaluer, minUngappedThr, nullptr);
                }
                if (batch.empty) {
                    Debug(Debug::ERROR) << "[DPU] Combined batch could not fit any query at cursor " << q_cursor;
                    break;
                }
                q_cursor = batch.next_q_idx;
                query_batch_num++;

                for (auto &v : dpu_hits) v.clear();

                // DATA-RESIDENT: Only broadcast queries (targets already in MRAM)
                workflow_.broadcastCommon(batch.common_buffer->data(), batch.common_size, max_layout.common_data_offset);

                // Build descriptors (one per DPU) with updated query batch info
                std::vector<CombinedBatchDescriptor> query_descriptors(num_dpus);
                const uint32_t qmeta_size = static_cast<uint32_t>(batch.meta.size() * sizeof(QueryMetadata));
                const uint32_t pssm_size = static_cast<uint32_t>(batch.pssm.size());

                for (uint32_t d : active_dpus) {
                    const uint32_t qmeta_off = max_layout.common_data_offset;
                    const uint32_t pssm_off = qmeta_off + DpuCommunicationManager::alignToMram(qmeta_size);

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
                        0,
                        limits.tasklet_limit);
                    hdr.batch_id = static_cast<uint32_t>(query_batch_num);

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

                    query_descriptors[d] = bd;
                }

                // DATA-RESIDENT: Only scatter descriptors (targets already in MRAM)
                workflow_.scatterDescriptorsOnly(query_descriptors);

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

                // DPU execution + result gathering with timing
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostDispatcherWait);
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
                            
                            // Parallel gather all ready DPUs at once (reference pattern: rank-level bulk transfer)
                            parallelGatherCombinedResults(dpu_ready);
                            
                            // Clear ready/active flags for gathered DPUs
                            for (uint32_t d = 0; d < num_dpus; ++d) {
                                if (dpu_ready[d]) {
                                    dpu_ready[d] = false;
                                    dpu_active[d] = false;
                                }
                            }
                        }
                    }
                }

                // Count total hits for profiling
                uint64_t total_hits = 0;
                for (const auto& hits : dpu_hits) total_hits += hits.size();

                std::vector<std::vector<Matcher::result_t>> batch_results(batch.meta.size());
                {
                    auto timer = dpu_comm_.timeSlot(PS::HostProcessHits);
                    processDpuHits(dpu_hits, batch, tdbr, par, evaluer, sameDB, taxonomyHook, batch_results);
                    dpu_comm_.recordSlotMetrics(PS::HostProcessHits, 0, total_hits);
                }

                {
                    auto timer = dpu_comm_.timeSlot(PS::HostResultWrite);
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
            dpu_comm_.dumpProfile("combined_prefilter");
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