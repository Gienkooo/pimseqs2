#include "DpuPrefilterHostPipeline.h"
#include "Debug.h"
#include "Sequence.h"
#include "StripedSmithWaterman.h"
#include "Matcher.h"
#include "QueryMatcher.h"
#include "QueryMatcherTaxonomyHook.h"
#include "SubstitutionMatrix.h"
#include "KmerGenerator.h"
#include "Indexer.h"
#include "ExtendedSubstitutionMatrix.h"
#include "Alignment.h"
#include "DpuKernelManager.h"

#include <cstring>
#include <unistd.h>
#include <limits.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mmseqs::dpu
{

    // Forward declaration for tasklet calculation helper (defined in HELPERS section)
    static inline uint8_t calculateActiveTasklets(uint32_t wram_per_tasklet_bytes);

    // ============================================================================
    // HELPER: Build Kmer Hash Table (Copied from original logic)
    // ============================================================================
    static std::vector<KmerEntry> buildQueryKmerHashTableWithSimilar(
        const std::vector<uint8_t> &query,
        uint32_t table_size,
        int k,
        KmerGenerator *kmerGen,
        Indexer *indexer,
        size_t &totalKmersInserted,
        const uint8_t *spacedPattern = nullptr,
        int patternSpan = 0)
    {
        std::vector<KmerEntry> table(table_size, {0, 0, 0});
        uint32_t mask = table_size - 1;
        totalKmersInserted = 0;
        size_t totalKmersAttempted = 0;

        int windowSize = (spacedPattern && patternSpan > 0) ? patternSpan : k;
        if (query.size() < (size_t)windowSize)
            return table;

        unsigned char kmerBuf[32];

        for (size_t pos = 0; pos <= query.size() - windowSize; pos++)
        {
            const unsigned char *kmer;

            if (spacedPattern && patternSpan > 0)
            {
                for (int j = 0; j < k; j++)
                {
                    kmerBuf[j] = query[pos + spacedPattern[j]];
                }
                kmer = kmerBuf;
            }
            else
            {
                kmer = query.data() + pos;
            }

            std::pair<size_t *, size_t> kmerList = kmerGen->generateKmerList(kmer);
            totalKmersAttempted += kmerList.second;

            for (size_t i = 0; i < kmerList.second; i++)
            {
                uint32_t kmerVal = (uint32_t)kmerList.first[i];
                uint32_t idx = kmerVal & mask;
                for (int p = 0; p < 256; p++)
                {
                    uint32_t slot = (idx + p) & mask;
                    if (table[slot].kmer == 0)
                    {
                        table[slot].kmer = kmerVal;
                        table[slot].query_id = 0;
                        table[slot].query_pos = (uint16_t)pos;
                        totalKmersInserted++;
                        break;
                    }
                    if (table[slot].kmer == kmerVal)
                    {
                        break;
                    }
                }
            }
        }

        return table;
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
        int alignmentMode)
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
            runDpuKmerBatch(par, subMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
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
        Parameters &par, BaseMatrix *subMat,
        DBReader<unsigned int> *qdbr, DBReader<unsigned int> *tdbr,
        EvalueComputation *evaluer, QueryMatcherTaxonomyHook *taxonomyHook,
        bool sameDB, DBWriter &resultWriter)
    {

        Debug(Debug::INFO) << "[DPU] K-mer prefilter: " << qdbr->getSize() << " queries, " << tdbr->getSize() << " targets\n";

        const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
        if (num_dpus == 0)
            return;

        // --- Kmer Generator Setup ---
        int ksize = par.kmerSize;
        if (ksize <= 0)
            ksize = 6;

        int kmerThr = par.kmerScore.values.sequence();
        if (kmerThr == INT_MAX)
            kmerThr = 112;

        int alphabetSize = subMat->alphabetSize;
        KmerGenerator *kmerGen = new KmerGenerator(ksize, alphabetSize, (short)kmerThr);
        Indexer *indexer = new Indexer(alphabetSize, ksize);

        ScoreMatrix extMatTwo = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 2);
        ScoreMatrix extMatThree = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 3);
        kmerGen->setDivideStrategy(&extMatThree, &extMatTwo);

        bool useSpacedKmers = (par.spacedKmer != 0);
        uint8_t spacedPattern[16] = {0};
        int patternSpan = ksize;

        if (useSpacedKmers)
        {
            const int8_t *rawPattern = nullptr;
            int rawPatternLen = 0;

            switch (ksize)
            {
            case 6:
                rawPattern = spaced_seed_6;
                rawPatternLen = 6;
                break;
            case 7:
                rawPattern = spaced_seed_7;
                rawPatternLen = 7;
                break;
            case 8:
                rawPattern = spaced_seed_8;
                rawPatternLen = 8;
                break;
            case 9:
                rawPattern = spaced_seed_9;
                rawPatternLen = 9;
                break;
            case 10:
                rawPattern = spaced_seed_10;
                rawPatternLen = 10;
                break;
            default:
                useSpacedKmers = false;
                break;
            }

            if (rawPattern && rawPatternLen > 0)
            {
                int patternIdx = 0;
                for (int i = 0; i < rawPatternLen && patternIdx < ksize; i++)
                {
                    if (rawPattern[i])
                        spacedPattern[patternIdx++] = (uint8_t)i;
                }
                patternSpan = rawPatternLen;
            }
            else
            {
                useSpacedKmers = false;
            }
        }

        const uint32_t HASH_TABLE_SIZE = 2097152;

        auto perDpuTargetIndices = buildLoadBalancedDistribution(tdbr, num_dpus);
        std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
        std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);

#pragma omp parallel for schedule(dynamic)
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx)
        {
            if (!perDpuTargetIndices[dpu_idx].empty())
            {
                assembleTargetBatchByIndices(tdbr, perDpuTargetIndices[dpu_idx],
                                             perDpuTargetData[dpu_idx],
                                             perDpuTargetMeta[dpu_idx], subMat);
            }
        }

        for (size_t qId = 0; qId < qdbr->getSize(); ++qId)
        {
            kernel_mgr_.loadKernel(DpuKernelManager::KernelType::KMER);

            uint32_t queryKey = qdbr->getDbKey(qId);
            uint32_t queryLen = qdbr->getSeqLen(qId);
            const char *querySeq = qdbr->getData(qId, 0);

            std::vector<uint8_t> encodedQuery(queryLen);
            for (size_t i = 0; i < queryLen; i++)
            {
                unsigned char aa = static_cast<unsigned char>(querySeq[i]);
                encodedQuery[i] = (subMat->aa2num) ? subMat->aa2num[aa] : 20;
                if (encodedQuery[i] >= 21)
                    encodedQuery[i] = 20;
            }

            size_t kmersInserted = 0;
            std::vector<KmerEntry> hashTable = buildQueryKmerHashTableWithSimilar(
                encodedQuery, HASH_TABLE_SIZE, ksize, kmerGen, indexer, kmersInserted,
                useSpacedKmers ? spacedPattern : nullptr, patternSpan);

            uint32_t common_size = hashTable.size() * sizeof(KmerEntry);
            uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(KmerBatchDescriptor));
            workflow_.broadcastCommon(hashTable.data(), common_size, bd_size);

            std::vector<DpuWorkflow::MramLayout> layouts(num_dpus);
            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                layouts[d] = workflow_.calculateLayout(
                    sizeof(KmerBatchDescriptor), common_size, perDpuTargetMeta[d].size(), perDpuTargetData[d].size(), sizeof(Hit));

                KmerBatchDescriptor bd = {};
                bd.header.num_queries = 1;
                bd.header.num_targets = perDpuTargetMeta[d].size();
                bd.header.query_len = queryLen;
                bd.header.pssm_data_offset = layouts[d].common_data_offset; 
                bd.header.pssm_total_size = HASH_TABLE_SIZE;
                bd.header.targets_metadata_offset = layouts[d].target_meta_offset;
                bd.header.targets_data_offset = layouts[d].target_data_offset;
                bd.header.results_offset = layouts[d].results_offset;
                bd.header.results_buffer_size = layouts[d].results_capacity;

                bd.kmer_size = ksize;
                bd.min_score = (int16_t)par.minDiagScoreThr;
                bd.use_spaced_kmers = useSpacedKmers ? 1 : 0;
                bd.spaced_pattern_span = (uint8_t)patternSpan;
                memcpy(bd.spaced_pattern, spacedPattern, sizeof(bd.spaced_pattern));

                // Dynamic tasklets: conservatively estimate ~20KB WRAM per tasklet for KMER
                bd.header.num_active_tasklets = calculateActiveTasklets(20000);

                workflow_.scatterBatch(d, bd, perDpuTargetMeta[d], perDpuTargetData[d], layouts[d]);
            }

            dpu_comm_.executeKernels();

            std::vector<hit_t> hits;
            for (uint32_t d = 0; d < num_dpus; ++d)
            {
                std::vector<Hit> dpuHits = workflow_.gatherResults<Hit>(d, layouts[d].results_offset);
                for (const auto &hit : dpuHits)
                {
                    hit_t shortHit;
                    shortHit.seqId = tdbr->getDbKey(hit.target_id);
                    shortHit.prefScore = hit.score;
                    shortHit.diagonal = hit.diagonal;
                    hits.push_back(shortHit);
                }
            }

            if (!hits.empty())
            {
                std::sort(hits.begin(), hits.end(), hit_t::compareHitsByScoreAndId);
                size_t keep = std::min(hits.size(), (size_t)par.maxResListLen);
                std::string resultBuffer;
                for (size_t i = 0; i < keep; ++i)
                {
                    char outbuf[256];
                    size_t len = QueryMatcher::prefilterHitToBuffer(outbuf, hits[i]);
                    resultBuffer.append(outbuf, len);
                }
                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
            }
        }

        ExtendedSubstitutionMatrix::freeScoreMatrix(extMatTwo);
        ExtendedSubstitutionMatrix::freeScoreMatrix(extMatThree);
        delete kmerGen;
        delete indexer;
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