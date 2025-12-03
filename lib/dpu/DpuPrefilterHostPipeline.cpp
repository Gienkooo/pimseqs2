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
#include <cstring>
#include <unistd.h>
#include <limits.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mmseqs::dpu {

DpuPrefilterHostPipeline::DpuPrefilterHostPipeline(uint32_t num_dpus)
    : dpu_comm_(num_dpus) {
    Debug(Debug::INFO) << "[DPU] Initialized pipeline with " << num_dpus << " DPUs\n";
}

DpuPrefilterHostPipeline::~DpuPrefilterHostPipeline() {}

void DpuPrefilterHostPipeline::runPrefilterOnDpu(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    SequenceLookup* sequenceLookup, bool sameDB, DBWriter& resultWriter,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    int alignmentMode) {

    Debug(Debug::INFO) << "[DPU] Dispatch: prefMode=" << par.prefMode << " alignMode=" << alignmentMode << "\n";

    if (alignmentMode == 1 || par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE || 
        par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED) {
        runDpuGappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    } else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED) {
        runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    } else if (par.prefMode == Parameters::PREF_MODE_KMER) {
        if (alignmentMode == 0 && tinySubMat != NULL) {
            runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        } else {
            runDpuKmerBatch(par, subMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
        }
    } else {
        Debug(Debug::WARNING) << "[DPU] Mode " << par.prefMode << " not supported, falling back to ungapped\n";
        runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    }
}

static std::vector<KmerEntry> buildQueryKmerHashTableWithSimilar(
    const std::vector<uint8_t>& query,
    uint32_t table_size,
    int k,
    KmerGenerator* kmerGen,
    Indexer* indexer,
    size_t& totalKmersInserted,
    const uint8_t* spacedPattern = nullptr,
    int patternSpan = 0)
{
    std::vector<KmerEntry> table(table_size, {0, 0, 0});
    uint32_t mask = table_size - 1;
    totalKmersInserted = 0;
    size_t totalKmersAttempted = 0;
    
    int windowSize = (spacedPattern && patternSpan > 0) ? patternSpan : k;
    if (query.size() < (size_t)windowSize) return table;

    unsigned char kmerBuf[32];
    
    for (size_t pos = 0; pos <= query.size() - windowSize; pos++) {
        const unsigned char* kmer;
        
        if (spacedPattern && patternSpan > 0) {
            for (int j = 0; j < k; j++) {
                kmerBuf[j] = query[pos + spacedPattern[j]];
            }
            kmer = kmerBuf;
        } else {
            kmer = query.data() + pos;
        }
        
        std::pair<size_t*, size_t> kmerList = kmerGen->generateKmerList(kmer);
        totalKmersAttempted += kmerList.second;
        
        for (size_t i = 0; i < kmerList.second; i++) {
            uint32_t kmerVal = (uint32_t)kmerList.first[i];
            uint32_t idx = kmerVal & mask;
            for (int p = 0; p < 256; p++) {
                uint32_t slot = (idx + p) & mask;
                if (table[slot].kmer == 0) {
                    table[slot].kmer = kmerVal;
                    table[slot].query_id = 0;
                    table[slot].query_pos = (uint16_t)pos;
                    totalKmersInserted++;
                    break;
                }
                if (table[slot].kmer == kmerVal) {
                    break;
                }
            }
        }
    }
    
    if (totalKmersInserted < totalKmersAttempted) {
        Debug(Debug::WARNING) << "[DPU] Hash table collision/full: inserted " << totalKmersInserted 
                              << " of " << totalKmersAttempted << " kmers (" 
                              << (totalKmersAttempted - totalKmersInserted) << " dropped)\n";
    }
    
    return table;
}

void DpuPrefilterHostPipeline::runDpuKmerBatch(
    Parameters& par, BaseMatrix* subMat, DBReader<unsigned int>* qdbr,
    DBReader<unsigned int>* tdbr, EvalueComputation* evaluer,
    QueryMatcherTaxonomyHook* taxonomyHook, bool sameDB, DBWriter& resultWriter) {
    
    Debug(Debug::INFO) << "[DPU] K-mer prefilter: " << qdbr->getSize() << " queries, " << tdbr->getSize() << " targets\n";
    
    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) {
        Debug(Debug::ERROR) << "[DPU] No active DPUs available\n";
        return;
    }

    const char* kPathKmer = "lib/mmseqs/dpu/kmer_prefilter";
    if (access("build/lib/dpu/kernels/kmer_prefilter", F_OK) != -1) {
        kPathKmer = "build/lib/dpu/kernels/kmer_prefilter";
    }
    dpu_comm_.loadKernel(kPathKmer);

    int ksize = par.kmerSize;
    if (ksize <= 0) ksize = 6;
    
    int kmerThr = par.kmerScore.values.sequence();
    if (kmerThr == INT_MAX) {
        kmerThr = 112;
    }
    
    int alphabetSize = subMat->alphabetSize;
    KmerGenerator* kmerGen = new KmerGenerator(ksize, alphabetSize, (short)kmerThr);
    Indexer* indexer = new Indexer(alphabetSize, ksize);
    
    ScoreMatrix extMatTwo = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 2);
    ScoreMatrix extMatThree = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 3);
    kmerGen->setDivideStrategy(&extMatThree, &extMatTwo);
    
    bool useSpacedKmers = (par.spacedKmer != 0);
    uint8_t spacedPattern[16] = {0};
    int patternSpan = ksize;
    
    if (useSpacedKmers) {
        const int8_t* rawPattern = nullptr;
        int rawPatternLen = 0;
        
        switch (ksize) {
            case 6:  rawPattern = spaced_seed_6;  rawPatternLen = sizeof(spaced_seed_6);  break;
            case 7:  rawPattern = spaced_seed_7;  rawPatternLen = sizeof(spaced_seed_7);  break;
            case 8:  rawPattern = spaced_seed_8;  rawPatternLen = sizeof(spaced_seed_8);  break;
            case 9:  rawPattern = spaced_seed_9;  rawPatternLen = sizeof(spaced_seed_9);  break;
            case 10: rawPattern = spaced_seed_10; rawPatternLen = sizeof(spaced_seed_10); break;
            default: useSpacedKmers = false; break;
        }
        
        if (rawPattern && rawPatternLen > 0) {
            int patternIdx = 0;
            for (int i = 0; i < rawPatternLen && patternIdx < ksize; i++) {
                if (rawPattern[i]) {
                    spacedPattern[patternIdx++] = (uint8_t)i;
                }
            }
            patternSpan = rawPatternLen;
            Debug(Debug::INFO) << "[DPU] Spaced k-mers: k=" << ksize << " span=" << patternSpan << "\n";
        } else {
            useSpacedKmers = false;
        }
    }
    
    const uint32_t HASH_TABLE_SIZE = 2097152;  // 2M entries for similar k-mers
    
    uint64_t totalCells = 0;
    uint64_t totalHits = 0;
    uint64_t totalSimilarKmers = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    uint64_t targetTotalLen = 0;
    for (size_t t = 0; t < tdbr->getSize(); ++t) {
        targetTotalLen += tdbr->getSeqLen(t);
    }
    
    uint32_t maxQueryLen = 0;

    for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
        // SIMULATOR: Reload kernel to reset WRAM state between queries
        dpu_comm_.loadKernel(kPathKmer);
        
        uint32_t queryKey = qdbr->getDbKey(qId);
        uint32_t queryLen = qdbr->getSeqLen(qId);
        const char* querySeq = qdbr->getData(qId, 0);
        
        if (queryLen > maxQueryLen) maxQueryLen = queryLen;
        totalCells += (uint64_t)queryLen * targetTotalLen;

        std::vector<uint8_t> encodedQuery(queryLen);
        for (size_t i = 0; i < queryLen; i++) {
            unsigned char aa = static_cast<unsigned char>(querySeq[i]);
            encodedQuery[i] = (subMat->aa2num) ? subMat->aa2num[aa] : 20;
            if (encodedQuery[i] >= 21) encodedQuery[i] = 20;
        }

        size_t kmersInserted = 0;
        std::vector<KmerEntry> hashTable = buildQueryKmerHashTableWithSimilar(
            encodedQuery, HASH_TABLE_SIZE, ksize, kmerGen, indexer, kmersInserted,
            useSpacedKmers ? spacedPattern : nullptr, patternSpan);
        totalSimilarKmers += kmersInserted;

        Debug(Debug::INFO) << "[DPU] Kmer query " << (qId + 1) << "/" << qdbr->getSize()
                           << " (len=" << queryLen << ", kmers=" << kmersInserted << ")\n";

        // Split targets among DPUs
        uint32_t totalTargets = tdbr->getSize();
        uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;
        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // MRAM layout: [BatchDescriptor][HashTable][TargetMeta][TargetData][Results]
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t hash_off = bd_aligned;
        uint32_t hash_size_aligned = DpuCommunicationManager::alignToMram(hashTable.size() * sizeof(KmerEntry));

        // Broadcast hash table
        {
            std::vector<uint8_t> hash_buf(hash_size_aligned, 0);
            memcpy(hash_buf.data(), hashTable.data(), hashTable.size() * sizeof(KmerEntry));
            dpu_comm_.broadcastData(hash_buf.data(), hash_buf.size(), hash_off);
        }

        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            uint32_t start_t = dpu_idx * targetsPerDpu;
            uint32_t count_t = (start_t >= totalTargets) ? 0 : std::min(targetsPerDpu, totalTargets - start_t);

            // Assemble encoded targets
            std::vector<uint8_t> packed_targets;
            std::vector<TargetMetadata> tmeta;
            assembleTargetBatch(tdbr, start_t, count_t, packed_targets, tmeta, subMat);

            // Compute offsets
            uint32_t off = hash_off + hash_size_aligned;
            uint32_t t_meta_off = off; off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; off += DpuCommunicationManager::alignToMram(packed_targets.size());
            uint32_t res_off = off;
            uint32_t res_size = tmeta.size() * sizeof(Hit);
            off += DpuCommunicationManager::alignToMram(res_size);

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;

            BatchDescriptor bd;
            memset(&bd, 0, sizeof(bd));
            bd.num_queries = 1;
            bd.num_targets = count_t;
            bd.query_len = queryLen;
            bd.pssm_data_offset = hash_off;
            bd.pssm_total_size = HASH_TABLE_SIZE;
            bd.kmer_size = ksize;
            bd.targets_metadata_offset = t_meta_off;
            bd.targets_data_offset = t_data_off;
            bd.results_offset = res_off;
            memcpy(bd.spaced_pattern, spacedPattern, sizeof(bd.spaced_pattern));
            bd.spaced_pattern_span = (uint8_t)patternSpan;
            bd.use_spaced_kmers = useSpacedKmers ? 1 : 0;

            dpu_comm_.scatterDataToDPU(dpu_idx, &bd, bd_aligned, 0);

            // Scatter targets
            if (count_t > 0) {
                std::vector<uint8_t> meta_buf(DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata)), 0);
                memcpy(meta_buf.data(), tmeta.data(), tmeta.size() * sizeof(TargetMetadata));
                dpu_comm_.scatterDataToDPU(dpu_idx, meta_buf.data(), meta_buf.size(), t_meta_off);

                std::vector<uint8_t> data_buf(DpuCommunicationManager::alignToMram(packed_targets.size()), 0);
                memcpy(data_buf.data(), packed_targets.data(), packed_targets.size());
                dpu_comm_.scatterDataToDPU(dpu_idx, data_buf.data(), data_buf.size(), t_data_off);
            }
        }

        dpu_comm_.executeKernels();

        std::vector<hit_t> hits;
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            std::vector<Hit> results(res_sizes[dpu_idx] / sizeof(Hit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                if (hit.score <= 0) continue;
                totalHits++;

                hit_t shortHit;
                shortHit.seqId = tdbr->getDbKey(hit.target_id);
                shortHit.prefScore = hit.score;
                shortHit.diagonal = hit.diagonal;
                hits.push_back(shortHit);
            }
        }

        // Sort, limit, write
        if (!hits.empty()) {
            std::sort(hits.begin(), hits.end(), hit_t::compareHitsByScoreAndId);
            size_t keep = std::min(hits.size(), (size_t)par.maxResListLen);
            std::string resultBuffer;
            for (size_t i = 0; i < keep; ++i) {
                char outbuf[256];
                size_t len = QueryMatcher::prefilterHitToBuffer(outbuf, hits[i]);
                resultBuffer.append(outbuf, len);
            }
            resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(endTime - startTime).count();
    double gcups = (double)totalCells / seconds / 1e9;
    
    Debug(Debug::INFO) << "[DPU] Kmer complete: " << gcups << " GCUPS, " << totalHits << " hits in " << seconds << "s\n";
    
    ExtendedSubstitutionMatrix::freeScoreMatrix(extMatTwo);
    ExtendedSubstitutionMatrix::freeScoreMatrix(extMatThree);
    delete kmerGen;
    delete indexer;
}

void DpuPrefilterHostPipeline::runDpuGappedBatch(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    bool sameDB, DBWriter& resultWriter) {
    
    Debug(Debug::INFO) << "[DPU] Gapped batch: " << qdbr->getSize() << " queries, " << tdbr->getSize() << " targets\n";

    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) {
        Debug(Debug::ERROR) << "[DPU] No active DPUs available\n";
        return;
    }

    const char* kPath = "lib/mmseqs/dpu/gapped_prefilter";
    if (access("build/lib/dpu/kernels/gapped_prefilter", F_OK) != -1) {
        kPath = "build/lib/dpu/kernels/gapped_prefilter";
    }
    dpu_comm_.loadKernel(kPath);

    std::vector<float> compositionBias;
    if (par.compBiasCorrection) {
        compositionBias.resize(qdbr->getMaxSeqLen() + 1, 0.0f);
    }

    uint32_t batch_flags = (par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE) ? 1 : 0;
    
    uint64_t totalCells = 0;
    uint64_t totalHits = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    uint64_t targetTotalLen = 0;
    for (size_t t = 0; t < tdbr->getSize(); ++t) {
        targetTotalLen += tdbr->getSeqLen(t);
    }
    
    // Karlin-Altschul parameters for BLOSUM62 with gap penalties (11, 1)
    const double K = 0.041;
    const double lambda = 0.267;
    const double n = static_cast<double>(targetTotalLen);
    const double targetEvalue = par.evalThr > 0.0 ? par.evalThr : 0.001;
    const int16_t MIN_SCORE_FLOOR = 15;
    
    uint32_t totalTargets = tdbr->getSize();
    uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;
    
    // Pre-assemble target data for each DPU (optimization: targets don't change per query)
    std::vector<std::vector<uint8_t>> perDpuTargetData(num_dpus);
    std::vector<std::vector<TargetMetadata>> perDpuTargetMeta(num_dpus);
    std::vector<uint32_t> perDpuTargetCount(num_dpus);
    
    #pragma omp parallel for schedule(dynamic) if(num_dpus > 1)
    for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
        uint32_t start_t = dpu_idx * targetsPerDpu;
        uint32_t count_t = (start_t >= totalTargets) ? 0 : std::min(targetsPerDpu, totalTargets - start_t);
        perDpuTargetCount[dpu_idx] = count_t;
        
        if (count_t > 0) {
            assembleTargetBatch(tdbr, start_t, count_t, perDpuTargetData[dpu_idx], 
                              perDpuTargetMeta[dpu_idx], subMat);
        }
    }

    for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
        // SIMULATOR ONLY: Reload kernel to reset DPU WRAM state (diag buffers, etc.)
        // On real hardware, WRAM is zeroed on boot, but simulator preserves state between launches.
        // TODO: Remove this reload when running on real UPMEM hardware for better performance.
        dpu_comm_.loadKernel(kPath);
        
        uint32_t queryKey = qdbr->getDbKey(qId);
        uint32_t queryLen = qdbr->getSeqLen(qId);
        const char* querySeq = qdbr->getData(qId, 0);
        
        totalCells += (uint64_t)queryLen * targetTotalLen;

        // Dynamic threshold: S >= (ln(K * m * n) - ln(E)) / lambda
        double m = static_cast<double>(queryLen);
        // TODO: apply log arthmetic to move constants out of this loop
        int16_t minScoreThreshold = static_cast<int16_t>((std::log(K * m * n) - std::log(targetEvalue)) / lambda);
        if (minScoreThreshold < MIN_SCORE_FLOOR) {
            minScoreThreshold = MIN_SCORE_FLOOR;
        }   
        
        if (qId == 0) {
            Debug(Debug::INFO) << "[DPU] Karlin-Altschul: K=" << K << " lambda=" << lambda 
                               << " dbSize=" << n << " E=" << targetEvalue << "\n";
        }

        std::vector<int8_t> pssm = buildPSSMFromSequence(
            querySeq, queryLen, subMat, 
            par.compBiasCorrection, par.compBiasCorrectionScale, compositionBias
        );

        QueryMetadata qmeta;
        qmeta.query_id = queryKey;
        qmeta.query_len = queryLen;
        qmeta.pssm_offset_in_batch = 0;
        qmeta.padding = 0;

        uint32_t queries_meta_size = sizeof(QueryMetadata);
        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // MRAM layout: [BD][QueryMeta][PSSM][TargetMeta][TargetData][Results]
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t q_meta_off_global = bd_aligned;
        uint32_t pssm_off_global = q_meta_off_global + DpuCommunicationManager::alignToMram(queries_meta_size);
        uint32_t pssm_size_aligned = DpuCommunicationManager::alignToMram(pssm.size());

        // Broadcast query data to all DPUs
        {
            std::vector<uint8_t> common(DpuCommunicationManager::alignToMram(queries_meta_size) + pssm_size_aligned, 0);
            memcpy(common.data(), &qmeta, sizeof(qmeta));
            memcpy(common.data() + DpuCommunicationManager::alignToMram(queries_meta_size), pssm.data(), pssm.size());
            dpu_comm_.broadcastData(common.data(), common.size(), q_meta_off_global);
        }

        Debug(Debug::INFO) << "[DPU] Query " << (qId + 1) << "/" << qdbr->getSize()
                           << " (len=" << queryLen << ", minScore=" << minScoreThreshold << ")\n";

        std::vector<BatchDescriptor> perDpuBD(num_dpus);
        std::vector<uint32_t> perDpuTMetaOff(num_dpus);
        std::vector<uint32_t> perDpuTDataOff(num_dpus);
        
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            uint32_t count_t = perDpuTargetCount[dpu_idx];
            const auto& packed_targets = perDpuTargetData[dpu_idx];
            const auto& tmeta = perDpuTargetMeta[dpu_idx];

            uint32_t off = pssm_off_global + pssm_size_aligned;
            uint32_t t_meta_off = off; 
            off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; 
            off += DpuCommunicationManager::alignToMram(packed_targets.size());
            uint32_t res_off = off; 
            uint32_t res_size = DpuCommunicationManager::alignToMram(count_t * sizeof(GappedHit));

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;
            perDpuTMetaOff[dpu_idx] = t_meta_off;
            perDpuTDataOff[dpu_idx] = t_data_off;

            memset(&perDpuBD[dpu_idx], 0, sizeof(BatchDescriptor));
            perDpuBD[dpu_idx].num_queries = 1;
            perDpuBD[dpu_idx].num_targets = count_t;
            perDpuBD[dpu_idx].query_len = queryLen;
            perDpuBD[dpu_idx].queries_metadata_offset = q_meta_off_global;
            perDpuBD[dpu_idx].pssm_data_offset = pssm_off_global;
            perDpuBD[dpu_idx].targets_metadata_offset = t_meta_off;
            perDpuBD[dpu_idx].targets_data_offset = t_data_off;
            perDpuBD[dpu_idx].results_offset = res_off;
            perDpuBD[dpu_idx].pssm_total_size = (uint32_t)pssm.size();
            perDpuBD[dpu_idx].targets_total_size = (uint32_t)packed_targets.size();
            perDpuBD[dpu_idx].results_buffer_size = res_size;
            perDpuBD[dpu_idx].flags = batch_flags;
            perDpuBD[dpu_idx].min_score = minScoreThreshold;
            perDpuBD[dpu_idx].min_score_padding = 0;
        }

        // Transfer data to all DPUs
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            dpu_comm_.scatterDataToDPU(dpu_idx, &perDpuBD[dpu_idx], 
                                       DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor)), 0);
            
            uint32_t count_t = perDpuTargetCount[dpu_idx];
            if (count_t > 0) {
                const auto& tmeta = perDpuTargetMeta[dpu_idx];
                const auto& packed_targets = perDpuTargetData[dpu_idx];
                
                std::vector<uint8_t> meta_buf(DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata)), 0);
                memcpy(meta_buf.data(), tmeta.data(), tmeta.size() * sizeof(TargetMetadata));
                dpu_comm_.scatterDataToDPU(dpu_idx, meta_buf.data(), meta_buf.size(), perDpuTMetaOff[dpu_idx]);

                std::vector<uint8_t> data_buf(DpuCommunicationManager::alignToMram(packed_targets.size()), 0);
                memcpy(data_buf.data(), packed_targets.data(), packed_targets.size());
                dpu_comm_.scatterDataToDPU(dpu_idx, data_buf.data(), data_buf.size(), perDpuTDataOff[dpu_idx]);
            }
        }

        dpu_comm_.executeKernels();

        // Gather results
        std::string resultBuffer; 
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            std::vector<GappedHit> results(res_sizes[dpu_idx] / sizeof(GappedHit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                if (hit.score <= par.minDiagScoreThr) continue;
                totalHits++;

                if (par.prefMode == Parameters::PREF_MODE_EXHAUSTIVE || par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED) {
                    hit_t shortHit;
                    shortHit.seqId = tdbr->getDbKey(hit.target_id);
                    shortHit.prefScore = hit.score;
                    shortHit.diagonal = (int16_t)(hit.t_end - hit.q_end);
                    char buffer[256];
                    size_t len = QueryMatcher::prefilterHitToBuffer(buffer, shortHit);
                    resultBuffer.append(buffer, len);
                } 
            }
        }
        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    double gcups = (double)totalCells / elapsedSec / 1e9;
    
    Debug(Debug::INFO) << "[DPU] Gapped complete: " << gcups << " GCUPS, " << totalHits << " hits in " << elapsedSec << "s\n";
}

void DpuPrefilterHostPipeline::runDpuUngappedBatch(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    bool sameDB, DBWriter& resultWriter) {
    
    Debug(Debug::INFO) << "[DPU] Ungapped batch: " << qdbr->getSize() << " queries, " << tdbr->getSize() << " targets\n";

    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) {
        Debug(Debug::ERROR) << "[DPU] No active DPUs available\n";
        return;
    }

    const char* kPath = "lib/mmseqs/dpu/ungapped_prefilter";
    if (access("build/lib/dpu/kernels/ungapped_prefilter", F_OK) != -1) {
        kPath = "build/lib/dpu/kernels/ungapped_prefilter";
    }
    dpu_comm_.loadKernel(kPath);

    std::vector<float> compositionBias;
    if (par.compBiasCorrection) {
        compositionBias.resize(qdbr->getMaxSeqLen() + 1, 0.0f);
    }

    for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
        // SIMULATOR ONLY: Reload kernel to reset DPU WRAM state (diag buffers, etc.)
        // On real hardware, WRAM is zeroed on boot, but simulator preserves state between launches.
        // TODO: Remove this reload when running on real UPMEM hardware for better performance.
        dpu_comm_.loadKernel(kPath);
        
        uint32_t queryKey = qdbr->getDbKey(qId);
        uint32_t queryLen = qdbr->getSeqLen(qId);
        const char* querySeq = qdbr->getData(qId, 0);

        std::vector<int8_t> pssm = buildPSSMFromSequence(
            querySeq, queryLen, subMat, 
            par.compBiasCorrection, par.compBiasCorrectionScale, compositionBias
        );

        uint32_t totalTargets = tdbr->getSize();
        uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;

        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // MRAM layout: [BD][QueryMeta][PSSM][TargetMeta][TargetData][Results]
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t q_meta_off_global = bd_aligned;
        uint32_t q_meta_size_aligned = DpuCommunicationManager::alignToMram(sizeof(QueryMetadata));
        uint32_t pssm_off_global = q_meta_off_global + q_meta_size_aligned;
        uint32_t pssm_size_aligned = DpuCommunicationManager::alignToMram(pssm.size());

        // Broadcast query data
        QueryMetadata qm = {queryKey, queryLen, 0, 0};
        {
            std::vector<uint8_t> common(q_meta_size_aligned + pssm_size_aligned, 0);
            memcpy(common.data(), &qm, sizeof(qm));
            memcpy(common.data() + q_meta_size_aligned, pssm.data(), pssm.size());
            dpu_comm_.broadcastData(common.data(), common.size(), q_meta_off_global);
        }

        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            uint32_t start_t = dpu_idx * targetsPerDpu;
            uint32_t count_t = (start_t >= totalTargets) ? 0 : std::min(targetsPerDpu, totalTargets - start_t);

            std::vector<uint8_t> packed_targets;
            std::vector<TargetMetadata> tmeta;
            assembleTargetBatch(tdbr, start_t, count_t, packed_targets, tmeta, subMat);

            uint32_t off = pssm_off_global + pssm_size_aligned;
            uint32_t t_meta_off = off; off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; off += DpuCommunicationManager::alignToMram(packed_targets.size());
            uint32_t res_off = off;
            uint32_t res_size = tmeta.size() * sizeof(Hit);
            off += DpuCommunicationManager::alignToMram(res_size);

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;

            BatchDescriptor bd;
            memset(&bd, 0, sizeof(bd));
            bd.num_queries = 1;
            bd.num_targets = count_t;
            bd.query_len = queryLen;
            bd.queries_metadata_offset = q_meta_off_global;
            bd.pssm_data_offset = pssm_off_global;
            bd.targets_metadata_offset = t_meta_off;
            bd.targets_data_offset = t_data_off;
            bd.results_offset = res_off;
            bd.pssm_total_size = (uint32_t)pssm.size();
            bd.targets_total_size = (uint32_t)packed_targets.size();
            bd.results_buffer_size = res_size;
            dpu_comm_.scatterDataToDPU(dpu_idx, &bd, DpuCommunicationManager::alignToMram(sizeof(bd)), 0);

            // Scatter targets
            if (count_t > 0) {
                std::vector<uint8_t> meta_buf(DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata)), 0);
                memcpy(meta_buf.data(), tmeta.data(), tmeta.size() * sizeof(TargetMetadata));
                dpu_comm_.scatterDataToDPU(dpu_idx, meta_buf.data(), meta_buf.size(), t_meta_off);

                std::vector<uint8_t> data_buf(DpuCommunicationManager::alignToMram(packed_targets.size()), 0);
                memcpy(data_buf.data(), packed_targets.data(), packed_targets.size());
                dpu_comm_.scatterDataToDPU(dpu_idx, data_buf.data(), data_buf.size(), t_data_off);
            }
        }

        dpu_comm_.executeKernels();

        std::vector<hit_t> queryResults;
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            std::vector<Hit> results(res_sizes[dpu_idx] / sizeof(Hit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                if (hit.score <= par.minDiagScoreThr) continue;
                if (taxonomyHook && !taxonomyHook->expression[0]->isAncestor(taxonomyHook->taxonomyMapping->lookup(hit.target_id))) continue;

                hit_t shortHit;
                shortHit.seqId = tdbr->getDbKey(hit.target_id);
                shortHit.prefScore = hit.score;
                shortHit.diagonal = hit.diagonal;
                queryResults.push_back(shortHit);
            }
        }

        if (!queryResults.empty()) {
            std::sort(queryResults.begin(), queryResults.end(), hit_t::compareHitsByScoreAndId);
        }
        size_t keep = std::min(queryResults.size(), (size_t)par.maxResListLen);
        std::string resultBuffer;
        for (size_t i = 0; i < keep; ++i) {
            char buffer[256];
            size_t len = QueryMatcher::prefilterHitToBuffer(buffer, queryResults[i]);
            resultBuffer.append(buffer, len);
        }
        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
    }
}

std::vector<int8_t> DpuPrefilterHostPipeline::buildPSSMFromSequence(
    const char* sequence, uint32_t seq_len, BaseMatrix* subMat,
    bool compBiasCorrection, float compBiasCorrectionScale, std::vector<float>& compositionBias) {
    
    const int KERNEL_AA_SLOTS = 21;
    std::vector<int8_t> pssm(seq_len * KERNEL_AA_SLOTS, 0);
    int alphSize = subMat->alphabetSize;
    
    std::vector<unsigned char> query_indices(seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) {
        unsigned char aa = static_cast<unsigned char>(sequence[i]);
        int qidx = subMat->aa2num ? subMat->aa2num[aa] : 0;
        if (qidx >= alphSize) qidx = 0; 
        query_indices[i] = qidx;
    }

    if (compBiasCorrection) {
        if (compositionBias.size() < seq_len) compositionBias.resize(seq_len, 0.0f);
        SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, query_indices.data(), seq_len, compositionBias.data(), compBiasCorrectionScale);
    }

    for (uint32_t i = 0; i < seq_len; ++i) {
        short bias = 0;
        if (compBiasCorrection) {
            float val = compositionBias[i];
            bias = static_cast<short>((val < 0.0f) ? (val - 0.5f) : (val + 0.5f));
        }
        unsigned char qidx = query_indices[i];
        for (int aa = 0; aa < alphSize && aa < KERNEL_AA_SLOTS; ++aa) {
            short score = subMat->subMatrix[qidx][aa] + bias;
            if (score > 127) score = 127; else if (score < -128) score = -128;
            pssm[i * KERNEL_AA_SLOTS + aa] = static_cast<int8_t>(score);
        }
    }
    return pssm;
}

void DpuPrefilterHostPipeline::assembleTargetBatch(
    DBReader<unsigned int>* tdbr, uint32_t start, uint32_t count,
    std::vector<uint8_t>& packed_sequences,
    std::vector<TargetMetadata>& metadata,
    BaseMatrix* subMat) { 
    
    metadata.clear();
    packed_sequences.clear();
    uint32_t offset = 0;
    
    for (uint32_t i = 0; i < count && (start + i) < tdbr->getSize(); i++) {
        uint32_t target_id = start + i;
        size_t seq_len = 0;
        const char* seq = tdbr->getData(target_id, 0);
        while (seq[seq_len] != '\0') seq_len++;
        
        TargetMetadata meta;
        meta.target_id = target_id;
        meta.target_len = seq_len;
        meta.offset_in_data = offset;
        meta.padding = 0;
        metadata.push_back(meta);
        
        for (size_t j = 0; j < seq_len; j++) {
            unsigned char aa = static_cast<unsigned char>(seq[j]);
            // Normalize lowercase to uppercase
            if (aa >= 'a' && aa <= 'z') aa = aa - 32;
            int num_aa = subMat->aa2num ? subMat->aa2num[aa] : 20;
            if (num_aa >= 21) num_aa = 20;
            packed_sequences.push_back((uint8_t)num_aa);
        }
        while (packed_sequences.size() % 8 != 0) packed_sequences.push_back(0);
        offset = packed_sequences.size();
    }
}

std::vector<Hit> DpuPrefilterHostPipeline::collectResults(uint32_t, uint32_t) { return {}; }

}