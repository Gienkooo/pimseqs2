#include "DpuPrefilterHostPipeline.h"
#include "Debug.h"
#include "Sequence.h"
#include "StripedSmithWaterman.h"
#include "Matcher.h"
#include "QueryMatcher.h"
#include "QueryMatcherTaxonomyHook.h"
#include "SubstitutionMatrix.h"
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

namespace mmseqs::dpu {

DpuPrefilterHostPipeline::DpuPrefilterHostPipeline(uint32_t num_dpus)
    : dpu_comm_(num_dpus) {
    fprintf(stderr, "[DPU HOST] Initialized DPU pipeline with %u DPUs\n", num_dpus);
}

DpuPrefilterHostPipeline::~DpuPrefilterHostPipeline() {
    fprintf(stderr, "[DPU HOST] Shutting down DPU pipeline\n");
}

void DpuPrefilterHostPipeline::runPrefilterOnDpu(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    SequenceLookup* sequenceLookup, bool sameDB, DBWriter& resultWriter,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    int alignmentMode) {

    fprintf(stderr, "[DPU HOST] Dispatch: par.prefMode=%d alignmentMode=%d\n", par.prefMode, alignmentMode);

    if (par.prefMode == Parameters::PREF_MODE_KMER) {
        // Mode 0: K-mer Prefilter
        runDpuKmerBatch(par, subMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    }
    else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED) {
        // Mode 1: Ungapped Alignment
        runDpuUngappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    } 
    else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED) {
        // Mode 2: Gapped Alignment
        runDpuGappedBatch(par, subMat, tinySubMat, qdbr, tdbr, evaluer, taxonomyHook, sameDB, resultWriter);
    } 
    else {
        fprintf(stderr, "[DPU HOST] WARNING: Requested prefilter mode %d is not implemented on DPU. Falling back to CPU or skipping.\n", par.prefMode);
    }
}

static std::vector<KmerEntry> buildKmerIndex(
    const std::vector<std::vector<uint8_t>>& queries,
    uint32_t size,
    int k)
{
    std::vector<KmerEntry> table(size, {0,0,0});
    uint32_t mask = size - 1;
    
    for (size_t qId = 0; qId < queries.size(); ++qId) {
        const auto& seq = queries[qId];
        if (seq.size() < (size_t)k) continue;

        uint32_t kmer = 0;
        for (int i=0; i<k-1; i++) kmer = (kmer << 5) | (seq[i] & 0x1F);

        for (int i=k-1; i<seq.size(); i++) {
            kmer = ((kmer << 5) | (seq[i] & 0x1F)) & 0x3FFFFFFF;
            
            // Simple Hash Insert
            uint32_t idx = kmer & mask;
            for (int p=0; p<4; p++) {
                uint32_t curr = (idx + p) & mask;
                if (table[curr].kmer == 0 && table[curr].query_id == 0) {
                    table[curr] = {kmer, (uint16_t)qId, (uint16_t)i};
                    break;
                }
            }
        }
    }
    return table;
}

// Forward declaration for verification helper defined later
static int16_t verifyScoreOnHost(const std::vector<uint8_t>& target_seq, const std::vector<int8_t>& pssm, uint32_t q_len);

void DpuPrefilterHostPipeline::runDpuKmerBatch(
    Parameters& par, BaseMatrix* subMat, DBReader<unsigned int>* qdbr,
    DBReader<unsigned int>* tdbr, EvalueComputation* evaluer,
    QueryMatcherTaxonomyHook* taxonomyHook, bool sameDB, DBWriter& resultWriter) {
    
    fprintf(stderr, "[DPU HOST] DPU K-mer batch processing.\n");
    
    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) return;

        // Choose kernel path relative to repo root when available (build area),
        // otherwise fallback to in-tree installed path. Print chosen path for debugging.
        const char* kPathKmer = "lib/mmseqs/dpu/kmer_prefilter";
        if (access("build/lib/dpu/kernels/kmer_prefilter", F_OK) != -1) {
            kPathKmer = "build/lib/dpu/kernels/kmer_prefilter";
        }
        fprintf(stderr, "[DPU HOST] Loading DPU kernel (kmer): %s\n", kPathKmer);
        dpu_comm_.loadKernel(kPathKmer);

    uint32_t totalQueries = qdbr->getSize();
    // Batch queries (per-DPU broadcast)
    const uint32_t Q_BATCH_SIZE = 128;

    for (size_t qStart = 0; qStart < totalQueries; qStart += Q_BATCH_SIZE) {
        size_t qEnd = std::min(qStart + Q_BATCH_SIZE, (size_t)totalQueries);
        
        // 1. Prepare query batch and k-mer index
        std::vector<std::vector<uint8_t>> query_batch;
        std::vector<uint32_t> query_keys;
        std::vector<uint32_t> query_lens;

        for (size_t q = qStart; q < qEnd; ++q) {
            size_t len = qdbr->getSeqLen(q);
            const char* seq = qdbr->getData(q, 0);
            
            std::vector<uint8_t> encoded(len);
            for(size_t i=0; i<len; i++) {
                // Convert to Index (same as ungapped fix)
                unsigned char aa = static_cast<unsigned char>(seq[i]);
                encoded[i] = (subMat->aa2num) ? subMat->aa2num[aa] : 20;
                if (encoded[i] >= 21) encoded[i] = 20;
            }
            query_batch.push_back(encoded);
            query_keys.push_back(qdbr->getDbKey(q));
            query_lens.push_back(len);
        }

        // Build hash table (power-of-two size)
        uint32_t table_size = 32768;
        int ksize = par.kmerSize;
        if (ksize <= 0) ksize = 6;
        std::vector<KmerEntry> index = buildKmerIndex(query_batch, table_size, ksize);

        // 2. Split targets among DPUs
        uint32_t totalTargets = tdbr->getSize();
        uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;
        
        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // Precompute global offsets for BD and Hash Table so we can broadcast the
        // static hash table once and then scatter per-DPU descriptors and targets.
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t table_off_global = bd_aligned;
        uint32_t table_size_aligned = DpuCommunicationManager::alignToMram(index.size() * sizeof(KmerEntry));

        // Broadcast the hash table once (padded to aligned size)
        {
            std::vector<uint8_t> index_buf(table_size_aligned, 0);
            memcpy(index_buf.data(), index.data(), index.size() * sizeof(KmerEntry));
            dpu_comm_.broadcastData(index_buf.data(), index_buf.size(), table_off_global);
        }

        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            uint32_t start_t = dpu_idx * targetsPerDpu;
            uint32_t count_t = (start_t >= totalTargets) ? 0 : std::min(targetsPerDpu, totalTargets - start_t);

            // Assemble encoded targets
            std::vector<uint8_t> packed_targets;
            std::vector<TargetMetadata> tmeta;
            assembleTargetBatch(tdbr, start_t, count_t, packed_targets, tmeta, subMat);

            // Offsets (BD at 0, table at table_off_global)
            uint32_t off = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
            uint32_t table_off = table_off_global;
            off = table_off + table_size_aligned;
            uint32_t t_meta_off = off; off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; off += DpuCommunicationManager::alignToMram(packed_targets.size());
            uint32_t res_off = off;
            uint32_t res_size = tmeta.size() * sizeof(Hit);
            off += DpuCommunicationManager::alignToMram(res_size);

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;

            // Descriptor (per-DPU) and scatter to that DPU only
            BatchDescriptor bd;
            memset(&bd, 0, sizeof(bd));
            bd.num_queries = query_batch.size();
            bd.num_targets = count_t;
            bd.query_len = (query_lens.empty()) ? 0 : query_lens[0];
            bd.pssm_data_offset = table_off;
            bd.pssm_total_size = index.size() * sizeof(KmerEntry); // Pass size in bytes
            bd.kmer_size = ksize;
            bd.targets_metadata_offset = t_meta_off;
            bd.targets_data_offset = t_data_off;
            bd.results_offset = res_off;

            // Scatter per-DPU descriptor (do NOT broadcast the per-DPU descriptor)
            dpu_comm_.scatterDataToDPU(dpu_idx, &bd, sizeof(bd), 0);

            // Scatter Targets
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
        // Read and print any DPU-side log output (kernel printf)
        dpu_comm_.readAndPrintLog();
        std::string resultBuffer;

        // 3. Gather results per local query id
        std::vector<std::vector<hit_t>> queryResults(query_batch.size());
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            std::vector<Hit> results(res_sizes[dpu_idx] / sizeof(Hit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                if (hit.score <= 0) continue;
                uint32_t local_q = hit.query_id;
                if (local_q >= query_batch.size()) continue; // guard

                hit_t shortHit;
                shortHit.seqId = tdbr->getDbKey(hit.target_id);
                shortHit.prefScore = hit.score;
                shortHit.diagonal = hit.diagonal;
                queryResults[local_q].push_back(shortHit);
            }
        }

        // 4. Post-process per-query: sort, limit, write
        for (size_t local_q = 0; local_q < queryResults.size(); ++local_q) {
            auto &hits = queryResults[local_q];
            if (hits.empty()) continue;
            std::sort(hits.begin(), hits.end(), hit_t::compareHitsByScoreAndId);
            size_t keep = std::min(hits.size(), (size_t)par.maxResListLen);
            std::string resultBuffer;
            for (size_t i = 0; i < keep; ++i) {
                char outbuf[256];
                size_t len = QueryMatcher::prefilterHitToBuffer(outbuf, hits[i]);
                resultBuffer.append(outbuf, len);
            }
            uint32_t outQueryKey = query_keys[local_q];
            resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), outQueryKey, 0);
        }
    }
}

void DpuPrefilterHostPipeline::runDpuGappedBatch(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    bool sameDB, DBWriter& resultWriter) {
    
    fprintf(stderr, "[DPU HOST] DPU Ungapped+Gapped batch processing (PREF_MODE_UNGAPPED_AND_GAPPED).\n");

    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) return;

    // 1. Load the Gapped Kernel
    // Ensure this kernel is compiled and available at this path
    const char* kPath = "lib/mmseqs/dpu/gapped_prefilter";
    if (access("build/lib/dpu/kernels/gapped_prefilter", F_OK) != -1) {
        kPath = "build/lib/dpu/kernels/gapped_prefilter";
    }
    fprintf(stderr, "[DPU HOST] Loading DPU kernel (gapped): %s\n", kPath);
    dpu_comm_.loadKernel(kPath);

    // 2. Pre-allocate Composition Bias Buffer
    std::vector<float> compositionBias;
    if (par.compBiasCorrection) {
        compositionBias.resize(qdbr->getMaxSeqLen() + 1, 0.0f);
    }

    // 3. Iterate Over Queries
    for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
        uint32_t queryKey = qdbr->getDbKey(qId);
        uint32_t queryLen = qdbr->getSeqLen(qId);
        const char* querySeq = qdbr->getData(qId, 0);

        // Build PSSM with Bias Correction
        std::vector<int8_t> pssm = buildPSSMFromSequence(
            querySeq, queryLen, subMat, 
            par.compBiasCorrection, par.compBiasCorrectionScale, compositionBias
        );

        QueryMetadata qmeta;
        qmeta.query_id = queryKey;
        qmeta.query_len = queryLen;
        qmeta.pssm_offset_in_batch = 0;
        qmeta.padding = 0;

        uint32_t queries_meta_size = static_cast<uint32_t>(sizeof(QueryMetadata));
        uint32_t pssm_size = static_cast<uint32_t>(pssm.size());

        uint32_t totalTargets = tdbr->getSize();
        uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;

        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // Compute global offsets for QueryMetadata and PSSM (BD at 0)
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t q_meta_off_global = bd_aligned;
        uint32_t q_meta_size_aligned = DpuCommunicationManager::alignToMram(queries_meta_size);
        uint32_t pssm_off_global = q_meta_off_global + q_meta_size_aligned;
        uint32_t pssm_size_aligned = DpuCommunicationManager::alignToMram(pssm.size());

        // Broadcast QueryMeta + PSSM once
        {
            std::vector<uint8_t> common(q_meta_size_aligned + pssm_size_aligned, 0);
            memcpy(common.data(), &qmeta, sizeof(qmeta));
            memcpy(common.data() + q_meta_size_aligned, pssm.data(), pssm.size());
            dpu_comm_.broadcastData(common.data(), common.size(), q_meta_off_global);
        }

        // 4. Prepare Batches for DPUs
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            uint32_t start_t = dpu_idx * targetsPerDpu;
            uint32_t count_t = (start_t >= totalTargets) ? 0 : std::min(targetsPerDpu, totalTargets - start_t);

            std::vector<uint8_t> packed_targets;
            std::vector<TargetMetadata> tmeta;
            
            // Reuse the assembled target batch logic (uses encoded sequences)
            assembleTargetBatch(tdbr, start_t, count_t, packed_targets, tmeta, subMat);

            // Calculate MRAM Offsets (Aligned) using global qmeta/pssm offsets
            uint32_t off = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
            uint32_t q_meta_off = q_meta_off_global;
            uint32_t pssm_off = pssm_off_global;
            off = pssm_off + pssm_size_aligned;
            uint32_t t_meta_off = off; off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; off += DpuCommunicationManager::alignToMram(packed_targets.size());

            uint32_t res_off = off; 
            // Allocate space for GappedHit results
            uint32_t res_size = tmeta.size() * sizeof(GappedHit);
            off += DpuCommunicationManager::alignToMram(res_size);

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;

            // Batch Descriptor (per-DPU) and scatter only to this DPU
            BatchDescriptor bd = {0, 1, count_t, queryLen, q_meta_off, pssm_off, t_meta_off, t_data_off, res_off, (uint32_t)pssm.size(), 0, (uint32_t)packed_targets.size(), res_size};
            dpu_comm_.scatterDataToDPU(dpu_idx, &bd, sizeof(bd), 0);

            // Scatter: Target Meta + Target Data
            if (count_t > 0) {
                std::vector<uint8_t> meta_buf(DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata)), 0);
                memcpy(meta_buf.data(), tmeta.data(), tmeta.size() * sizeof(TargetMetadata));
                dpu_comm_.scatterDataToDPU(dpu_idx, meta_buf.data(), meta_buf.size(), t_meta_off);

                std::vector<uint8_t> data_buf(DpuCommunicationManager::alignToMram(packed_targets.size()), 0);
                memcpy(data_buf.data(), packed_targets.data(), packed_targets.size());
                dpu_comm_.scatterDataToDPU(dpu_idx, data_buf.data(), data_buf.size(), t_data_off);
            }
        }

        // 5. Execute Gapped Kernel
        dpu_comm_.executeKernels();
        // Read and print any DPU-side log output (kernel printf)
        dpu_comm_.readAndPrintLog();

        // 6. Gather and Process Results
        std::string resultBuffer; // buffer for this query (gapped results)
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            
            // Retrieve GappedHit structures
            std::vector<GappedHit> results(res_sizes[dpu_idx] / sizeof(GappedHit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                // Filter low scores (Gapped threshold check should ideally happen on DPU too)
                if (hit.score <= par.minDiagScoreThr) continue;

                // Taxonomy Filter
                if (taxonomyHook != NULL) {
                    TaxID currTax = taxonomyHook->taxonomyMapping->lookup(hit.target_id);
                    if (taxonomyHook->expression[0]->isAncestor(currTax) == false) continue;
                }

                // Construct Full Alignment Result
                Matcher::result_t res;
                res.dbKey = tdbr->getDbKey(hit.target_id);
                res.score = evaluer->computeBitScore(hit.score); // Convert raw score to bitscore
                res.eval = evaluer->computeEvalue(hit.score, queryLen);
                
                // Coordinates returned by DPU (0-indexed)
                res.qEndPos = hit.q_end;
                res.dbEndPos = hit.t_end;
                
                // Lengths
                res.qLen = queryLen;
                res.dbLen = tdbr->getSeqLen(hit.target_id);

                // Coverage (Simplified, as DPU assumes local alignment ending at q_end/t_end)
                // Ideally, DPU returns start positions too, but for prefiltering, this is often sufficient
                res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen); 
                res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen);

                // Filter by E-value and Coverage
                if (!Alignment::checkCriteria(res, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr)) {
                    continue;
                }

                // Write Result
                char buffer[4096];
                // false = no backtrace string (DPU doesn't compute full CIGAR yet)
                size_t len = Matcher::resultToBuffer(buffer, res, false);
                resultBuffer.append(buffer, len);
            }
        }
        // write once per query (even if empty)
        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.size(), queryKey, 0);
    }
}

// === CPU VERIFICATION FUNCTION ===
int16_t verifyScoreOnHost(
    const std::vector<uint8_t>& target_seq,
    const std::vector<int8_t>& pssm,
    uint32_t q_len) {
    
    if (target_seq.empty()) return 0;
    uint32_t t_len = target_seq.size();
    
    // Replicate DPU Diagonal Logic
    std::vector<int16_t> diag_buffer(q_len + t_len, 0);
    int16_t max_score = 0;
    const int KERNEL_AA_SLOTS = 21;

    for (uint32_t q = 0; q < q_len; ++q) {
        // PSSM Row
        const int8_t* pssm_row = &pssm[q * KERNEL_AA_SLOTS];
        
        for (uint32_t t = 0; t < t_len; ++t) {
            uint8_t aa = target_seq[t];
            if (aa >= KERNEL_AA_SLOTS) aa = 20; // Clamp

            int8_t score = pssm_row[aa];
            int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
            
            if (diag_idx >= 0 && diag_idx < diag_buffer.size()) {
                int16_t prev = diag_buffer[diag_idx];
                int16_t curr = prev + score;
                if (curr < 0) curr = 0;
                diag_buffer[diag_idx] = curr;
                if (curr > max_score) max_score = curr;
            }
        }
    }
    return max_score;
}

void DpuPrefilterHostPipeline::runDpuUngappedBatch(
    Parameters& par, BaseMatrix* subMat, int8_t* tinySubMat,
    DBReader<unsigned int>* qdbr, DBReader<unsigned int>* tdbr,
    EvalueComputation* evaluer, QueryMatcherTaxonomyHook* taxonomyHook,
    bool sameDB, DBWriter& resultWriter) {
    
    fprintf(stderr, "[DPU HOST] DPU ungapped batch processing.\n");

    const uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
    if (num_dpus == 0) return;

    std::vector<float> compositionBias;
    if (par.compBiasCorrection) {
        compositionBias.resize(qdbr->getMaxSeqLen() + 1, 0.0f);
    }

    for (size_t qId = 0; qId < qdbr->getSize(); ++qId) {
        uint32_t queryKey = qdbr->getDbKey(qId);
        uint32_t queryLen = qdbr->getSeqLen(qId);
        const char* querySeq = qdbr->getData(qId, 0);

        std::vector<int8_t> pssm = buildPSSMFromSequence(
            querySeq, queryLen, subMat, 
            par.compBiasCorrection, par.compBiasCorrectionScale, compositionBias
        );

        // Load Kernel
        const char* kPath = "lib/mmseqs/dpu/ungapped_prefilter";
        if (access("build/lib/dpu/kernels/ungapped_prefilter", F_OK) != -1) {
            kPath = "build/lib/dpu/kernels/ungapped_prefilter";
        }
        fprintf(stderr, "[DPU HOST] Loading DPU kernel (ungapped): %s\n", kPath);
        dpu_comm_.loadKernel(kPath);

        uint32_t totalTargets = tdbr->getSize();
        uint32_t targetsPerDpu = (totalTargets + num_dpus - 1) / num_dpus;

        std::vector<uint32_t> res_offsets(num_dpus), res_sizes(num_dpus);

        // Compute global offsets for QueryMetadata and PSSM (BD at 0)
        uint32_t bd_aligned = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
        uint32_t q_meta_off_global = bd_aligned;
        uint32_t q_meta_size_aligned = DpuCommunicationManager::alignToMram(sizeof(QueryMetadata));
        uint32_t pssm_off_global = q_meta_off_global + q_meta_size_aligned;
        uint32_t pssm_size_aligned = DpuCommunicationManager::alignToMram(pssm.size());

        // Prepare and broadcast QueryMeta + PSSM once for all DPUs
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

            // Optional host-side verification for the first target
            if (dpu_idx == 0 && !packed_targets.empty() && count_t > 0) {
                uint32_t len = tmeta[0].target_len;
                std::vector<uint8_t> first_tgt(packed_targets.begin(), packed_targets.begin() + len);
                int16_t host_score = verifyScoreOnHost(first_tgt, pssm, queryLen);
                fprintf(stderr, "[DPU HOST][VERIFY] Q=%u T=%u Length=%u HostScore=%d\n", queryKey, tdbr->getDbKey(tmeta[0].target_id), len, host_score);
            }

            // Calculate Offsets (using global offsets for qmeta and pssm)
            uint32_t off = DpuCommunicationManager::alignToMram(sizeof(BatchDescriptor));
            uint32_t q_meta_off = q_meta_off_global;
            uint32_t pssm_off = pssm_off_global;
            off = pssm_off + pssm_size_aligned;
            uint32_t t_meta_off = off; off += DpuCommunicationManager::alignToMram(tmeta.size() * sizeof(TargetMetadata));
            uint32_t t_data_off = off; off += DpuCommunicationManager::alignToMram(packed_targets.size());
            uint32_t res_off = off; 
            uint32_t res_size = tmeta.size() * sizeof(Hit);
            off += DpuCommunicationManager::alignToMram(res_size);

            res_offsets[dpu_idx] = res_off;
            res_sizes[dpu_idx] = res_size;

            // Prepare per-DPU Descriptor and scatter it only to that DPU
            BatchDescriptor bd = {0, 1, count_t, queryLen, q_meta_off, pssm_off, t_meta_off, t_data_off, res_off, (uint32_t)pssm.size(), 0, (uint32_t)packed_targets.size(), res_size};
            dpu_comm_.scatterDataToDPU(dpu_idx, &bd, sizeof(bd), 0);

            // Scatter
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
        // Read and print any DPU-side log output (kernel printf)
        dpu_comm_.readAndPrintLog();

        // Gather & buffer results for this single query (queryKey)
        std::vector<hit_t> queryResults;
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (res_sizes[dpu_idx] == 0) continue;
            std::vector<Hit> results(res_sizes[dpu_idx] / sizeof(Hit));
            dpu_comm_.gatherDataFromDPU(dpu_idx, results.data(), res_sizes[dpu_idx], res_offsets[dpu_idx]);

            for (const auto& hit : results) {
                if (hit.score <= par.minDiagScoreThr) continue;
                if (taxonomyHook && !taxonomyHook->expression[0]->isAncestor(taxonomyHook->taxonomyMapping->lookup(hit.target_id))) continue;

                // CPU verification: re-encode the target sequence and recompute ungapped score
                // uint32_t tLen = tdbr->getSeqLen(hit.target_id);
                // const char* tSeq = tdbr->getData(hit.target_id, 0);
                // std::vector<uint8_t> enc_t;
                // enc_t.reserve(tLen);
                // for (uint32_t ii = 0; ii < tLen; ++ii) {
                //     unsigned char aa = static_cast<unsigned char>(tSeq[ii]);
                //     if (aa >= 'a' && aa <= 'z') aa -= 32;
                //     int num_aa = subMat->aa2num ? subMat->aa2num[aa] : 20;
                //     if (num_aa >= 21) num_aa = 20;
                //     enc_t.push_back((uint8_t)num_aa);
                // }

                // int16_t host_score = verifyScoreOnHost(enc_t, pssm, queryLen);

                // // Log both DPU and CPU scores for comparison
                // Debug(Debug::INFO) << "[VERIFY] Q=" << queryKey << " T=" << tdbr->getDbKey(hit.target_id)
                //                    << " DpuScore=" << hit.score << " CpuScore=" << host_score
                //                    << " Diag=" << hit.diagonal << "\n";

                hit_t shortHit;
                shortHit.seqId = tdbr->getDbKey(hit.target_id);
                shortHit.prefScore = hit.score;
                shortHit.diagonal = hit.diagonal;
                queryResults.push_back(shortHit);
            }
        }

        // Sort, limit, and buffer results per-query, then write once (even if empty)
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
            // Apply soft-masking normalization like Sequence::mapSequence
            // Only convert lowercase to uppercase; avoid turning all chars into '!'
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