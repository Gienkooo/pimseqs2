#pragma once

#include "shared/DpuSharedTypes.h"
#include "DpuCommunicationManager.h"
#include "DpuMramAllocator.h"
#include "DpuRankDispatcher.h"
#include "DpuKernelManager.h"

#include "DBReader.h"
#include "DBWriter.h"
#include "BaseMatrix.h"
#include "Parameters.h"
#include "SubstitutionMatrix.h"
#include "Matcher.h"
#include "QueryMatcher.h"

#include <memory>
#include <vector>
#include <future>

class SequenceLookup;
class Sequence;
class EvalueComputation;
class QueryMatcherTaxonomyHook;

namespace mmseqs::dpu {

struct DpuIndexBuffer;

class DpuPrefilterHostPipeline {
 public:
  explicit DpuPrefilterHostPipeline(uint32_t num_dpus);
  ~DpuPrefilterHostPipeline();

  void runPrefilterOnDpu(
      Parameters& par,
      BaseMatrix* subMat,
      int8_t* tinySubMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      SequenceLookup* sequenceLookup,
      bool sameDB,
      DBWriter& resultWriter,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      int alignmentMode,
      int kmerThr,
      ScoreMatrix* extMatTwo,
      ScoreMatrix* extMatThree,
      const std::string& spacedPatternStr,
      bool takeOnlyBestKmer);

 private:
  DpuCommunicationManager dpu_comm_;
  DpuKernelManager kernel_mgr_; 
  
  // Pipeline-native typed communication helpers that replace the duplicate DpuWorkflow middleman
  void broadcastCommonData(const void* data, uint32_t size, uint32_t mram_offset) {
      if (size == 0) return;
      uint32_t aligned = DpuCommunicationManager::alignToMram(size);
      if (size == aligned) {
          dpu_comm_.broadcastData(data, aligned, mram_offset);
      } else {
          std::vector<uint8_t> buf(aligned, 0);
          memcpy(buf.data(), data, size);
          dpu_comm_.broadcastData(buf.data(), aligned, mram_offset);
      }
  }

  template <typename BatchDescT>
  void scatterBatchToDpus(const std::vector<BatchDescT>& descriptors, const std::vector<std::vector<TargetMetadata>>& t_meta, const std::vector<std::vector<uint8_t>>& t_data, uint32_t meta_off, uint32_t data_off) {
      uint32_t num_dpus = descriptors.size();
      if (num_dpus == 0) return;

      std::vector<std::vector<uint8_t>> desc_bufs(num_dpus);
      uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(BatchDescT));
      for (uint32_t i = 0; i < num_dpus; ++i) {
          desc_bufs[i].resize(bd_size);
          memcpy(desc_bufs[i].data(), &descriptors[i], sizeof(BatchDescT));
      }
      dpu_comm_.scatterDataParallel(desc_bufs, 0);

      std::vector<std::vector<uint8_t>> meta_bufs(num_dpus);
      for (uint32_t i = 0; i < num_dpus; ++i) {
          if (!t_meta[i].empty()) {
              uint32_t tmeta_bytes = t_meta[i].size() * sizeof(TargetMetadata);
              meta_bufs[i].resize(tmeta_bytes); 
              memcpy(meta_bufs[i].data(), t_meta[i].data(), tmeta_bytes);
          }
      }
      dpu_comm_.scatterDataParallel(meta_bufs, meta_off);
      dpu_comm_.scatterDataParallel(t_data, data_off);
  }

  void scatterUngappedTargetsOnly(const std::vector<std::vector<TargetMetadata>>& t_meta, const std::vector<std::vector<uint8_t>>& t_data, uint32_t meta_off, uint32_t data_off);
  
  template <typename BatchDescT>
  void scatterDescriptorsOnly(const std::vector<BatchDescT>& descriptors) {
      uint32_t num_dpus = descriptors.size();
      if (num_dpus == 0) return;
      std::vector<std::vector<uint8_t>> desc_bufs(num_dpus);
      uint32_t bd_size = DpuCommunicationManager::alignToMram(sizeof(BatchDescT));
      for (uint32_t i = 0; i < num_dpus; ++i) {
          desc_bufs[i].resize(bd_size);
          memcpy(desc_bufs[i].data(), &descriptors[i], sizeof(BatchDescT));
      }
      dpu_comm_.scatterDataParallel(desc_bufs, 0);
  }

  template <typename HitType>
  std::vector<HitType> gatherResultsClamped(uint32_t dpu_id, uint32_t results_mram_offset, uint32_t result_capacity_bytes, uint32_t& out_overflow) {
      uint64_t hdr = 0;
      dpu_comm_.gatherDataFromDPU(dpu_id, &hdr, 8, results_mram_offset);
      uint32_t hitcount = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
      uint32_t hi = static_cast<uint32_t>((hdr >> 32) & 0xFFFFFFFFu);
      out_overflow = (hi >> 31) & 1u;

      if (result_capacity_bytes <= 8) return {};
      uint32_t maxHits = (result_capacity_bytes - 8) / sizeof(HitType);
      if (hitcount > maxHits) hitcount = maxHits;
      if (hitcount == 0) return {};

      uint32_t hits_offset = results_mram_offset + 8;
      uint32_t data_size = hitcount * sizeof(HitType);
      uint32_t aligned_size = DpuCommunicationManager::alignToMram(data_size);

      std::vector<HitType> hits(hitcount);
      if (aligned_size != data_size) {
          std::vector<uint8_t> buf(aligned_size);
          dpu_comm_.gatherDataFromDPU(dpu_id, buf.data(), aligned_size, hits_offset);
          memcpy(hits.data(), buf.data(), data_size);
      } else {
          dpu_comm_.gatherDataFromDPU(dpu_id, hits.data(), aligned_size, hits_offset);
      }
      return hits;
  }
  
  std::vector<KmerDoubleHit> gatherKmerResultsClamped(
      uint32_t dpu_id, 
      uint32_t results_mram_offset, 
      uint32_t result_capacity_bytes, 
      uint32_t& out_overflow);

  template <typename HitType>
  std::vector<std::vector<HitType>> gatherResultsParallel(uint32_t results_mram_offset, uint32_t result_capacity_bytes) {
      uint32_t num_dpus = dpu_comm_.getNumDPUsActive();
      std::vector<std::vector<HitType>> all_hits(num_dpus);

      std::vector<std::vector<uint8_t>> count_bufs;
      dpu_comm_.gatherDataParallel(count_bufs, 8, results_mram_offset);

      uint32_t max_hits = 0;
      std::vector<uint32_t> hit_counts(num_dpus);

      for (uint32_t i = 0; i < num_dpus; ++i) {
          uint64_t hdr = 0;
          memcpy(&hdr, count_bufs[i].data(), 8);
          uint32_t count = static_cast<uint32_t>(hdr & 0xFFFFFFFFu);
          
          if (result_capacity_bytes > 8) {
              uint32_t max_cap = (result_capacity_bytes - 8) / sizeof(HitType);
              if (count > max_cap) count = max_cap;
          } else {
              count = 0;
          }
          hit_counts[i] = count;
          if (count > max_hits) max_hits = count;
      }

      if (max_hits == 0) return all_hits;

      uint32_t transfer_size = max_hits * sizeof(HitType);
      uint32_t aligned_transfer = DpuCommunicationManager::alignToMram(transfer_size);
      
      std::vector<std::vector<uint8_t>> hit_bufs;
      dpu_comm_.gatherDataParallel(hit_bufs, aligned_transfer, results_mram_offset + 8);

      for (uint32_t i = 0; i < num_dpus; ++i) {
          if (hit_counts[i] > 0) {
              all_hits[i].resize(hit_counts[i]);
              memcpy(all_hits[i].data(), hit_bufs[i].data(), hit_counts[i] * sizeof(HitType));
          }
      }
      return all_hits;
  } 

  // Shared batch container for query packing across gapped/combined modes.
  struct BatchData {
      std::vector<QueryMetadata> meta;
      std::vector<uint8_t> pssm;
      std::vector<uint32_t> keys;
      std::vector<uint32_t> lens;
            std::vector<size_t> qids; // global query indices
      uint32_t max_q_len = 0;
      int16_t min_score = 32767;
      uint32_t common_size = 0;
      size_t next_q_idx = 0;
      bool empty = true;
            std::shared_ptr<std::vector<uint8_t>> common_buffer;
  };

  struct BatchLimits {
      uint32_t max_queries;
      uint32_t max_pssm_bytes;
      uint32_t max_common_bytes;
  };



  BatchData buildQueryBatch(
      size_t start_q_idx,
      DBReader<unsigned int>* qdbr,
      BaseMatrix* subMat,
      const Parameters& par,
      std::vector<float>& compBiasBuffer,
      const BatchLimits& limits,
      EvalueComputation* evaluer,
    int16_t minScoreThr,
    Sequence* seqMapper);

  void processDpuHits(
      const std::vector<std::vector<GappedHit>>& dpu_hits,
      const BatchData& batch,
      DBReader<unsigned int>* tdbr,
      const Parameters& par,
      EvalueComputation* evaluer,
      bool sameDB,
      QueryMatcherTaxonomyHook* taxonomyHook,
      std::vector<std::vector<Matcher::result_t>>& out_results,
      bool is_gpu);


  
  void runDpuKmerBatch(
      Parameters& par,
      BaseMatrix* subMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      bool sameDB,
      DBWriter& resultWriter,
      int kmerThr,
      ScoreMatrix* extMatTwo,
      ScoreMatrix* extMatThree,
      const std::string& spacedPatternStr,
      bool takeOnlyBestKmer);
  
  // K-mer batch helpers
  struct KmerRunContext;
  
  // Double-buffered batch data for async pipeline
  struct KmerBatchData {
      std::vector<KmerQueryPacket> packets;
      std::vector<size_t> query_indices;  // Queries present in this batch
      size_t packet_count;
      bool valid;  // Indicates if this batch has data to process
      
      KmerBatchData() : packet_count(0), valid(false) {}
  };
  
  std::vector<std::vector<uint8_t>> prepareKmerDescriptors(
      const KmerRunContext& ctx,
      const std::vector<DpuIndexBuffer>& wave_indices,
      const std::vector<std::vector<uint32_t>>& splits,
      uint32_t num_packets,
      size_t wave_start,
      size_t wave_size);
  
  std::vector<std::vector<KmerDoubleHit>> executeKmerBatchWithOverflow(
      const KmerRunContext& ctx,
      const std::vector<std::vector<uint8_t>>& descriptors,
      uint32_t& out_overflows,
      DpuGroupManager& group_mgr);
  
  std::vector<std::vector<KmerDoubleHit>> processKmerBatchOnDpu(
      const KmerRunContext& ctx,
      const KmerBatchData& batch,
      const std::vector<DpuIndexBuffer>& wave_indices,
      const std::vector<std::vector<uint32_t>>& splits,
      size_t wave_start,
      size_t wave_size,
      uint32_t& out_overflows,
      DpuGroupManager& group_mgr);
  
  void runDpuUngappedBatch(
      Parameters& par,
      BaseMatrix* subMat,
      int8_t* tinySubMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      bool sameDB,
      DBWriter& resultWriter);
  
  void runDpuGappedBatch(
      Parameters& par,
      BaseMatrix* subMat,
      int8_t* tinySubMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      bool sameDB,
      DBWriter& resultWriter);

  void runDpuUngappedGappedBatch(
      Parameters& par,
      BaseMatrix* subMat,
      int8_t* tinySubMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      bool sameDB,
      DBWriter& resultWriter);

  std::vector<int8_t> extractPSSMFromProfile(
      const char* profileData,
      uint32_t seqlen,
      BaseMatrix* subMat);

  std::vector<int8_t> buildPSSMFromSequence(
      const char* sequence, uint32_t seq_len, BaseMatrix* subMat,
      bool compBiasCorrection, float compBiasCorrectionScale, std::vector<float>& compositionBias);
  
  void assembleTargetBatchByIndices(
      DBReader<unsigned int>* tdbr,
      const std::vector<uint32_t>& target_indices,
      std::vector<uint8_t>& packed_sequences,
      std::vector<TargetMetadata>& metadata,
      BaseMatrix* subMat);
  
  std::vector<std::vector<uint32_t>> buildLoadBalancedDistribution(
      DBReader<unsigned int>* tdbr, uint32_t num_dpus);

  int getMaxTargetsPerDpu(std::vector<std::vector<TargetMetadata>> &perDpuTargetMeta);
};

}  // namespace mmseqs::dpu