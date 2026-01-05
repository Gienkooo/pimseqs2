#pragma once

#include "shared/DpuSharedTypes.h"
#include "DpuCommunicationManager.h"
#include "DpuWorkflow.h"
#include "DpuKernelManager.h"

#include "DBReader.h"
#include "DBWriter.h"
#include "BaseMatrix.h"
#include "Parameters.h"
#include "SubstitutionMatrix.h"
#include "Matcher.h"
#include "QueryMatcher.h"

#include <vector>

class SequenceLookup;
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
  DpuWorkflow workflow_;
  DpuKernelManager kernel_mgr_; 

  struct TargetChunk {
      bool valid = false;
      size_t count = 0;
      std::vector<uint32_t> indices;
      std::vector<uint8_t> data;
      std::vector<TargetMetadata> meta;
      DpuWorkflow::MramLayout layout{};
  };

  TargetChunk buildTargetChunk(
      uint32_t dpu_id, 
      size_t cursor, 
      const std::vector<std::vector<uint32_t>>& perDpuTargetIndices, 
      DBReader<unsigned int>* tdbr,
      BaseMatrix* subMat,
      uint32_t query_count, 
      uint32_t common_size, 
      uint32_t scratch_bytes,
      size_t descriptor_size,
      size_t result_size);

  bool canFitAtLeastOneTarget(
      uint32_t num_dpus,
      const std::vector<uint32_t>& perDpuLens,
      uint32_t query_count,
      uint32_t common_size,
      uint32_t scratch_bytes,
      size_t descriptor_size,
      size_t result_size);

  void processGappedHits(
      const std::vector<GappedHit>& hits, 
      const TargetChunk& chunk,
      uint32_t queryLen,
      unsigned int queryKey,
      bool sameDB,
      Parameters& par,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      std::vector<Matcher::result_t>& resultsForQuery);

  void processUngappedHits(
      const std::vector<Hit>& hits,
      const TargetChunk& chunk,
      const std::vector<unsigned int>& batchQueryKeys,
      DBReader<unsigned int>* tdbr,
      std::vector<std::vector<hit_t>>& resultsByQuery);

  void processCombinedHits(
      const std::vector<GappedHit>& hits,
      const TargetChunk& chunk,
      const std::vector<unsigned int>& batchQueryKeys,
      const std::vector<uint32_t>& batchQueryLens,
      bool sameDB,
      Parameters& par,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      std::vector<std::vector<Matcher::result_t>>& resultsByQuery);
  
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
  
  std::vector<std::vector<uint8_t>> prepareKmerDescriptors(
      const KmerRunContext& ctx,
      const std::vector<DpuIndexBuffer>& wave_indices,
      const std::vector<std::vector<uint32_t>>& splits,
      uint32_t num_packets,
      size_t wave_start,
      size_t wave_size);
  
  std::vector<std::vector<KmerDoubleHit>> executeKmerBatchWithOverflow(
      const KmerRunContext& ctx,
      const std::vector<std::vector<uint8_t>>& descriptors);
  
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

};

}  // namespace mmseqs::dpu