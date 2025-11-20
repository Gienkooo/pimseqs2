#pragma once

#include "DpuStructures.h"
#include "DpuCommunicationManager.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "BaseMatrix.h"
#include "Parameters.h"
#include "SubstitutionMatrix.h"

#include <vector>

class SequenceLookup;
class EvalueComputation;
class QueryMatcherTaxonomyHook;

namespace mmseqs::dpu {

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
      int alignmentMode);

 private:
  DpuCommunicationManager dpu_comm_;
  
  void runDpuKmerBatch(
      Parameters& par,
      BaseMatrix* subMat,
      DBReader<unsigned int>* qdbr,
      DBReader<unsigned int>* tdbr,
      EvalueComputation* evaluer,
      QueryMatcherTaxonomyHook* taxonomyHook,
      bool sameDB,
      DBWriter& resultWriter);
  
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

  std::vector<int8_t> buildPSSMFromSequence(
      const char* sequence, uint32_t seq_len, BaseMatrix* subMat,
      bool compBiasCorrection, float compBiasCorrectionScale, std::vector<float>& compositionBias);
  
  void assembleTargetBatch(
      DBReader<unsigned int>* tdbr, uint32_t start, uint32_t count,
      std::vector<uint8_t>& packed_sequences,
      std::vector<TargetMetadata>& metadata,
      BaseMatrix* subMat);
  
  std::vector<Hit> collectResults(uint32_t dpu_id, uint32_t num_hits);
};

}  // namespace mmseqs::dpu