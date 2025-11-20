#pragma once

#include <cstdint>
#include <cstring>

namespace mmseqs::dpu {

// Packed structures define the MRAM binary contract with the DPU kernels.
// Sizes and alignment must be identical on host and DPU.

/**
 * Batch descriptor (sent to DPU)
 */
typedef struct {
  uint32_t batch_id;
  uint32_t num_queries;
  uint32_t num_targets;
  uint32_t query_len;  // All queries same length in this batch
  
  // MRAM offsets (must be 8-byte aligned)
  uint32_t queries_metadata_offset;  // QueryMetadata array
  uint32_t pssm_data_offset;         // Flattened PSSM data
  uint32_t targets_metadata_offset;  // TargetMetadata array
  uint32_t targets_data_offset;      // Packed target sequences
  uint32_t results_offset;           // Output hit buffer
  
  // Sizes for bounds checking
  uint32_t pssm_total_size;
  uint32_t kmer_size;
  uint32_t targets_total_size;
  uint32_t results_buffer_size;
  uint32_t reserved; // pad to 8-byte multiple
} __attribute__((packed)) BatchDescriptor;

/**
 * Query metadata (per-query)
 */
typedef struct {
  uint32_t query_id;           // Global query ID from DBReader
  uint32_t query_len;          // Query sequence length
  uint32_t pssm_offset_in_batch;  // Offset within PSSM section
  uint32_t padding;            // Alignment padding
} __attribute__((packed)) QueryMetadata;

/**
 * Target metadata (per-target)
 */
typedef struct {
  uint32_t target_id;          // Global target ID from DBReader
  uint32_t target_len;         // Sequence length
  uint32_t offset_in_data;     // Offset within targets_data
  uint32_t padding;            // Alignment padding
} __attribute__((packed)) TargetMetadata;

/** K-mer table entry */
typedef struct {
  uint32_t kmer;       // The actual k-mer (encoded)
  uint16_t query_id;   // Local Batch ID (0..N)
  uint16_t query_pos;  // Position in query (for diagonal calc)
} __attribute__((packed)) KmerEntry;

/** K-mer batch info */
typedef struct {
  uint32_t num_queries;
  uint32_t hash_table_size; // Power of 2
  uint32_t hash_table_offset; // MRAM offset
  uint32_t kmer_size;       // K (e.g. 6 or 7)
} __attribute__((packed)) KmerBatchInfo;

/** Prefilter hit (packed) */
typedef struct {
  uint32_t target_id;
  uint16_t query_id; // local query id within batch
  int16_t score;
  int16_t diagonal;
  uint32_t pad1;
  uint16_t pad2;
} __attribute__((packed)) Hit;

typedef struct {
  uint32_t target_id;
  int16_t score;       // Gapped Score
  uint16_t q_end;      // Query End Position (0-indexed)
  uint16_t t_end;      // Target End Position (0-indexed)
  uint16_t padding[3]; // Align to 8 bytes
} __attribute__((packed)) GappedHit;

}  // namespace mmseqs::dpu