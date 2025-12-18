#ifndef DPU_SHARED_TYPES_H
#define DPU_SHARED_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Alignment & Constants */
#define DPU_MRAM_ALIGN 8
#define DPU_ALIGN_SIZE(x) (((x) + (DPU_MRAM_ALIGN - 1)) & ~(DPU_MRAM_ALIGN - 1))

/* Coverage Modes */
#define DPU_COV_MODE_BIDIRECTIONAL  0
#define DPU_COV_MODE_TARGET         1
#define DPU_COV_MODE_QUERY          2
#define DPU_COV_MODE_LENGTH_QUERY   3
#define DPU_COV_MODE_LENGTH_TARGET  4
#define DPU_COV_MODE_LENGTH_SHORTER 5

// Kernel contract: when adding new DPU kernels, follow these rules:
// - Include `dpu_common.h` for shared helpers/macros.
// - Use a descriptor struct whose first field is `DpuBatchHeader`.
// - Honor `header.num_active_tasklets` (use `is_tasklet_active()` in kernels).
// - Results MRAM layout: write a 32-bit hit count at results_offset (first 4 bytes)
//   and start hits at `results_offset + 8` (8-byte aligned header).

/* --- 1. Common Batch Header --- */
typedef struct {
    uint32_t batch_id;
    uint32_t num_queries;
    uint32_t num_targets;
    uint32_t query_len;
    
    /* Offsets */
    uint32_t queries_metadata_offset;
    uint32_t pssm_data_offset;      
    uint32_t targets_metadata_offset;
    uint32_t targets_data_offset;
    uint32_t results_offset;
    
    /* Sizes & Limits */
    uint32_t pssm_total_size;
    uint32_t targets_total_size;
    uint32_t results_buffer_size;
    
    uint16_t flags;
    uint8_t  num_active_tasklets; /* Dynamic tasklet control */
    uint8_t  pad[5];              /* pad to 8 bytes */
} __attribute__((packed)) DpuBatchHeader;

/* Kmer descriptor */
typedef struct {
    DpuBatchHeader header;
    
    uint32_t kmer_size;
    int16_t  min_score;
    uint8_t  use_spaced_kmers;
    uint8_t  spaced_pattern_span;
    uint8_t  spaced_pattern[16];
    uint8_t  padding[4];
} __attribute__((packed)) KmerBatchDescriptor;

/* Ungapped descriptor */
typedef struct {
    DpuBatchHeader header;
    
    int16_t min_score;
    int16_t gap_open_cost;      // Not used by ungapped, but kept for struct similarity if needed
    int16_t gap_extend_cost;
    int16_t pssm_bias;
} __attribute__((packed)) UngappedBatchDescriptor;

/* Gapped descriptor */
typedef struct {
    DpuBatchHeader header;
    
    int16_t min_score;
    int16_t gap_open_cost;
    int16_t gap_extend_cost;
    int16_t xdrop_threshold;
    int16_t pssm_bias;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;
    
    uint8_t padding[4];
} __attribute__((packed)) GappedBatchDescriptor;

/* Combined (ungapped+gapped) descriptor */
typedef struct {
    DpuBatchHeader header;
    
    int16_t min_ungapped_score;
    int16_t min_score;          // Gapped score threshold
    int16_t gap_open_cost;
    int16_t gap_extend_cost;
    int16_t xdrop_threshold;
    int16_t pssm_bias;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;
    
    uint8_t padding[2];
} __attribute__((packed)) CombinedBatchDescriptor;


/* Data Structures */

typedef struct {
    uint32_t query_id;
    uint32_t query_len;
    uint32_t pssm_offset_in_batch;
    uint32_t padding;
} __attribute__((packed)) QueryMetadata;

typedef struct {
    uint32_t target_id;
    uint32_t target_len;
    uint32_t offset_in_data;
    uint32_t padding;
} __attribute__((packed)) TargetMetadata;

typedef struct {
    uint32_t target_id;
    uint16_t query_id;
    int16_t score;
    int16_t diagonal;
    uint32_t pad1;
    uint16_t pad2;
} __attribute__((packed)) Hit;

typedef struct {
    uint32_t target_id;
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
    uint16_t padding[3];
} __attribute__((packed)) GappedHit;

typedef struct {
    uint32_t kmer;
    uint16_t query_id;
    uint16_t query_pos;
} __attribute__((packed)) KmerEntry;

#ifdef __cplusplus
}
#endif

#endif /* DPU_SHARED_TYPES_H */