/**
 * DPU Shared Types - Structures shared between Host (C++) and DPU (C) kernels
 * All structs are packed and 8-byte aligned for MRAM transfers.
 */

#ifndef DPU_SHARED_TYPES_H
#define DPU_SHARED_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define DPU_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define DPU_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#define DPU_MRAM_ALIGN 8
#define DPU_ALIGN_SIZE(x) (((x) + (DPU_MRAM_ALIGN - 1)) & ~(DPU_MRAM_ALIGN - 1))

/* Batch Descriptor - Main control structure sent to DPU (80 bytes) */
typedef struct {
    uint32_t batch_id;
    uint32_t num_queries;
    uint32_t num_targets;
    uint32_t query_len;

    uint32_t queries_metadata_offset;
    uint32_t pssm_data_offset;
    uint32_t targets_metadata_offset;
    uint32_t targets_data_offset;
    uint32_t results_offset;

    uint32_t pssm_total_size;
    uint32_t kmer_size;
    uint32_t targets_total_size;
    uint32_t results_buffer_size;
    
    uint32_t flags;                     /* Bit 0: Force gapped alignment */
    int16_t min_score;                  /* Karlin-Altschul threshold */
    uint16_t min_score_padding;

    uint8_t spaced_pattern[16];
    uint8_t spaced_pattern_span;
    uint8_t use_spaced_kmers;
    uint16_t spaced_padding;
} __attribute__((packed)) DpuBatchDescriptor;

DPU_STATIC_ASSERT(sizeof(DpuBatchDescriptor) == 80, "DpuBatchDescriptor must be 80 bytes");

/* Query Metadata (16 bytes) */
typedef struct {
    uint32_t query_id;
    uint32_t query_len;
    uint32_t pssm_offset_in_batch;
    uint32_t padding;
} __attribute__((packed)) DpuQueryMetadata;

DPU_STATIC_ASSERT(sizeof(DpuQueryMetadata) == 16, "DpuQueryMetadata must be 16 bytes");

/* Target Metadata (16 bytes) */
typedef struct {
    uint32_t target_id;
    uint32_t target_len;
    uint32_t offset_in_data;
    uint32_t padding;
} __attribute__((packed)) DpuTargetMetadata;

DPU_STATIC_ASSERT(sizeof(DpuTargetMetadata) == 16, "DpuTargetMetadata must be 16 bytes");

/* K-mer Hash Table Entry (8 bytes) */
typedef struct {
    uint32_t kmer;
    uint16_t query_id;
    uint16_t query_pos;
} __attribute__((packed)) DpuKmerEntry;

DPU_STATIC_ASSERT(sizeof(DpuKmerEntry) == 8, "DpuKmerEntry must be 8 bytes");

/* Ungapped/K-mer Prefilter Hit (16 bytes) */
typedef struct {
    uint32_t target_id;
    uint16_t query_id;
    int16_t score;
    int16_t diagonal;
    uint32_t pad1;
    uint16_t pad2;
} __attribute__((packed)) DpuHit;

DPU_STATIC_ASSERT(sizeof(DpuHit) == 16, "DpuHit must be 16 bytes");

/* Gapped Prefilter Hit (16 bytes) */
typedef struct {
    uint32_t target_id;
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
    uint16_t padding[3];
} __attribute__((packed)) DpuGappedHit;

DPU_STATIC_ASSERT(sizeof(DpuGappedHit) == 16, "DpuGappedHit must be 16 bytes");

#define DPU_FLAG_FORCE_GAPPED   (1U << 0)

/* Type aliases for DPU kernels */
typedef DpuBatchDescriptor   BatchDescriptor;
typedef DpuQueryMetadata     QueryMetadata;
typedef DpuTargetMetadata    TargetMetadata;
typedef DpuKmerEntry         KmerEntry;
typedef DpuHit               Hit;
typedef DpuGappedHit         GappedHit;

#ifdef __cplusplus
}
#endif

#endif /* DPU_SHARED_TYPES_H */
