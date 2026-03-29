/* Ungapped DPU kernel — diagonal-based prefilter. */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "dpu_common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

/* Kernel-specific constants.
 * WRAM Budget per tasklet with 11 tasklets: ~3.5KB
 * - Target buffer: process in 1KB chunks (streamed from MRAM)
 * - Diag buffer: stored in MRAM, process tiles in WRAM
 * - PSSM cache: 512 bytes (24 residues * 21 scores)
 * This kernel processes targets in tiles to fit WRAM constraints. */
#define TARGET_TILE_SIZE 1024   /* Process target in 1KB chunks */
#define PSSM_CACHE_SIZE 512     /* Safe cache for ~24 query residues */
#define DIAG_TILE_SIZE 512      /* Process diagonals in tiles */

/* Maximum diagonal entries per tasklet.
 * CONSTRAINT: max_query_len + max_target_len <= MAX_DIAG_ENTRIES
 * 
 * With MAX_DIAG_ENTRIES=2048:
 *   - query=500  -> max target = 1548 residues
 *   - query=1000 -> max target = 1048 residues  
 * 
 * This covers 95%+ of proteins (average ~300, 95th percentile ~1000).
 * Sequences exceeding this limit are silently skipped.
 * For very long sequences, use gapped mode which streams via MRAM.
 * 
 * Memory: 2048 * 2 bytes = 4KB per tasklet diagonal buffer.
 * With 5 active tasklets: 20KB diagonal buffers + 5KB tiles + 2.5KB cache = ~27.5KB WRAM. */
#define MAX_DIAG_ENTRIES 2048

/* Globals */
__dma_aligned UngappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;
__host uint32_t g_overflow;

/* Shared tasklet limit - set by tasklet 0, read by all after barrier */
__host uint8_t g_effective_tasklets;

__dma_aligned QueryMetadata g_query_meta[MAX_BATCH_QUERIES];

/* Compute diagonal scoring using signed PSSM values.
 * Uses Smith-Waterman style local alignment scoring: scores cannot go negative.
 * The CPU uses unsigned 8-bit arithmetic which caps scores at 255.
 * We emulate this behavior to ensure validation matches.
 * NOTE: num_diags must fit in the allocated diag_buffer (max MAX_DIAG_ENTRIES).
 */
static void compute_ungapped_diagonal_with_diag(
    uint8_t *target_seq, uint32_t t_len, uint32_t q_len, uintptr_t pssm_mram_base,
    int16_t *diag_buffer, uint8_t *pssm_cache, uint32_t diag_buffer_capacity,
    int16_t *out_score, int16_t *out_diag) 
{
    // CPU Logic Constants for BLOSUM62
    const int32_t MAX_CPU_SCORE = 255;

    int16_t global_max_score = 0;
    int32_t global_best_diag = 0;
    
    // Reset diagonal buffer using 64-bit writes for efficiency
    uint32_t num_diags = t_len + q_len;
    if (num_diags > diag_buffer_capacity) {
        num_diags = diag_buffer_capacity; /* Safety clamp */
    }
    
    uint64_t *diag64 = (uint64_t *)diag_buffer;
    uint32_t num_diags_64 = (num_diags * sizeof(int16_t)) / sizeof(uint64_t);
    for (uint32_t i = 0; i < num_diags_64; ++i)
        diag64[i] = 0;
    /* Handle remainder */
    for (uint32_t i = num_diags_64 * 4; i < num_diags; ++i)
        diag_buffer[i] = 0;
    
    // Iterate PSSM in chunks to fit in WRAM heap cache
    for (uint32_t q_start = 0; q_start < q_len; ) {
        // Calculate chunk size (aligned to ALPHA_SIZE columns)
        uint32_t max_res_in_cache = PSSM_CACHE_SIZE / ALPHA_SIZE;
        uint32_t chunk_len = q_len - q_start;
        if (chunk_len > max_res_in_cache) chunk_len = max_res_in_cache;
        
        uint32_t chunk_end = q_start + chunk_len;
        
        // DMA Read PSSM Chunk (Align read size to 8 bytes)
        // Use aligned bulk read for large chunks - more efficient DMA path
        uint32_t read_size = chunk_len * ALPHA_SIZE;
        if (read_size & 7) read_size = (read_size + 7) & ~7;
        
        uintptr_t src = pssm_mram_base + (q_start * ALPHA_SIZE);
        if ((src & 7U) != 0) {
            // Defensive catch, though statically impossible with host pipeline
            continue;
        }
        mram_read((__mram_ptr void*)src, pssm_cache, read_size);

        for (uint32_t q = q_start; q < chunk_end; ++q) {
            // Pointer to cached column (PSSM is SIGNED int8_t from host)
            int8_t *pssm_col = (int8_t*)&pssm_cache[(q - q_start) * ALPHA_SIZE];

            for (uint32_t t = 0; t < t_len; ++t) {
                uint8_t aa = target_seq[t];
                if (aa >= ALPHA_SIZE) aa = ALPHA_SIZE - 1; // 'X' or padding
                
                int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
                
                if (diag_idx >= 0 && diag_idx < (int32_t)num_diags) {
                    int32_t prev = diag_buffer[diag_idx];
                    int8_t sub = pssm_col[aa];  // SIGNED substitution score
                    int32_t score_with_sub = prev + (int32_t)sub;

                    // Clamp to [0, MAX_CPU_SCORE] like CPU ungapped scoring
                    if (score_with_sub < 0) score_with_sub = 0;
                    if (score_with_sub > MAX_CPU_SCORE) score_with_sub = MAX_CPU_SCORE;
                    
                    int16_t curr = (int16_t)score_with_sub;
                    
                    diag_buffer[diag_idx] = curr;
                    
                    if (curr > global_max_score) {
                        global_max_score = curr;
                        global_best_diag = diag_idx;
                    }
                }
            }
        }
        q_start += chunk_len;
    }
    *out_score = global_max_score;
    *out_diag = (int16_t)global_best_diag; 
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)DPU_MRAM_HEAP_POINTER;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(UngappedBatchDescriptor)));
        g_hit_count = 0;
        g_hit_write_offset = 8; 
        g_overflow = 0;

        // Compute effective tasklets into shared global
        uint8_t eff = g_bd.header.num_active_tasklets;
        if (eff == 0 || eff > MAX_SAFE_TASKLETS) {
            eff = MAX_SAFE_TASKLETS;
        }
        g_effective_tasklets = eff;

        // Cache Query Metadata for faster access
        uint32_t num_q = g_bd.header.num_queries;
        if (num_q > MAX_BATCH_QUERIES) num_q = MAX_BATCH_QUERIES;
        uintptr_t qmeta_base_init = mram_base + g_bd.header.queries_metadata_offset;
        mram_read((__mram_ptr void*)qmeta_base_init, g_query_meta, MRAM_ALIGN_SIZE(num_q * sizeof(QueryMetadata)));
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&my_barrier);

    /* Use shared global - all tasklets can now safely read this */
    bool is_active = (tasklet_id < g_effective_tasklets);
    
    /* Early exit if results buffer is too small - but all tasklets must sync first */
    if (g_bd.header.results_buffer_size < 8) {
        barrier_wait(&my_barrier);
        return 0;
    }
    
    /* WRAM Budget per active tasklet:
     * With 11 tasklets: ~3.5KB each
     * - target_tile: TARGET_TILE_SIZE = 1024 bytes
     * - diag_buffer: For short seqs only, max ~2KB (1024 diags * 2 bytes)
     * - pssm_cache: PSSM_CACHE_SIZE = 512 bytes
     * Total per tasklet: ~3.5KB - fits! */
    
    /* Allocations only for active tasklets */
    uint8_t *task_target_seq = NULL;
    int16_t *diag_buffer = NULL;
    uint8_t *pssm_cache = NULL;
    
    /* Calculate max target length we can handle with limited diagonal buffer.
     * MAX_DIAG_ENTRIES is defined at file scope (1024). */
    const uint32_t max_query_len = g_bd.header.query_len;
    const uint32_t max_target_for_diag = MAX_DIAG_ENTRIES > max_query_len 
                                           ? MAX_DIAG_ENTRIES - max_query_len 
                                           : 256;
    
    if (is_active) {
        /* Allocate target tile buffer (process sequences in chunks) */
        task_target_seq = (uint8_t *)mem_alloc(TARGET_TILE_SIZE);
        
        /* Allocate diagonal buffer - size limited to fit WRAM */
        uint32_t diag_bytes = ALIGN8(MAX_DIAG_ENTRIES * sizeof(int16_t));
        diag_buffer = (int16_t *)mem_alloc(diag_bytes);
        
        pssm_cache = (uint8_t *)mem_alloc(PSSM_CACHE_SIZE);
        
        if (!task_target_seq || !diag_buffer || !pssm_cache) {
            /* Active tasklet failed alloc - mark inactive to skip work loop */
            is_active = false;
        }
    }
    
    int16_t min_score = g_bd.min_score; 
    uintptr_t pssm_base_start = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;

    /* Buffered hit output to reduce mutex contention */
    const uint32_t HIT_STRIDE = MRAM_ALIGN_SIZE(sizeof(Hit));
    Hit local_hits[32];
    uint32_t local_count = 0;

#define FLUSH_LOCAL_HITS()                                                                 \
    do {                                                                                   \
        if (local_count) {                                                                 \
            uint32_t bytes = local_count * HIT_STRIDE;                                     \
            mutex_lock(hit_mutex);                                                         \
            if (g_hit_write_offset + bytes > g_bd.header.results_buffer_size) {            \
                g_overflow = 1;                                                            \
                mutex_unlock(hit_mutex);                                                   \
                local_count = 0;                                                           \
            } else {                                                                       \
                uint32_t offset = g_hit_write_offset;                                      \
                g_hit_write_offset += bytes;                                               \
                g_hit_count += local_count;                                                \
                mutex_unlock(hit_mutex);                                                   \
                mram_write(local_hits, (__mram_ptr void *)(results_base + offset), bytes); \
                local_count = 0;                                                           \
            }                                                                              \
        }                                                                                  \
    } while (0)
    
    /* Only active tasklets execute the work loop */
    if (is_active) {
    
    /* Calculate maximum target length that fits in diagonal buffer */
    const uint32_t max_target_len_for_diags = MAX_DIAG_ENTRIES > max_query_len 
                                               ? MAX_DIAG_ENTRIES - max_query_len 
                                               : 256;
    
    for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += g_effective_tasklets) {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        /* Skip if target is too long for diagonal buffer or empty */
        if (meta.target_len == 0 || meta.target_len > max_target_len_for_diags) continue;
        
        /* Read target in tiles (stream from MRAM) */
        uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
        uint32_t read_size = (meta.target_len <= TARGET_TILE_SIZE) 
                              ? MRAM_ALIGN_SIZE(meta.target_len) 
                              : TARGET_TILE_SIZE;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, read_size);
        
        for (uint32_t q_idx = 0; q_idx < g_bd.header.num_queries; ++q_idx) {
            // Use cached metadata if available, else read from MRAM
            QueryMetadata qmeta;
            if (q_idx < MAX_BATCH_QUERIES) {
                qmeta = g_query_meta[q_idx];
            } else {
                mram_read((__mram_ptr void*)(qmeta_base + q_idx * sizeof(QueryMetadata)), &qmeta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
            }

            if (qmeta.query_len > max_query_len) continue; 

            int16_t score = 0;
            int16_t diagonal = 0;
            uintptr_t pssm_addr = pssm_base_start + qmeta.pssm_offset_in_batch;

            compute_ungapped_diagonal_with_diag(
                task_target_seq, meta.target_len, qmeta.query_len, pssm_addr, 
                diag_buffer, pssm_cache, MAX_DIAG_ENTRIES, &score, &diagonal
            );
            
            if (score >= min_score) {
                Hit hit;
                hit.target_id = meta.target_id;
                hit.query_id = (uint16_t)q_idx; 
                hit.score = score;
                hit.diagonal = diagonal;
                hit.pad1 = 0;
                hit.pad2 = 0;
                
                local_hits[local_count++] = hit;
                if (local_count == 32)
                    FLUSH_LOCAL_HITS();
            }
        }
    }
    
    FLUSH_LOCAL_HITS();
    
    } /* End of if (is_active) block */

#undef FLUSH_LOCAL_HITS

    /* All tasklets must reach this barrier */
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = g_overflow; 
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}