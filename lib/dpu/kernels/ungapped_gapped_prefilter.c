#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "dpu_common.h"

/* Kernel-specific constants */
#undef T_TILE_SIZE
#define T_TILE_SIZE 128   

typedef struct {
    int16_t H_col[Q_TILE_SIZE + 4] __attribute__((aligned(8)));
    int16_t E_col[Q_TILE_SIZE + 4] __attribute__((aligned(8)));

    int16_t H_top[T_TILE_SIZE + 4] __attribute__((aligned(8)));
    int16_t F_top[T_TILE_SIZE + 4] __attribute__((aligned(8)));
    int16_t H_bot[T_TILE_SIZE + 4] __attribute__((aligned(8)));
    int16_t F_bot[T_TILE_SIZE + 4] __attribute__((aligned(8)));

    uint8_t target_tile[T_TILE_SIZE + 8] __attribute__((aligned(8)));
    int8_t pssm_tile[Q_TILE_SIZE * ALPHA_SIZE + 32] __attribute__((aligned(8)));
} TaskletScratch;

#define STRUCT_SIZE sizeof(TaskletScratch)
#define MAX_DIAG_Space (1024 * 2) /* Conservative 2KB for diagonals */
#define PER_TASKLET_BYTES (STRUCT_SIZE + MAX_DIAG_Space)

/* Globals */
__dma_aligned CombinedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;
__host uint32_t g_overflow;
__host uint8_t g_effective_tasklets;

uint8_t *g_scratch_pool = NULL;
uint32_t g_scratch_stride = 0;

__dma_aligned QueryMetadata g_query_meta[MAX_BATCH_QUERIES];

/* -------------------------------------------------------------------------
 * UNGAPPED HEURISTIC
 * ------------------------------------------------------------------------- */
static bool compute_ungapped_diagonal(
    uintptr_t target_mram_addr, uint32_t t_len, uint32_t q_len, uintptr_t pssm_mram_base,
    uint8_t *scratch_buffer, uint32_t scratch_size,
    int16_t *out_score)
{
    /* 1. Setup Buffers */
    /* Diagonal buffer sits at the start of scratch */
    uint32_t num_diags = t_len + q_len;
    uint32_t diag_buf_bytes = ALIGN8(num_diags * sizeof(int16_t));
    
    /* PSSM Cache sits after diagonal buffer */
    #define U_Q_TILE 64
    uint32_t pssm_tile_bytes = ALIGN8(U_Q_TILE * ALPHA_SIZE); 
    
    uint32_t used_so_far = diag_buf_bytes + pssm_tile_bytes;
    if (used_so_far >= scratch_size) return false;

    int16_t *diag_buffer = (int16_t *)scratch_buffer;

    memset(diag_buffer, 0, diag_buf_bytes);

    int8_t *pssm_cache = (int8_t *)(scratch_buffer + diag_buf_bytes);
    uint8_t *t_buf = (uint8_t *)(scratch_buffer + used_so_far);
    uint32_t t_buf_size = scratch_size - used_so_far;
    
    if (t_buf_size < 32) return false;

    /* 2. Outer Loop: Target Chunks */
    for (uint32_t t_start = 0; t_start < t_len; )
    {
        uintptr_t src = target_mram_addr + t_start;
        uintptr_t aligned_src = src & ~7U;
        uint32_t off = (uint32_t)(src & 7U);
        
        uint32_t chunk_len = t_len - t_start;
        uint32_t max_useful = t_buf_size - off;
        if (chunk_len > max_useful) chunk_len = max_useful;
        
        mram_read((__mram_ptr void *)aligned_src, t_buf, ALIGN8(chunk_len + off));
        uint8_t *target_chunk = t_buf + off;
        
        /* 3. Inner Loop: Query Tiles */
        for (uint32_t q_start = 0; q_start < q_len; q_start += U_Q_TILE)
        {
            uint32_t q_chunk_len = min_u32(U_Q_TILE, q_len - q_start);
            uintptr_t pssm_src = pssm_mram_base + (q_start * ALPHA_SIZE);
            mram_read_aligned_bulk(pssm_src, pssm_cache, q_chunk_len * ALPHA_SIZE);

            int8_t *pssm_col_ptr = pssm_cache; 

            for (uint32_t j = 0; j < q_chunk_len; ++j)
            {
                uint32_t q_real = q_start + j;
                
                // Base diagonal index for this query position
                int32_t base_diag_idx = (int32_t)(q_len - 1) - (int32_t)q_real;

                int16_t *diag_ptr = &diag_buffer[base_diag_idx + t_start];

                #pragma unroll 4
                for (uint32_t i = 0; i < chunk_len; ++i)
                {
                    uint8_t aa = target_chunk[i];
                    aa = (aa >= ALPHA_SIZE) ? 20 : aa;

                    int8_t score = pssm_col_ptr[aa]; 
                    
                    // Direct pointer access (load -> add -> store -> inc)
                    int16_t curr = *diag_ptr + score;
                    
                    curr = (curr < 0) ? 0 : curr;
                    
                    *diag_ptr = curr;
                    diag_ptr++;
                }
                pssm_col_ptr += ALPHA_SIZE; 
            }
        }
        t_start += chunk_len;
    }

    /* 4. Find Max Score (Manual Unroll) */
    int16_t global_max_score = 0;
    uint32_t i = 0;
    for (; i + 3 < num_diags; i += 4) {
        int16_t m0 = diag_buffer[i];
        int16_t m1 = diag_buffer[i+1];
        int16_t m2 = diag_buffer[i+2];
        int16_t m3 = diag_buffer[i+3];
        int16_t mx1 = (m0 > m1) ? m0 : m1;
        int16_t mx2 = (m2 > m3) ? m2 : m3;
        int16_t mx  = (mx1 > mx2) ? mx1 : mx2;
        if (mx > global_max_score) global_max_score = mx;
    }
    for (; i < num_diags; ++i) {
        if (diag_buffer[i] > global_max_score) global_max_score = diag_buffer[i];
    }
    
    *out_score = global_max_score;
    return true;
}

/* -------------------------------------------------------------------------
 * GAPPED ALIGNMENT (TILED SW)
 * ------------------------------------------------------------------------- */

static SwResult compute_sw_tiled(
    uintptr_t mram_base,
    uint32_t target_data_offset,
    uint32_t target_len,
    uint32_t query_len,
    uintptr_t pssm_mram_base,
    uintptr_t mram_scratch_vectors,
    TaskletScratch *scratch, 
    int16_t gap_open,
    int16_t gap_extend)
{
    SwResult result = (SwResult){0, 0, 0};
    int16_t max_score = 0;
    
    // Unpack struct pointers (Compiler uses immediate offsets now)
    int16_t *H_col = scratch->H_col;
    int16_t *E_col = scratch->E_col;
    int16_t *H_top = scratch->H_top;
    int16_t *F_top = scratch->F_top;
    int16_t *H_bot = scratch->H_bot;
    int16_t *F_bot = scratch->F_bot;
    uint8_t *target_tile = scratch->target_tile;
    int8_t *pssm_tile = scratch->pssm_tile;

    uint32_t col_vec_len_bytes = ALIGN8((query_len + 1) * sizeof(int16_t));
    uintptr_t mram_H_vec = mram_scratch_vectors;
    uintptr_t mram_E_vec = mram_scratch_vectors + col_vec_len_bytes;

    // Init MRAM vectors
    mram_fill_i16(mram_H_vec, query_len + 1, 0);
    mram_fill_i16(mram_E_vec, query_len + 1, NEG_INF);

    uint32_t num_t_tiles = (target_len + T_TILE_SIZE - 1) / T_TILE_SIZE;

    for (uint32_t t_tile_idx = 0; t_tile_idx < num_t_tiles; t_tile_idx++)
    {
        uint32_t t_start = t_tile_idx * T_TILE_SIZE;
        uint32_t t_size = min_u32(T_TILE_SIZE, target_len - t_start);

        mram_read((__mram_ptr void *)(mram_base + target_data_offset + t_start),
                  target_tile, ALIGN8(t_size));

        // Init Row 0
        #pragma unroll 4
        for (uint32_t col = 0; col <= t_size; col++) {
            H_top[col] = 0;
            F_top[col] = NEG_INF;
        }

        for (uint32_t q_start = 0; q_start < query_len; q_start += Q_TILE_SIZE)
        {
            uint32_t q_size = min_u32(Q_TILE_SIZE, query_len - q_start);
            uint32_t vec_bytes = ALIGN8((q_size + 1) * sizeof(int16_t));
            
            if (t_tile_idx == 0) {
                #pragma unroll 4
                for (uint32_t i = 0; i <= q_size; i++) {
                    H_col[i] = 0;
                    E_col[i] = NEG_INF;
                }
            } else {
                mram_read((__mram_ptr void *)(mram_H_vec + q_start * sizeof(int16_t)), H_col, vec_bytes);
                mram_read((__mram_ptr void *)(mram_E_vec + q_start * sizeof(int16_t)), E_col, vec_bytes);
            }

            // Load PSSM
            uintptr_t tile_addr = pssm_mram_base + (uintptr_t)q_start * ALPHA_SIZE;
            uint32_t tile_bytes = q_size * ALPHA_SIZE;
            if ((tile_addr & 7U) != 0) {
                // Defensive catch, though statically impossible with host pipeline
                continue; 
            }
            mram_read_aligned_bulk(tile_addr, pssm_tile, tile_bytes);

            H_bot[0] = H_col[q_size];
            F_bot[0] = NEG_INF;

            // --- Hot Loop ---
            for (uint32_t col = 1; col <= t_size; col++)
            {
                uint8_t aa = target_tile[col - 1];
                aa = (aa >= ALPHA_SIZE) ? 20 : aa; 

                int16_t h_up = H_top[col];
                int16_t f_up = F_top[col];
                int16_t h_diag = H_top[col - 1];

                int8_t *pssm_ptr = &pssm_tile[aa];

                #pragma unroll 4
                for (uint32_t i = 1; i <= q_size; i++)
                {
                    int16_t h_left = H_col[i];
                    int16_t e_left = E_col[i];

                    int16_t sub = (int16_t)(*pssm_ptr);
                    pssm_ptr += ALPHA_SIZE;

                    int16_t e_ext = sat_sub(e_left, gap_extend);
                    int16_t e_open = sat_sub(h_left, gap_open);
                    int16_t e_new = max2(e_ext, e_open);

                    int16_t f_ext = sat_sub(f_up, gap_extend);
                    int16_t f_open = sat_sub(h_up, gap_open);
                    int16_t f_new = max2(f_ext, f_open);

                    int16_t diag = sat_add(h_diag, sub);
                    int16_t h_new = max3(diag, e_new, f_new);
                    if (h_new < 0) h_new = 0;

                    if (h_new >= max_score) {
                        max_score = h_new;
                        result.score = h_new;
                        result.q_end = (uint16_t)(q_start + i);
                        result.t_end = (uint16_t)(t_start + col);
                    }

                    h_diag = h_left;
                    h_up = h_new;
                    f_up = f_new;

                    H_col[i] = h_new;
                    E_col[i] = e_new;
                }
                H_bot[col] = h_up;
                F_bot[col] = f_up;
            }

            mram_write(H_col, (__mram_ptr void *)(mram_H_vec + q_start * sizeof(int16_t)), vec_bytes);
            mram_write(E_col, (__mram_ptr void *)(mram_E_vec + q_start * sizeof(int16_t)), vec_bytes);

            // Swap pointers
            int16_t *tmp;
            tmp = H_top; H_top = H_bot; H_bot = tmp;
            tmp = F_top; F_top = F_bot; F_bot = tmp;
        }
    }
    result.score = max_score;
    return result;
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)DPU_MRAM_HEAP_POINTER;

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(CombinedBatchDescriptor)));
        uint8_t req = g_bd.header.num_active_tasklets;
        if (req == 0 || req > MAX_SAFE_TASKLETS) req = MAX_SAFE_TASKLETS;

        __dma_aligned uint32_t hdr[2] = {0, 0};
        mram_write(hdr, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);

        g_hit_count = 0;
        g_hit_write_offset = 8;
        g_overflow = 0;

        uint32_t num_q = g_bd.header.num_queries;
        if (num_q > MAX_BATCH_QUERIES) num_q = MAX_BATCH_QUERIES;
        uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
        mram_read((__mram_ptr void*)qmeta_base, g_query_meta, MRAM_ALIGN_SIZE(num_q * sizeof(QueryMetadata)));

        mem_reset();
        /* Allocation: Use the size of the optimized struct */
        uint32_t per_tasklet = ALIGN8(sizeof(TaskletScratch) + 2048); /* +2KB buffer for ungapped diagonals */
        uint8_t *pool_raw = (uint8_t *)mem_alloc(req * per_tasklet);
        
        if (!pool_raw) {
            g_effective_tasklets = 0;
        } else {
            g_scratch_pool = pool_raw;
            g_scratch_stride = per_tasklet;
            g_effective_tasklets = req;
        }
    }
    barrier_wait(&my_barrier);

    bool is_active = (tasklet_id < g_effective_tasklets);
    if (g_effective_tasklets == 0) return 0;

    /* Tasklet-local pointers based on Struct */
    uint8_t *my_scratch_raw = NULL;
    TaskletScratch *my_scratch_struct = NULL;
    
    if (is_active) {
        my_scratch_raw = g_scratch_pool + (tasklet_id * g_scratch_stride);
        my_scratch_struct = (TaskletScratch*)(my_scratch_raw + 2048); // Place struct after diag buffer space
    }

    uint32_t max_query_len = g_bd.header.query_len;
    int16_t min_ungapped_score = g_bd.min_ungapped_score;
    bool force_gapped = (g_bd.header.flags & 1);
    uintptr_t pssm_base_start = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;

    uint32_t vec_stride = ALIGN8((Q_TILE_SIZE + 1) * sizeof(int16_t));
    uint32_t num_q_tiles = (max_query_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE;
    uint32_t single_vec_size = num_q_tiles * vec_stride;
    uint32_t total_scratch_bytes = g_effective_tasklets * (2 * single_vec_size);
    uint32_t limit_for_hits = g_bd.header.results_buffer_size - total_scratch_bytes;
    uintptr_t mram_scratch_vectors = results_base + limit_for_hits + (tasklet_id * (2 * single_vec_size));

    GappedHit local_hits[32];
    uint32_t local_count = 0;

#define FLUSH_LOCAL_HITS() do { \
    if (local_count) { \
        uint32_t bytes = local_count * sizeof(GappedHit); \
        mutex_lock(hit_mutex); \
        if (g_hit_write_offset + bytes > limit_for_hits) { \
            g_overflow = 1; \
            mutex_unlock(hit_mutex); \
            local_count = 0; \
        } else { \
            uint32_t offset = g_hit_write_offset; \
            g_hit_write_offset += bytes; \
            g_hit_count += local_count; \
            mutex_unlock(hit_mutex); \
            mram_write(local_hits, (__mram_ptr void *)(results_base + offset), bytes); \
            local_count = 0; \
        } \
    } \
} while (0)

    if (is_active) {
        for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += g_effective_tasklets)
        {
            __dma_aligned TargetMetadata meta;
            uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
            mram_read((__mram_ptr void *)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

            if (meta.target_len == 0) continue;

            for (uint32_t q_idx = 0; q_idx < g_bd.header.num_queries; ++q_idx) {
                QueryMetadata qmeta = (q_idx < MAX_BATCH_QUERIES) ? g_query_meta[q_idx] : g_query_meta[0]; // Simple fallback for meta, should allow full read if needed
                if (q_idx >= MAX_BATCH_QUERIES) {
                     uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
                     mram_read((__mram_ptr void*)(qmeta_base + q_idx * sizeof(QueryMetadata)), &qmeta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
                }

                if (!can_be_covered(qmeta.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) continue;

                int16_t ungapped_score = 0;
                uintptr_t pssm_addr = pssm_base_start + qmeta.pssm_offset_in_batch;

                /* Reuse raw scratch buffer for ungapped diagonal matrix */
                bool ran_prefilter = compute_ungapped_diagonal(
                    mram_base + g_bd.header.targets_data_offset + meta.offset_in_data,
                    meta.target_len, qmeta.query_len, pssm_addr,
                    my_scratch_raw, 2048, &ungapped_score);

                if (ran_prefilter && !force_gapped && ungapped_score < min_ungapped_score) continue;

                SwResult sw = compute_sw_tiled(
                    mram_base,
                    g_bd.header.targets_data_offset + meta.offset_in_data, 
                    meta.target_len,
                    qmeta.query_len,
                    pssm_addr,
                    mram_scratch_vectors,
                    my_scratch_struct,
                    g_bd.gap_open_cost,
                    g_bd.gap_extend_cost);

                if (sw.score < qmeta.min_score) continue;

                uint16_t aln_len = max_u32((uint32_t)sw.q_end, (uint32_t)sw.t_end);
                if (g_bd.min_aln_len > 0 && aln_len < g_bd.min_aln_len) continue;
                if (!has_coverage(sw.q_end, sw.t_end, qmeta.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) continue;
                if (!passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, g_bd.seq_id_thr_pct)) continue;

                __dma_aligned GappedHit hit;
                hit.target_id = meta.target_id;
                hit.score = sw.score;
                hit.q_end = sw.q_end;
                hit.t_end = sw.t_end;
                hit.padding[0] = (uint16_t)q_idx;
                hit.padding[1] = 0;

                local_hits[local_count++] = hit;
                if (local_count == 32) FLUSH_LOCAL_HITS();
            }
        }
        FLUSH_LOCAL_HITS();
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        uint32_t hi = (uint32_t)(g_effective_tasklets & 0xFFFFu);
        if (g_overflow) hi |= (1u << 31);
        count_buf[1] = hi;
        mram_write(count_buf, (__mram_ptr void *)(results_base), 8);
    }
    return 0;
}