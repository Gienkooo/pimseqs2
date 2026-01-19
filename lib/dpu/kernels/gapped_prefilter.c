/* Gapped DPU kernel — Smith-Waterman prefilter (INT32 VERSION). */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "dpu_common.h"

/* Dynamic scratch size based on tasklet count. */
#define SCRATCH_SIZE SCRATCH_PER_TASKLET(MAX_SAFE_TASKLETS)

/* Constants for INT32 SW */
#define NEG_INF_32 -1000000000

__dma_aligned GappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;
__host uint32_t g_overflow;

__host uint8_t g_effective_tasklets;

__dma_aligned QueryMetadata g_query_meta[MAX_BATCH_QUERIES];

static SwResult compute_sw_tiled(
    uintptr_t mram_base,
    uint32_t target_data_offset,
    uint32_t target_len,
    uint32_t query_len,
    uintptr_t pssm_mram_base,
    uintptr_t mram_scratch_vectors,
    uint8_t *wram_scratch,
    uint32_t wram_scratch_size,
    int32_t gap_open,    
    int32_t gap_extend)  
{
    SwResult result = (SwResult){0, 0, 0};
    int32_t max_score = 0; 
    
    // Calculate Safe Stride including alignment padding
    uint32_t vec_stride = ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));
    
    // MRAM vectors base pointers
    uintptr_t mram_H_vec = mram_scratch_vectors;
    uint32_t num_q_tiles = (query_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE;
    uintptr_t mram_E_vec = mram_scratch_vectors + (num_q_tiles * vec_stride);

    // Initialize MRAM vectors (Column 0)
    // We need to clear H to 0 and E to NEG_INF_32. 
    // Since mram_fill_i16 isn't suitable, we use a small WRAM buffer to init MRAM.
    // NOTE: This assumes wram_scratch is available and large enough for one tile vector.
    {
        int32_t *init_buf = (int32_t*)wram_scratch;
        uint32_t count = (Q_TILE_SIZE + 1);
        uint32_t bytes = ALIGN8(count * sizeof(int32_t));
        
        // Sanity check WRAM
        if (bytes > wram_scratch_size) return result;

        // Init H vector to 0
        for(uint32_t i=0; i<count; i++) init_buf[i] = 0;
        for(uint32_t t=0; t<num_q_tiles; t++) {
             mram_write(init_buf, (__mram_ptr void*)(mram_H_vec + t*vec_stride), bytes);
        }

        // Init E vector to NEG_INF_32
        for(uint32_t i=0; i<count; i++) init_buf[i] = NEG_INF_32;
        for(uint32_t t=0; t<num_q_tiles; t++) {
             mram_write(init_buf, (__mram_ptr void*)(mram_E_vec + t*vec_stride), bytes);
        }
    }

    uint8_t *ptr = wram_scratch;

    int32_t *H_top = (int32_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int32_t));
    int32_t *F_top = (int32_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int32_t));
    int32_t *H_bot = (int32_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int32_t));
    int32_t *F_bot = (int32_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int32_t));

    int32_t *H_col = (int32_t *)ptr;
    ptr += ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));
    int32_t *E_col = (int32_t *)ptr;
    ptr += ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));

    uint8_t *target_tile = (uint8_t *)ptr;
    ptr += ALIGN8(T_TILE_SIZE);
    int8_t *pssm_tile = (int8_t *)ptr;
    ptr += ALIGN8(Q_TILE_SIZE * ALPHA_SIZE + 8);

    // SAFETY CHECK: Ensure we didn't overrun WRAM
    if ((uint32_t)(ptr - wram_scratch) > wram_scratch_size)
        return result;

    uint32_t num_t_tiles = (target_len + T_TILE_SIZE - 1) / T_TILE_SIZE;

    for (uint32_t t_tile_idx = 0; t_tile_idx < num_t_tiles; t_tile_idx++)
    {
        uint32_t t_start = t_tile_idx * T_TILE_SIZE;
        uint32_t t_size = min_u32(T_TILE_SIZE, target_len - t_start);

        mram_read((__mram_ptr void *)(mram_base + target_data_offset + t_start),
                  target_tile, ALIGN8(t_size));

        for (uint32_t col = 0; col <= t_size; col++)
        {
            H_top[col] = 0;
            F_top[col] = NEG_INF_32;
        }

        uint32_t q_tile_idx = 0;
        for (uint32_t q_start = 0; q_start < query_len; q_start += Q_TILE_SIZE, q_tile_idx++)
        {
            uint32_t q_size = min_u32(Q_TILE_SIZE, query_len - q_start);

            uint32_t vec_bytes = ALIGN8((q_size + 1) * sizeof(int32_t));
            
            uint32_t mram_offset = q_tile_idx * vec_stride;

            if (t_tile_idx == 0)
            {
                for (uint32_t i = 0; i <= q_size; i++)
                {
                    H_col[i] = 0;
                    E_col[i] = NEG_INF_32;
                }
            }
            else
            {
                mram_read((__mram_ptr void *)(mram_H_vec + mram_offset), H_col, vec_bytes);
                mram_read((__mram_ptr void *)(mram_E_vec + mram_offset), E_col, vec_bytes);
            }

            // Load PSSM (stays int8_t)
            uintptr_t tile_addr = pssm_mram_base + (uintptr_t)q_start * ALPHA_SIZE;
            uint32_t tile_bytes = q_size * ALPHA_SIZE;
            if ((tile_addr & 7U) == 0) {
                mram_read_aligned_bulk(tile_addr, pssm_tile, tile_bytes);
            } else {
                mram_read_unaligned_bulk(tile_addr, pssm_tile, tile_bytes);
            }

            H_bot[0] = H_col[q_size];
            F_bot[0] = NEG_INF_32;

            // Process columns
            for (uint32_t col = 1; col <= t_size; col++)
            {
                uint8_t aa = target_tile[col - 1];
                if (aa >= ALPHA_SIZE) aa = 20;

                int32_t h_up = H_top[col];     
                int32_t f_up = F_top[col];     
                int32_t h_diag = H_top[col - 1]; 
                
                // pssm_ptr unused variable removed or used directly

                for (uint32_t i = 1; i <= q_size; i++)
                {
                    int32_t h_left = H_col[i]; 
                    int32_t e_left = E_col[i]; 

                    // PSSM remains int8, cast to int32 for math
                    int32_t sub = (int32_t)pssm_tile[(i - 1) * ALPHA_SIZE + aa];

                    // Standard 32-bit Math (No saturation needed for scores < 2 billion)
                    
                    // E(i,j)
                    int32_t e_ext = e_left - gap_extend;
                    int32_t e_open = h_left - gap_open;
                    int32_t e_new = (e_ext > e_open) ? e_ext : e_open;

                    // F(i,j)
                    int32_t f_ext = f_up - gap_extend;
                    int32_t f_open = h_up - gap_open;
                    int32_t f_new = (f_ext > f_open) ? f_ext : f_open;

                    // H(i,j)
                    int32_t diag = h_diag + sub;
                    
                    // Max3
                    int32_t h_new = diag;
                    if (e_new > h_new) h_new = e_new;
                    if (f_new > h_new) h_new = f_new;
                    
                    if (h_new < 0) h_new = 0;

                    if (h_new > max_score)
                    {
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

            mram_write(H_col, (__mram_ptr void *)(mram_H_vec + mram_offset), vec_bytes);
            mram_write(E_col, (__mram_ptr void *)(mram_E_vec + mram_offset), vec_bytes);

            int32_t *tmp;
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
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(GappedBatchDescriptor)));
        uint8_t eff = g_bd.header.num_active_tasklets;
        if (eff == 0 || eff > MAX_SAFE_TASKLETS) {
            eff = MAX_SAFE_TASKLETS;
        }
        g_effective_tasklets = eff;
    }
    barrier_wait(&my_barrier);

    bool is_active = (tasklet_id < g_effective_tasklets);

    if (tasklet_id == 0) {
        __dma_aligned uint32_t hdr[2] = {0, 0};
        mram_write(hdr, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);
        g_hit_count = 0;
        g_hit_write_offset = 8;
        g_overflow = 0;

        uint32_t num_q = g_bd.header.num_queries;
        if (num_q > MAX_BATCH_QUERIES) num_q = MAX_BATCH_QUERIES;
        uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
        mram_read((__mram_ptr void*)qmeta_base, g_query_meta, MRAM_ALIGN_SIZE(num_q * sizeof(QueryMetadata)));
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&my_barrier);

    uint8_t *scratch_buffer_raw = NULL;
    uint8_t *scratch_buffer = NULL;

    if (is_active) {
        scratch_buffer_raw  = (uint8_t *)mem_alloc(SCRATCH_SIZE + 8);
        scratch_buffer      = (uint8_t *)ALIGN8_PTR(scratch_buffer_raw);
    }

    uint32_t max_query_len = g_bd.header.query_len;
    // NOTE: per-query min_score will be used from QueryMetadata; do not use batch-level min_score

    uintptr_t pssm_base_start = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;
    
    uint32_t total_buffer_size = g_bd.header.results_buffer_size;

    // Calculate Allocation Size based on SAFE STRIDE for INT32
    // CHANGED: sizeof(int32_t)
    uint32_t vec_stride = ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));
    uint32_t num_q_tiles = (max_query_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE;
    uint32_t single_vec_size = num_q_tiles * vec_stride;
    
    uint32_t total_scratch_bytes = g_effective_tasklets * (2 * single_vec_size);
    
    uint32_t limit_for_hits = total_buffer_size - total_scratch_bytes;

    uint32_t task_offset = tasklet_id * (2 * single_vec_size);
    uintptr_t mram_scratch_vectors = results_base + limit_for_hits + task_offset;

    // Assumes GappedHit struct size is updated in header if necessary
    const uint32_t HIT_STRIDE = MRAM_ALIGN_SIZE(sizeof(GappedHit));
    GappedHit local_hits[8];
    uint32_t local_count = 0;

#define FLUSH_LOCAL_HITS()                                                                 \
    do {                                                                                   \
        if (local_count) {                                                                 \
            uint32_t bytes = local_count * HIT_STRIDE;                                     \
            mutex_lock(hit_mutex);                                                         \
            if (g_hit_write_offset + bytes > limit_for_hits) {                             \
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
    
    if (is_active) {
        for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += g_effective_tasklets)
        {
            __dma_aligned TargetMetadata meta;
            uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
            mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
            
            if (false && !can_be_covered(max_query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) {
                continue;
            }
            
            for (uint32_t q_idx = 0; q_idx < g_bd.header.num_queries; ++q_idx) {
                QueryMetadata qmeta;
                if (q_idx < MAX_BATCH_QUERIES) {
                    qmeta = g_query_meta[q_idx];
                } else {
                    uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
                    mram_read((__mram_ptr void*)(qmeta_base + q_idx * sizeof(QueryMetadata)), &qmeta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
                }

                int32_t current_min_score = (int32_t)qmeta.min_score;
                uintptr_t pssm_addr = pssm_base_start + qmeta.pssm_offset_in_batch;

                SwResult sw = compute_sw_tiled(
                    mram_base,
                    g_bd.header.targets_data_offset + meta.offset_in_data, 
                    meta.target_len,
                    qmeta.query_len,
                    pssm_addr,
                    mram_scratch_vectors,
                    scratch_buffer,
                    SCRATCH_SIZE,
                    (int32_t)g_bd.gap_open_cost,    // Cast to int32
                    (int32_t)g_bd.gap_extend_cost); // Cast to int32
                
                if (false && sw.score < current_min_score) 
                {
                    // printf("Skipping hit: score %d < min_score %d\n", sw.score, current_min_score);
                    continue;
                }
                
                uint16_t aln_len = (sw.q_end > sw.t_end) ? sw.q_end : sw.t_end;
                if (false && g_bd.min_aln_len > 0 && aln_len < g_bd.min_aln_len) 
                {
                    // printf("Skipping hit: aln_len %u < min_aln_len %u\n", aln_len, g_bd.min_aln_len);
                    continue;
                }
                
                if (false && !has_coverage(sw.q_end, sw.t_end, qmeta.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) 
                {
                    // printf("Skipping hit: coverage criteria not met\n");
                    continue;
                }
                
                if (false && !passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, g_bd.seq_id_thr_pct)) 
                {
                    // printf("Skipping hit: seq_id threshold not met\n");
                    continue;
                }
                
                __dma_aligned GappedHit hit;
                hit.target_id = meta.target_id;
                hit.score = sw.score; // Assumes GappedHit.score is int32
                hit.q_end = sw.q_end;
                hit.t_end = sw.t_end;
                hit.padding[0] = (uint16_t)q_idx;
                hit.padding[1] = 0;

                local_hits[local_count++] = hit;
                if (local_count == (sizeof(local_hits) / sizeof(local_hits[0])))
                    FLUSH_LOCAL_HITS();
            }
        }
        
        FLUSH_LOCAL_HITS();
    }

#undef FLUSH_LOCAL_HITS

    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = g_overflow;
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}