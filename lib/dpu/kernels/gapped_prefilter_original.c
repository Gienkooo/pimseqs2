#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dpu_common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

typedef enum { DOWN, RIGHT } Direction;

#define KERNEL_AA_SLOTS 21
#define MAX_TARGET_WRAM_LEN 2048
#define QUERY_CHUNK_SIZE 512
#define GAP_OPEN 11
#define GAP_EXTEND 1
#define W 1024
#define MAX_SCORE 32767
#define X_DROP 100

#ifndef PSSM_CACHE_SIZE
//                       AT LEAST W          + safe margin to be aligned
#define PSSM_CACHE_SIZE (W * KERNEL_AA_SLOTS + 16)
#endif

typedef struct {
    int16_t ppv[W];
    int16_t pv[W];
    int16_t fv[W];
    int16_t ev[W];
    int16_t cv[W];
    int16_t uv[W];
    int16_t lv[W];
    int8_t scv[W];
    int16_t cv_subvec[W];
    int8_t pssm_cache[PSSM_CACHE_SIZE];
    uintptr_t cache_mram_start;
    uintptr_t cache_mram_end;
} GappedScoreScratch;

/* Small helpers used in this legacy kernel */
static inline int min(int a, int b) { return (a < b) ? a : b; }
static inline int min3(int a, int b, int c) { int m = (a < b) ? a : b; return (m < c) ? m : c; }

__dma_aligned CombinedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;

static inline int16_t max(int16_t a, int16_t b) { return a > b ? a : b; }

static void compute_ungapped_diagonal_chunked(
    uint8_t *target_seq, uint32_t t_len, uint32_t q_len, uintptr_t pssm_mram_base,
    int16_t *diag_buffer, int16_t *out_score, int32_t *out_best_diag_idx) 
{
    int16_t global_max_score = 0;
    int32_t global_best_diag = q_len - 1;
    __dma_aligned int8_t temp_read_buf[32];
    
    for (uint32_t q_start = 0; q_start < q_len; q_start += QUERY_CHUNK_SIZE) {
        uint32_t chunk_end = q_start + QUERY_CHUNK_SIZE;
        if (chunk_end > q_len) chunk_end = q_len;
        uint32_t chunk_size = chunk_end - q_start;
        uint32_t num_local_diags = chunk_size + t_len;
        
        for (uint32_t i = 0; i < num_local_diags; ++i) diag_buffer[i] = 0;
        
        for (uint32_t q = q_start; q < chunk_end; ++q) {
            uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
            uintptr_t aligned_addr = row_addr & ~7U;
            uint32_t offset = row_addr & 7U;
            mram_read((__mram_ptr void *)aligned_addr, temp_read_buf, 32);
            int8_t *pssm_vals = &temp_read_buf[offset];
            
            for (uint32_t t = 0; t < t_len; ++t) {
                uint8_t aa = target_seq[t];
                if (aa >= KERNEL_AA_SLOTS) aa = 20;
                int16_t score_val = (int16_t)pssm_vals[aa] - (int16_t)g_bd.pssm_bias;
                
                uint32_t local_q = q - q_start;
                int32_t local_diag = (int32_t)t - (int32_t)local_q + (int32_t)(chunk_size - 1);
                
                if (local_diag >= 0 && local_diag < (int32_t)num_local_diags) {
                    int16_t curr = diag_buffer[local_diag] + score_val;
                    if (curr < 0) curr = 0;
                    diag_buffer[local_diag] = curr;
                    
                    if (curr > global_max_score) {
                        global_max_score = curr;
                        global_best_diag = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
                    }
                }
            }
        }
    }
    *out_score = global_max_score;
    *out_best_diag_idx = global_best_diag;
}

static void calc_cache(uintptr_t pssm_mram_base, uint32_t v_a_start, const uint8_t *target_subseq, uint16_t len, int8_t *result, GappedScoreScratch *scratch) {
    uintptr_t needed_mram_start = pssm_mram_base + (v_a_start * KERNEL_AA_SLOTS);
    uintptr_t needed_mram_end = needed_mram_start + (len * KERNEL_AA_SLOTS);

    if (needed_mram_start < scratch->cache_mram_start || needed_mram_end > scratch->cache_mram_end) {
        uintptr_t aligned_start = needed_mram_start & ~7U;
        uint32_t cache_size = sizeof(scratch->pssm_cache) & ~7U;
        uintptr_t current_addr = aligned_start;
        uint32_t loaded = 0;
        while (loaded < cache_size) {
            uint32_t chunk = cache_size - loaded;
            if (chunk > 2048) chunk = 2048;
            mram_read((__mram_ptr void *)current_addr, &scratch->pssm_cache[loaded], chunk);
            current_addr += chunk;
            loaded += chunk;
        }
        scratch->cache_mram_start = aligned_start;
        scratch->cache_mram_end = aligned_start + loaded;
    }

    // ! there is NO multiple cache reads, so cache size MUST BE at least of length len == W.
    for (uint16_t k = 0; k < len; k++) {
        uintptr_t row_addr = pssm_mram_base + ((v_a_start + k) * KERNEL_AA_SLOTS);
        uint32_t offset = row_addr - scratch->cache_mram_start;
        uint8_t aa = target_subseq[k];
        if (aa >= KERNEL_AA_SLOTS) aa = 20;
        result[k] = (int8_t)((int16_t)scratch->pssm_cache[offset + aa] - (int16_t)g_bd.pssm_bias);
    }
}

static void calc(uintptr_t pssm_mram_base, uint32_t v_a_start, const uint8_t *target_subseq, uint16_t len, int8_t *result) {
    __dma_aligned int8_t temp_read_buf[32];
    for (uint16_t k = 0; k < len; k++) {
        uintptr_t row_addr = pssm_mram_base + ((v_a_start + k) * KERNEL_AA_SLOTS);
        uintptr_t aligned_addr = row_addr & ~7U;
        uint32_t offset = row_addr & 7U;
        mram_read((__mram_ptr void *)aligned_addr, temp_read_buf, 32);
        result[k] = (int8_t)((int16_t)temp_read_buf[offset + ((target_subseq[k] >= KERNEL_AA_SLOTS) ? 20 : target_subseq[k])] - (int16_t)g_bd.pssm_bias);
    }
}

void to_cartesian_coords(int16_t vec_idx, uint32_t center[2], uint32_t *out_coords) {
    int16_t center_idx = W >> 1;
    int16_t diff = vec_idx - center_idx;
    out_coords[0] = center[0] - diff;
    out_coords[1] = center[1] + diff;
}

static void compute_gapped_score(uint8_t *target_seq, uint32_t t_len,
                                 uint32_t q_len, uintptr_t pssm_mram_base,
                                 int32_t best_diag_idx_ungapped,
                                 int16_t *out_score, uint16_t *out_q_end,
                                 uint16_t *out_t_end,
                                 GappedScoreScratch *scratch) {
    /* Read dynamic gap penalties and xdrop from host-provided BatchDescriptor */
    const int16_t Gi = g_bd.gap_open_cost;
    const int16_t Ge = g_bd.gap_extend_cost;
    const int16_t xdrop_dyn = g_bd.xdrop_threshold;
    int16_t *ppv = scratch->ppv;
    int16_t *pv = scratch->pv;
    int16_t *fv = scratch->fv;
    int16_t *ev = scratch->ev;
    int16_t *cv = scratch->cv;
    
    for (uint16_t k = 0; k < W; ++k) { ppv[k] = fv[k] = ev[k] = NEG_INF; pv[k] = 0; }
    
    ppv[(W >> 1) + 1] = 0; 
    pv[W >> 1] = -GAP_OPEN - GAP_EXTEND; 
    pv[(W >> 1) + 1] = -GAP_OPEN - GAP_EXTEND;
    int16_t center_max = pv[W >> 1];
    Direction direction = DOWN;
    uint16_t max_score_in_band_idx = 0;
    int16_t max_cv_subvec_elem = NEG_INF;
    __dma_aligned uint32_t max_score_center[2];
    max_score_center[0] = 0; max_score_center[1] = 0;

    uint32_t iteration = 0;
    uint32_t i = 1, j = 0;
    uint32_t A_LEN = q_len;
    uint32_t B_LEN = t_len;

    while ((i < A_LEN && j <= B_LEN) || (i <= A_LEN && j < B_LEN)) {
        if (pv[W >> 1] < center_max - xdrop_dyn) break;
        Direction prev_direction = direction;
        
        if (i <= (W >> 1)) direction = (prev_direction == DOWN) ? RIGHT : DOWN;
        else direction = (pv[W - 1] > pv[0]) ? DOWN : RIGHT;
        
        int16_t *uv = scratch->uv;
        int16_t *lv = scratch->lv;
        if (direction == DOWN) {
            j++;
            memcpy(uv, pv, W * sizeof(int16_t));
            memmove(lv, &pv[1], (W - 1) * sizeof(int16_t)); lv[W - 1] = NEG_INF;
            memmove(ev, &ev[1], (W - 1) * sizeof(int16_t)); ev[W - 1] = NEG_INF;
            if (prev_direction == DOWN) { memmove(ppv, &ppv[1], (W - 1) * sizeof(int16_t)); ppv[W - 1] = NEG_INF; }
        } else {
            i++;
            uv[0] = NEG_INF; memmove(&uv[1], pv, (W - 1) * sizeof(int16_t));
            memcpy(lv, pv, W * sizeof(int16_t));
            memmove(&fv[1], fv, (W - 1) * sizeof(int16_t)); fv[0] = NEG_INF;
            if (prev_direction == RIGHT) { memmove(&ppv[1], ppv, (W - 1) * sizeof(int16_t)); ppv[0] = NEG_INF; }
        }
        
        for (uint16_t k = 0; k < W; ++k) {
            ev[k] = max(sat_sub(ev[k], Ge), sat_sub(lv[k], Gi + Ge));
            fv[k] = max(sat_sub(fv[k], Ge), sat_sub(uv[k], Gi + Ge));
        }
        
        if (j <= (W >> 1)) ev[(W >> 1) - j] = fv[(W >> 1) - j] = NEG_INF;
        if ((W >> 1) + i < W) ev[(W >> 1) + i] = fv[(W >> 1) + i] = NEG_INF;
        
        uint32_t ppv_start = (W >> 1) - min3(j - 1, (W >> 1), A_LEN - i);
        uint32_t ppv_end = (W >> 1) + min3(i, (W >> 1), B_LEN - j + 1);
        ppv_start = max(0, ppv_start);
        ppv_end = min(W, ppv_end);
        uint32_t update_len = ppv_end - ppv_start;
        memcpy(cv, ppv, W * sizeof(int16_t));
        
        if (update_len > 0) {
            uint32_t v_a_start = max(i - min3(i, (W >> 1) - 1, B_LEN - j) - 1, 0);
            uint32_t v_b_start = j - min3(j, (W >> 1) + 1, A_LEN - i + 1);
            int8_t *scv = scratch->scv;
            calc_cache(pssm_mram_base, v_a_start, &target_seq[v_b_start], update_len, scv, scratch);
            int16_t *cv_subvec = scratch->cv_subvec;
            for (uint32_t k = 0; k < update_len; k++) {
                uint32_t idx = ppv_start + k;
                int16_t considered_cv_elem = sat_add(ppv[idx], (int16_t)scv[k]);
                cv_subvec[k] = max3(ev[idx], fv[idx], considered_cv_elem);
                cv_subvec[k] = max(cv_subvec[k], 0);
                if (cv_subvec[k] > max_cv_subvec_elem) {
                    max_cv_subvec_elem = cv_subvec[k];
                    max_score_in_band_idx = k + ppv_start;
                    max_score_center[0] = i; max_score_center[1] = j;
                }
            }
            memcpy(cv, ppv, W * sizeof(int16_t));
            memcpy(&cv[ppv_start], cv_subvec, (ppv_end - ppv_start) * sizeof(int16_t));
        }
        
        if (j <= (W >> 1)) cv[(W >> 1) - j] = -Gi - (i + j) * Ge;
        if ((W >> 1) + i < W) cv[(W >> 1) + i] = -Gi - (i + j) * Ge;
        
        memcpy(ppv, pv, W * sizeof(int16_t));
        memcpy(pv, cv, W * sizeof(int16_t));
        center_max = max(center_max, pv[W >> 1]);
        iteration++;
    }
    
    __dma_aligned uint32_t best_coords[2];
    to_cartesian_coords(max_score_in_band_idx, max_score_center, best_coords);
    *out_score = max_cv_subvec_elem;
    *out_q_end = best_coords[0];
    *out_t_end = best_coords[1];
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void *)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(CombinedBatchDescriptor)));
    }
    barrier_wait(&my_barrier);
    
    uint8_t *task_target_seq = (uint8_t *)mem_alloc(MAX_TARGET_WRAM_LEN);
    uint32_t diag_buf_size = (QUERY_CHUNK_SIZE + MAX_TARGET_WRAM_LEN) * sizeof(int16_t);
    int16_t *task_diag_buf = (int16_t *)mem_alloc(diag_buf_size);
    GappedScoreScratch *scratch = (GappedScoreScratch *)mem_alloc(sizeof(GappedScoreScratch));
    if (!task_target_seq || !task_diag_buf || !scratch) {
        return 0;
    }
    scratch->cache_mram_start = 0;
    scratch->cache_mram_end = 0;

    /* per-tasklet local hit buffer to avoid mutex contention */
    GappedHit *local_buf = (GappedHit*)mem_alloc(sizeof(GappedHit) * 8);
    int local_idx = 0;

    bool force_gapped = (g_bd.header.flags & 1);
    int16_t min_ungapped_score_threshold = g_bd.min_ungapped_score;
    int16_t min_score_threshold = g_bd.min_score;
    uint32_t query_len = g_bd.header.query_len;

    if (tasklet_id == 0) {
        g_hit_count = 0;
        g_hit_write_offset = 8; // reserve first 8 bytes for hit count
    }

    barrier_wait(&my_barrier);

    for (uint32_t i = tasklet_id; i < g_bd.header.num_targets; i += NR_TASKLETS) {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (i * sizeof(TargetMetadata));
        mram_read((__mram_ptr void *)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        __dma_aligned GappedHit h;
        h.target_id = meta.target_id;
        h.score = 0; h.q_end = 0; h.t_end = 0;
        h.padding[0] = 0; h.padding[1] = 0; h.padding[2] = 0;
        
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) {
            continue;
        }
        
        uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = MRAM_ALIGN_SIZE(meta.target_len);
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void *)seq_addr, task_target_seq, aligned_len);
        
        int16_t ungapped_score = 0;
        int32_t best_diag = 0;
        compute_ungapped_diagonal_chunked(task_target_seq, meta.target_len, query_len, 
                                          mram_base + g_bd.header.pssm_data_offset, task_diag_buf, 
                                          &ungapped_score, &best_diag);

        if (ungapped_score >= min_ungapped_score_threshold || force_gapped) {
            // TODO this approach requires some margin for ungapped to catch all 
            // the hits min_ungapped_score_threshold is calculated based on evalue, but ungapped alignment tends to produce lower scores than gapped
            int16_t gapped_score = 0;
            uint16_t gapped_q_end = 0, gapped_t_end = 0;
            compute_gapped_score(task_target_seq, meta.target_len, query_len, 
                                mram_base + g_bd.header.pssm_data_offset, best_diag, 
                                &gapped_score, &gapped_q_end, &gapped_t_end, scratch);
            
            if (gapped_score < min_score_threshold) {
                gapped_score = 0;
            }

            h.score = gapped_score;
            h.q_end = gapped_q_end;
            h.t_end = gapped_t_end;

            /* Buffer locally and flush in batches using atomic reserve to avoid mutex */
            local_buf[local_idx++] = h;
            if (local_idx >= 8) {
                uint32_t bytes_per_hit = MRAM_ALIGN_SIZE(sizeof(GappedHit));
                uint32_t reserve_bytes = bytes_per_hit * local_idx;
                uint32_t offset = __sync_fetch_and_add(&g_hit_write_offset, reserve_bytes);
                for (int hh = 0; hh < local_idx; ++hh) {
                    uintptr_t addr = mram_base + g_bd.header.results_offset + offset + hh * bytes_per_hit;
                    mram_write(&local_buf[hh], (__mram_ptr void *)addr, bytes_per_hit);
                }
                __sync_fetch_and_add(&g_hit_count, local_idx);
                local_idx = 0;
            }
        }
    }
    /* flush any remaining buffered hits from this tasklet */
    if (local_idx > 0) {
        uint32_t bytes_per_hit = MRAM_ALIGN_SIZE(sizeof(GappedHit));
        uint32_t reserve_bytes = bytes_per_hit * local_idx;
        uint32_t offset = __sync_fetch_and_add(&g_hit_write_offset, reserve_bytes);
        for (int hh = 0; hh < local_idx; ++hh) {
            uintptr_t addr = mram_base + g_bd.header.results_offset + offset + hh * bytes_per_hit;
            mram_write(&local_buf[hh], (__mram_ptr void *)addr, bytes_per_hit);
        }
        __sync_fetch_and_add(&g_hit_count, local_idx);
        local_idx = 0;
    }
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = 0;
        mram_write(count_buf, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);
    }

    return 0;
}