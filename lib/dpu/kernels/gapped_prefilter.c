/* Gapped DPU kernel — Smith-Waterman prefilter (no FP). */

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

#define ALPHA_SIZE 21
#define MAX_TARGET_LEN 5000

#define GAP_OPEN 11
#define GAP_EXTEND 1
/* X-drop (Z-drop) threshold to match CPU reference (MMseqs2) */
#define X_DROP 40

__dma_aligned GappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;

typedef struct {
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
} SwResult;

// Smith-Waterman with end position tracking.
static SwResult compute_sw_score_with_endpos(
    const uint8_t* target_seq,
    uint32_t target_len,
    uint32_t query_len,
    uintptr_t pssm_mram_base,
    int16_t* H_prev,
    int16_t* H_curr,
    int16_t* E,
    int16_t* F_prev,
    int16_t* F_curr
) {
    SwResult result = {0, 0, 0};
    int16_t max_score = 0;
    uint16_t max_q = 0;
    uint16_t max_t = 0;
    /* row_max: maximum score observed in the current row (i) - used for X-drop */
    int16_t row_max = 0;
    
    __dma_aligned int8_t pssm_row[32];
    /* Read dynamic params from batch descriptor (host-provided) */
    int16_t gap_open = g_bd.gap_open_cost;
    int16_t gap_extend = g_bd.gap_extend_cost;
    int16_t xdrop_thr = g_bd.xdrop_threshold;
    int16_t pssm_bias = g_bd.pssm_bias;
    (void)pssm_bias; (void)gap_extend; (void)gap_open; (void)xdrop_thr;
    
    for (uint32_t j = 0; j <= target_len; j++) {
        H_prev[j] = 0;
        E[j] = NEG_INF;
        F_prev[j] = NEG_INF;
    }
    
    /* Use incremental offset calculation instead of multiplication in hot loop.
     * DPU has no 32-bit hardware multiplier - this saves ~15 cycles per iteration.
     */
    uintptr_t row_addr = pssm_mram_base;  /* Start at row 0, but we use rows 0..query_len-1 */
    
    for (uint32_t i = 1; i <= query_len; i++) {
        /* row_addr already points to row (i-1) from previous iteration or init */
        uintptr_t aligned_addr = row_addr & ~7UL;
        uint32_t offset = row_addr & 7U;
        mram_read((__mram_ptr void*)aligned_addr, pssm_row, 32);
        int8_t* pssm_vals = &pssm_row[offset];
        
        /* Advance to next row for next iteration: mul21 uses shifts only */
        row_addr += ALPHA_SIZE;  /* Simple addition, no multiplication */
        
        H_curr[0] = 0;
        E[0] = NEG_INF;
        F_curr[0] = NEG_INF;
        row_max = 0;
        
        for (uint32_t j = 1; j <= target_len; j++) {
            uint8_t aa = target_seq[j - 1];
            if (aa >= ALPHA_SIZE) aa = 20;
            int8_t sub_score = pssm_vals[aa];
            
            // E: gap in query (horizontal gap)
            int16_t E_extend = (E[j-1] > NEG_INF + gap_extend) ? (E[j-1] - gap_extend) : NEG_INF;
            int16_t E_open = (H_curr[j-1] > NEG_INF + gap_open) ? 
                            (H_curr[j-1] - gap_open) : NEG_INF;  // FIXED: remove + gap_extend
            E[j] = max2(E_extend, E_open);
            
            // F: gap in target (vertical gap)
            int16_t F_extend = (F_prev[j] > NEG_INF + gap_extend) ? (F_prev[j] - gap_extend) : NEG_INF;
            int16_t F_open = (H_prev[j] > NEG_INF + gap_open) ? 
                            (H_prev[j] - gap_open) : NEG_INF;  // FIXED: remove + gap_extend
            F_curr[j] = max2(F_extend, F_open);
            
            // Score: diagonal match/mismatch, or continue gap (E or F)
            int16_t diag = H_prev[j-1] + sub_score;
            int16_t score = max3(diag, E[j], F_curr[j]);
            H_curr[j] = (score > 0) ? score : 0;
            
            if (H_curr[j] > max_score) {
                max_score = H_curr[j];
                max_q = (uint16_t)i;
                max_t = (uint16_t)j;
            }

            /* track per-row maximum for X-drop termination */
            if (H_curr[j] > row_max) row_max = H_curr[j];
        }
        
        int16_t* temp = H_prev;
        H_prev = H_curr;
        H_curr = temp;
        
        temp = F_prev;
        F_prev = F_curr;
        F_curr = temp;

        /* X-drop (terminate early if current row is far below global max) */
            if (max_score - row_max > g_bd.xdrop_threshold) {
            break;
        }
    }
    
    result.score = max_score;
    result.q_end = max_q;
    result.t_end = max_t;
        return result;
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(GappedBatchDescriptor)));
        g_hit_count = 0;
        g_hit_write_offset = 8; // Start writing hits after 8-byte count
    }
    barrier_wait(&my_barrier);

    /* DYNAMIC TASKLET CHECK: exit immediately if this tasklet is not active */
    if (!is_tasklet_active(g_bd.header.num_active_tasklets)) return 0;

    uint32_t row_size = (MAX_TARGET_LEN + 1) * sizeof(int16_t);
    int16_t* H_prev = (int16_t*)mem_alloc(row_size);
    int16_t* H_curr = (int16_t*)mem_alloc(row_size);
    int16_t* E_row = (int16_t*)mem_alloc(row_size);
    int16_t* F_prev = (int16_t*)mem_alloc(row_size);
    int16_t* F_curr = (int16_t*)mem_alloc(row_size);
    uint8_t* target_seq = (uint8_t*)mem_alloc(MAX_TARGET_LEN);
    
    if (!H_prev || !H_curr || !E_row || !F_prev || !F_curr || !target_seq) {
        return 0;
    }
    
    /* Extract filter parameters */
    int16_t min_score = g_bd.min_score;
    uint32_t query_len = g_bd.header.query_len;
    uint8_t cov_mode = g_bd.cov_mode;
    uint8_t cov_thr_pct = g_bd.cov_thr_pct;
    uint8_t min_aln_len = g_bd.min_aln_len;
    uint8_t seq_id_thr_pct = g_bd.seq_id_thr_pct;
    
    uintptr_t pssm_base = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;
    
    const uint32_t HIT_SIZE = 16;
    /* Per-tasklet local hit buffer to avoid mutex contention (stack-local to avoid extra allocations) */
    GappedHit local_hits[8];
    int local_count = 0;
    
    for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += NR_TASKLETS) {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_LEN) {
            continue;
        }
        
        /* FILTER 1: canBeCovered - skip pairs that can never achieve coverage */
        if (!can_be_covered(query_len, meta.target_len, cov_mode, cov_thr_pct)) {
            continue;
            printf("Skipped target %u by canBeCovered\n", meta.target_id);
        }
        
        /* Load target sequence and compute SW */
        uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = MRAM_ALIGN_SIZE(meta.target_len);
        mram_read((__mram_ptr void*)seq_addr, target_seq, aligned_len);
        
        SwResult sw = compute_sw_score_with_endpos(
            target_seq, meta.target_len, query_len, pssm_base,
            H_prev, H_curr, E_row, F_prev, F_curr
        );
        
        /* FILTER 2: min_score threshold */
        if (sw.score < min_score) {
            continue;
            printf("Skipped target %u by min_score\n", meta.target_id);
        }
        
        /* FILTER 3: minimum alignment length */
        uint16_t aln_len = (sw.q_end > sw.t_end) ? sw.q_end : sw.t_end;
        if (min_aln_len > 0 && aln_len < min_aln_len) {
            continue;
            printf("Skipped target %u by min_aln_len\n", meta.target_id);
        }
        
        /* FILTER 4: hasCoverage - check actual coverage from alignment */
        if (!has_coverage(sw.q_end, sw.t_end, query_len, meta.target_len, cov_mode, cov_thr_pct)) {
            continue;
            printf("Skipped target %u by hasCoverage\n", meta.target_id);
        }
        
        /* FILTER 5: estimated sequence identity (NO DIVISION!) */
        if (!passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, seq_id_thr_pct)) {
            continue;
            printf("Skipped target %u by seq_id threshold\n", meta.target_id);
        }
        
        /* All filters passed - write hit */
        // printf("Accepted target %u: score=%d q_end=%u t_end=%u\n",
        //        meta.target_id,  sw.score, sw.q_end, sw.t_end);
        __dma_aligned GappedHit hit;
        hit.target_id = meta.target_id;
        hit.score = sw.score;
        hit.q_end = sw.q_end;
        hit.t_end = sw.t_end;
        hit.padding[0] = 0;
        hit.padding[1] = 0;
        hit.padding[2] = 0;
        
        /* Buffer locally, flush when full to MRAM using atomic reserve */
        local_hits[local_count++] = hit;
        if (local_count >= 8) {
            uint32_t bytes_per_hit = MRAM_ALIGN_SIZE(sizeof(GappedHit));
            uint32_t reserve_bytes = bytes_per_hit * local_count;

            /* Reserve space and update counters under mutex, then write outside lock */
            mutex_lock(hit_mutex);
            uint32_t offset = g_hit_write_offset;
            g_hit_write_offset += reserve_bytes;
            g_hit_count += local_count;
            mutex_unlock(hit_mutex);

            /* write buffered hits to MRAM sequentially */
            for (int hh = 0; hh < local_count; ++hh) {
                uintptr_t result_addr = results_base + offset + hh * bytes_per_hit;
                mram_write(&local_hits[hh], (__mram_ptr void*)result_addr, bytes_per_hit);
            }
            local_count = 0;
        }
    }
    
    /* Flush any remaining hits from this tasklet before barrier */
    if (local_count > 0) {
        uint32_t bytes_per_hit = MRAM_ALIGN_SIZE(sizeof(GappedHit));
        uint32_t reserve_bytes = bytes_per_hit * local_count;

        mutex_lock(hit_mutex);
        uint32_t offset = g_hit_write_offset;
        g_hit_write_offset += reserve_bytes;
        g_hit_count += local_count;
        mutex_unlock(hit_mutex);

        for (int hh = 0; hh < local_count; ++hh) {
            uintptr_t result_addr = results_base + offset + hh * bytes_per_hit;
            mram_write(&local_hits[hh], (__mram_ptr void*)result_addr, bytes_per_hit);
        }
        local_count = 0;
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = 0;
        /* Write count at offset 0 of results buffer */
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}
