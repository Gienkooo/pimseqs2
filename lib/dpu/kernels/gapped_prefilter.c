/**
 * Gapped Prefilter Kernel for UPMEM DPU
 * 
 * Simple row-by-row Smith-Waterman algorithm with affine gap penalties.
 * Uses 2-row technique for H and F matrices.
 */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "DpuSharedTypes.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#define ALPHA_SIZE 21
/* Increased to handle very long targets - uses ~55KB per tasklet */
#define MAX_TARGET_LEN 5000

#define GAP_OPEN 11
#define GAP_EXTEND 1
#define NEG_INF (-30000)

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);

#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

static inline int16_t max2(int16_t a, int16_t b) {
    return (a > b) ? a : b;
}

static inline int16_t max3(int16_t a, int16_t b, int16_t c) {
    int16_t m = (a > b) ? a : b;
    return (m > c) ? m : c;
}

/**
 * Compute Smith-Waterman alignment score.
 */
static int16_t compute_sw_score(
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
    int16_t max_score = 0;
    __dma_aligned int8_t pssm_row[32];
    
    /* Initialize row 0 */
    for (uint32_t j = 0; j <= target_len; j++) {
        H_prev[j] = 0;
        E[j] = NEG_INF;
        F_prev[j] = NEG_INF;
    }
    
    /* Process each query position (row i) */
    for (uint32_t i = 1; i <= query_len; i++) {
        /* Read PSSM row for query position (i-1) from MRAM */
        uintptr_t row_addr = pssm_mram_base + ((i - 1) * ALPHA_SIZE);
        uintptr_t aligned_addr = row_addr & ~7UL;
        uint32_t offset = row_addr & 7U;
        mram_read((__mram_ptr void*)aligned_addr, pssm_row, 32);
        int8_t* pssm_vals = &pssm_row[offset];
        
        /* First column initialization */
        H_curr[0] = 0;
        E[0] = NEG_INF;
        F_curr[0] = NEG_INF;
        
        /* Process each target position (column j) */
        for (uint32_t j = 1; j <= target_len; j++) {
            /* Get substitution score from PSSM */
            uint8_t aa = target_seq[j - 1];
            if (aa >= ALPHA_SIZE) aa = 20;
            int8_t sub_score = pssm_vals[aa];
            
            /* E[i][j] - horizontal gap (gap in query, extension along target) */
            int16_t E_extend = (E[j-1] > NEG_INF + GAP_EXTEND) ? (E[j-1] - GAP_EXTEND) : NEG_INF;
            int16_t E_open = (H_curr[j-1] > NEG_INF + GAP_OPEN + GAP_EXTEND) ? 
                             (H_curr[j-1] - GAP_OPEN - GAP_EXTEND) : NEG_INF;
            E[j] = max2(E_extend, E_open);
            
            /* F[i][j] - vertical gap (gap in target, extension along query) */
            int16_t F_extend = (F_prev[j] > NEG_INF + GAP_EXTEND) ? (F_prev[j] - GAP_EXTEND) : NEG_INF;
            int16_t F_open = (H_prev[j] > NEG_INF + GAP_OPEN + GAP_EXTEND) ? 
                             (H_prev[j] - GAP_OPEN - GAP_EXTEND) : NEG_INF;
            F_curr[j] = max2(F_extend, F_open);
            
            /* H[i][j] = max(0, diagonal + sub, E, F) */
            int16_t diag = H_prev[j-1] + sub_score;
            int16_t score = max3(diag, E[j], F_curr[j]);
            H_curr[j] = (score > 0) ? score : 0;
            
            if (H_curr[j] > max_score) {
                max_score = H_curr[j];
            }
        }
        
        /* Swap rows for next iteration */
        int16_t* temp = H_prev;
        H_prev = H_curr;
        H_curr = temp;
        
        temp = F_prev;
        F_prev = F_curr;
        F_curr = temp;
    }
    
    return max_score;
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
    }
    barrier_wait(&my_barrier);
    
    /* Allocate working memory: 5 arrays + target buffer */
    /* 5 * (5001) * 2 + 5000 = ~55KB per tasklet */
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
    
    bool force_gapped = (g_bd.flags & 1);
    int16_t min_score = g_bd.min_score;
    uint32_t query_len = g_bd.query_len;
    uintptr_t pssm_base = mram_base + g_bd.pssm_data_offset;
    
    for (uint32_t t = tasklet_id; t < g_bd.num_targets; t += NR_TASKLETS) {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        __dma_aligned GappedHit hit;
        hit.target_id = meta.target_id;
        hit.score = 0;
        hit.q_end = 0;
        hit.t_end = 0;
        hit.padding[0] = 0;
        hit.padding[1] = 0;
        hit.padding[2] = 0;
        
        uintptr_t result_addr = mram_base + g_bd.results_offset + (t * sizeof(GappedHit));
        
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_LEN) {
            mram_write(&hit, (__mram_ptr void*)result_addr, MRAM_ALIGN_SIZE(sizeof(GappedHit)));
            continue;
        }
        
        uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = MRAM_ALIGN_SIZE(meta.target_len);
        mram_read((__mram_ptr void*)seq_addr, target_seq, aligned_len);
        
        int16_t sw_score = compute_sw_score(
            target_seq,
            meta.target_len,
            query_len,
            pssm_base,
            H_prev,
            H_curr,
            E_row,
            F_prev,
            F_curr
        );
        
        if (sw_score >= min_score || force_gapped) {
            hit.score = sw_score;
        }
        
        mram_write(&hit, (__mram_ptr void*)result_addr, MRAM_ALIGN_SIZE(sizeof(GappedHit)));
    }
    
    return 0;
}
