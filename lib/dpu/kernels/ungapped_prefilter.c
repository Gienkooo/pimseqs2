#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <string.h>

#include "DpuSharedTypes.h"

#define KERNEL_AA_SLOTS 21
#define MAX_TARGET_WRAM_LEN 4096
#define MAX_QUERY_LEN_FOR_DIAG 4096

#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

static int16_t compute_score_streaming(
    uint8_t* target_seq, uint32_t t_len, uint32_t q_len,
    uintptr_t pssm_mram_base, int16_t* diag_buffer, int16_t* out_diagonal)
{
    uint32_t num_diags = q_len + t_len;
    for (uint32_t i = 0; i < num_diags; ++i) {
        diag_buffer[i] = 0;
    }

    int16_t max_score = 0;
    int16_t best_diag = 0;
    __dma_aligned int8_t pssm_row_cache[24];

    for (uint32_t q = 0; q < q_len; ++q) {
        uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
        uintptr_t aligned_addr = row_addr & ~7U; 
        uint32_t offset = row_addr & 7U;
        
        __dma_aligned int8_t temp_read_buf[32]; 
        mram_read((__mram_ptr void*)aligned_addr, temp_read_buf, 32);
        int8_t* pssm_vals = &temp_read_buf[offset];

        for (uint32_t t = 0; t < t_len; ++t) {
            uint8_t aa = target_seq[t];
            if (aa >= KERNEL_AA_SLOTS) aa = 20;
            int8_t score_val = pssm_vals[aa];
            int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
            
            if (diag_idx >= 0 && diag_idx < num_diags) {
                int16_t prev = diag_buffer[diag_idx];
                int16_t curr = prev + score_val;
                if (curr < 0) curr = 0;
                diag_buffer[diag_idx] = curr;
                if (curr > max_score) {
                    max_score = curr;
                    best_diag = (int16_t)((int32_t)q - (int32_t)t);
                }
            }
        }
    }

    if (out_diagonal) *out_diagonal = best_diag;
    return max_score;
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
    }
    barrier_wait(&my_barrier);

    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
    uint32_t effective_query_len = g_bd.query_len;
    if (effective_query_len > MAX_QUERY_LEN_FOR_DIAG) {
        effective_query_len = MAX_QUERY_LEN_FOR_DIAG;
    }
    uint32_t diag_limit = effective_query_len + MAX_TARGET_WRAM_LEN;
    int16_t* task_diag_buf = (int16_t*)mem_alloc(diag_limit * sizeof(int16_t));

    if (task_target_seq == NULL || task_diag_buf == NULL) {
        return 0;
    }

    for (uint32_t i = tasklet_id; i < g_bd.num_targets; i += NR_TASKLETS) {
        TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.targets_metadata_offset + (i * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

        Hit h;
        h.target_id = meta.target_id;
        h.query_id = 0;
        h.score = 0;
        h.diagonal = 0;
        h.pad1 = 0;
        h.pad2 = 0;
        uintptr_t res_addr = mram_base + g_bd.results_offset + (i * sizeof(Hit));

        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) {
            mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
            continue;
        }

        uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = (meta.target_len + 7) & ~7U;
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

        int16_t diag = 0;
        int16_t score = compute_score_streaming(
            task_target_seq, meta.target_len, effective_query_len,
            mram_base + g_bd.pssm_data_offset, task_diag_buf, &diag);

        h.score = score;
        h.diagonal = diag;
        mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
    }

    return 0;
}