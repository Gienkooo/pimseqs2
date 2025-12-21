#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <string.h>

#include "DpuSharedTypes.h"

#define AA_NUMBER 21
#define MAX_TARGET_LEN 2560
#define MAX_QUERY_LEN 2560

#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U) // 8 * ceil(x/8)

BARRIER_INIT(barrier, NR_TASKLETS);

__dma_aligned BatchDescriptor batch_descriptor;


static int16_t compute_score_streaming(
    uint8_t* target_seq,        // target sequence (aminoacids array)
    uint32_t t_len,             // target length
    uint32_t q_len,             // query length
    uintptr_t pssm_mram_base,   // pointer to start of mram memory
    int16_t* diag_buffer,       // scores
    int16_t* out_diagonal)      // index of the best diagonal (SIGNED)
{
    // Initialize
    uint32_t num_diags = q_len + t_len;
    for (uint32_t i = 0; i < num_diags; ++i) diag_buffer[i] = 0;
    int16_t max_score = 0;
    int32_t best_diag_idx = 0;

    // Iterate over query
    for (uint32_t q = 0; q < q_len; ++q) {
        // Load PSSM
        uintptr_t pssm_addr = pssm_mram_base + (q * AA_NUMBER);
        uintptr_t aligned_pssm_addr = pssm_addr & ~7U; 
        uint32_t offset = pssm_addr & 7U;
        __dma_aligned int8_t temp_pssm_buf[32]; 
        mram_read((__mram_ptr void*)aligned_pssm_addr, temp_pssm_buf, 32);
        int8_t* pssm_vals = &temp_pssm_buf[offset];

        // Iterate over target
        for (uint32_t t = 0; t < t_len; ++t) {
            uint8_t aa = target_seq[t];         // 0-19 - aminoacids, 20 - invalid
            if (aa >= AA_NUMBER) aa = 20;       // handle invalid
            int8_t score = pssm_vals[aa];       // cell score
            int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
            if (diag_idx >= 0 && diag_idx < num_diags) {
                int16_t prev = diag_buffer[diag_idx];
                int16_t curr = prev + score;
                if (curr < 0) curr = 0;
                diag_buffer[diag_idx] = curr;
                if (curr > max_score) {
                    max_score = curr;
                    best_diag_idx = diag_idx;
                }
            }
        }
    }

    if (out_diagonal) *out_diagonal = (int16_t)((int32_t)(q_len-1) - best_diag_idx); // TODO: validate
    return max_score;
}


int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    // 1. Initialization - metadata read (Tasklet 0)
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &batch_descriptor, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
        printf("DPU[%u] Batch: Targets=%u QLen=%u\n", me(), batch_descriptor.num_targets, batch_descriptor.query_len);
    }
    barrier_wait(&barrier);

    // 2. Allocation - target and scores
    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_LEN);
    int16_t* task_diag_buf = (int16_t*)mem_alloc(2 * MAX_TARGET_LEN * sizeof(int16_t));

    if (task_target_seq == NULL || task_diag_buf == NULL) return 0;

    // 3. Processing Loop - iterate over queries and targets
    for (uint32_t i = 0; i < batch_descriptor.num_queries; ++i) {
        for (uint32_t j = tasklet_id; j < batch_descriptor.num_targets; j += NR_TASKLETS) {
            // Load Query metadata
            QueryMetadata query_meta;
            uintptr_t query_meta_addr = mram_base + batch_descriptor.queries_metadata_offset + (i * sizeof(TargetMetadata));
            mram_read((__mram_ptr void*)query_meta_addr, &query_meta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
            uint32_t effective_query_len = query_meta.query_len;
            if (effective_query_len > MAX_QUERY_LEN) effective_query_len = MAX_QUERY_LEN;

            // Load Target metadata
            TargetMetadata target_meta;
            uintptr_t target_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + (j * sizeof(TargetMetadata));
            mram_read((__mram_ptr void*)target_meta_addr, &target_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

            // Initialize Hit structure
            Hit hit;
            hit.target_id = target_meta.target_id;
            hit.query_id = query_meta.query_id;
            hit.score = 0;
            hit.diagonal = 0;
            hit.pad1 = 0;
            hit.pad2 = 0;
            uintptr_t res_addr = mram_base + batch_descriptor.results_offset + ((i * batch_descriptor.num_targets + j) * sizeof(Hit));

            // Skip if invalid or too large
            if (target_meta.target_len == 0 || target_meta.target_len > MAX_TARGET_LEN) {
                mram_write(&hit, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
                continue;
            }

            // Load Target Sequence
            uintptr_t target_seq_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data;
            uint32_t aligned_target_len = MRAM_ALIGN_SIZE(target_meta.target_len);
            if (aligned_target_len > MAX_TARGET_LEN) aligned_target_len = MAX_TARGET_LEN;
            mram_read((__mram_ptr void*)target_seq_addr, task_target_seq, aligned_target_len);

            // Compute score and diagonal
            int16_t diag = 0;
            int16_t score = compute_score_streaming(
                task_target_seq, 
                target_meta.target_len,
                effective_query_len,
                mram_base + batch_descriptor.pssm_data_offset,
                task_diag_buf,
                &diag
            );

            // Store score and diagonal
            hit.score = score;
            hit.diagonal = diag;
            mram_write(&hit, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
            
            if (score > 0) { printf("DPU[%u] Hit TgtId=%u Score=%d\n", me(), target_meta.target_id, score); }
        }
    }

    return 0;
}
