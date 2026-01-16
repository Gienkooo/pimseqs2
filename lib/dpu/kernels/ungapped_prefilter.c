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
#define MRAM_MAX 2048
#define MAX_TARGET_LEN 2048
#define MAX_QUERY_LEN 2048

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U) // 8 * ceil(x/8)

BARRIER_INIT(barrier, NR_TASKLETS);

__dma_aligned BatchDescriptor batch_descriptor;
__dma_aligned QueryMetadata query_meta;
__dma_aligned TargetMetadata target_meta;

uint8_t* target_buffer;
int8_t* pssm_buffer;
uint8_t* target;
int8_t* pssm;

typedef struct {
    int16_t score;
    int16_t diagonal;
} TaskletResult;

TaskletResult tasklet_results[NR_TASKLETS];


static void align_short(uint32_t tasklet_id, uint32_t t_len, uint32_t q_len) {
    int16_t max_score = 0;
    int16_t best_diag = 0;
    uint32_t num_diags = q_len + t_len - 1;

    // target & query debugging
    // for (int32_t i = 0; i < t_len; ++i) printf("%d\t", target[i]); printf("\n");
    // for (int32_t j = 0; j < q_len; ++j) { for(int32_t i = 0; i < 21; ++i) printf("%d\t", pssm[j*21+i]); printf("\n"); }

    // Iterate over diagonals
    for (uint32_t diag_idx = tasklet_id; diag_idx < num_diags; diag_idx += 1) {
        int32_t delta = (int32_t)diag_idx - (int32_t)q_len + 1;
        int32_t q_start = 0;
        int32_t q_end = q_len;
        if (-delta > q_start) q_start = -delta;
        if ((int32_t)t_len - delta < q_end) q_end = (int32_t)t_len - delta;
        if (q_start >= q_end) continue;

        // Iterate over diagonal
        int16_t diag_score = 0;
        int16_t score = 0;
        for (int32_t q = q_start; q < q_end; ++q) {
            int32_t t = q + delta;
            uint8_t aa = target[t];
            if (aa >= AA_NUMBER) aa = 20;
            int16_t val = (int16_t) pssm[q * AA_NUMBER + aa];
            score += val;
            if (score < 0) score = 0;
            if (score > diag_score) diag_score = score;
        }

        // Update Tasklet Best
        if (diag_score > max_score) {
            max_score = diag_score;
            best_diag = (int16_t)((int32_t)(q_len - 1) - (int32_t)diag_idx); 
        }
    }
    tasklet_results[tasklet_id].score = max_score;
    tasklet_results[tasklet_id].diagonal = best_diag;
}


int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    // Load batch descriptor
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &batch_descriptor, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
        printf("DPU[%u] Batch: Targets=%u QLen=%u\n", me(), batch_descriptor.num_targets, batch_descriptor.query_len);
        target_buffer = (uint8_t*)mem_alloc(MAX_TARGET_LEN);
        pssm_buffer = (int8_t*)mem_alloc(MAX_QUERY_LEN * AA_NUMBER);
    }
    barrier_wait(&barrier);

    if (target_buffer == NULL) return 0;

    // Iterate over queries
    for (uint32_t i = 0; i < batch_descriptor.num_queries; ++i) {

        // Load query metadata
        if (tasklet_id == 0) {
            uintptr_t query_meta_addr = mram_base + batch_descriptor.queries_metadata_offset + (i * sizeof(QueryMetadata));
            mram_read((__mram_ptr void*)query_meta_addr, &query_meta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
            uint32_t aligned_pssm_size = MRAM_ALIGN_SIZE(query_meta.query_len * AA_NUMBER);
            uintptr_t pssm_addr = mram_base + batch_descriptor.pssm_data_offset + query_meta.pssm_offset_in_batch;
            uintptr_t aligned_pssm_addr = pssm_addr & ~7U;
            uint32_t offset = pssm_addr & 7U;
            uint32_t pssm_read_offset = 0;
            for(;pssm_read_offset + MRAM_MAX < aligned_pssm_size; pssm_read_offset += MRAM_MAX) {
                mram_read((__mram_ptr void*)(aligned_pssm_addr + pssm_read_offset), pssm_buffer + pssm_read_offset, MRAM_MAX);
            } mram_read((__mram_ptr void*)(aligned_pssm_addr + pssm_read_offset), pssm_buffer + pssm_read_offset, aligned_pssm_size - pssm_read_offset);
            pssm = pssm_buffer + offset;
        }
        barrier_wait(&barrier);

        // Iterate over targets
        for (uint32_t j = 0; j < batch_descriptor.num_targets; j += 1) {

            if(tasklet_id == 0) {
                // Load Target Metadata
                uintptr_t target_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + (j * sizeof(TargetMetadata));
                mram_read((__mram_ptr void*)target_meta_addr, &target_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

                // Load Target Sequence
                if (target_meta.target_len > 0 && target_meta.target_len <= MAX_TARGET_LEN) {
                    uint32_t aligned_target_size = MRAM_ALIGN_SIZE(target_meta.target_len);
                    uintptr_t target_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data;
                    uintptr_t aligned_target_addr = target_addr & ~7U;
                    uint32_t offset = target_addr & 7U;
                    mram_read((__mram_ptr void*)aligned_target_addr, target_buffer, aligned_target_size);
                    target = target_buffer + offset;
                }
            }
            barrier_wait(&barrier);

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
            if (target_meta.target_len == 0 || target_meta.target_len > MAX_TARGET_LEN
                || query_meta.query_len == 0 || query_meta.query_len > MAX_QUERY_LEN) {
                mram_write(&hit, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
                continue;
            }

            // Load Target Sequence
            if (tasklet_id == 0) {
                uintptr_t target_seq_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data;
                uint32_t aligned_target_len = MRAM_ALIGN_SIZE(target_meta.target_len);
                if (aligned_target_len > MAX_TARGET_LEN) aligned_target_len = MAX_TARGET_LEN;
                mram_read((__mram_ptr void*)target_seq_addr, target_buffer, aligned_target_len);
            }
            // Initialize results
            tasklet_results[tasklet_id].score = 0;
            tasklet_results[tasklet_id].diagonal = 0;
            barrier_wait(&barrier);

            // Compute score and diagonal

            align_short(tasklet_id, target_meta.target_len, query_meta.query_len);
            barrier_wait(&barrier);

            // Store score and diagonal
            if (tasklet_id == 0) {
                int16_t best_score = 0;
                int16_t best_diag = 0;
                for (int k = 0; k < NR_TASKLETS; k++) {
                    if (tasklet_results[k].score > best_score) {
                        best_score = tasklet_results[k].score;
                        best_diag = tasklet_results[k].diagonal;
                    }
                }
                Hit hit;
                hit.target_id = target_meta.target_id;
                hit.query_id = query_meta.query_id;
                hit.score = best_score;
                hit.diagonal = best_diag;
                hit.pad1 = 0;
                hit.pad2 = 0;

                uintptr_t res_addr = mram_base + batch_descriptor.results_offset + ((i * batch_descriptor.num_targets + j) * sizeof(Hit));
                mram_write(&hit, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));

                printf("DPU[%u] Q=%u T=%u S=%d D=%d\n", me(), i, target_meta.target_id, best_score, best_diag);
            }

            barrier_wait(&barrier);
        }
    }

    return 0;
}
