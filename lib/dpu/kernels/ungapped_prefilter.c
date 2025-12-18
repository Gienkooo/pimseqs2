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

#define ALPHA_SIZE 21
#define MAX_TARGET_WRAM_LEN 6144
#define CHUNK_SIZE 512

__dma_aligned UngappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;

/* Compute simple diagonal heuristic and return score + best diagonal */
static void compute_ungapped_diagonal_with_diag(
    uint8_t *target_seq, uint32_t t_len, uint32_t q_len, uintptr_t pssm_mram_base,
    int16_t *diag_buffer, int16_t *out_score, int16_t *out_diag) 
{
    int16_t global_max_score = 0;
    int32_t global_best_diag = 0;
    
    // Total diagonals = t_len + q_len - 1. We size buffer for t_len + q_len.
    uint32_t num_diags = t_len + q_len;
    for (uint32_t i = 0; i < num_diags; ++i) diag_buffer[i] = 0;
    
    __dma_aligned int8_t temp_read_buf[32];
    
    for (uint32_t q_start = 0; q_start < q_len; q_start += CHUNK_SIZE) {
        uint32_t chunk_end = q_start + CHUNK_SIZE;
        if (chunk_end > q_len) chunk_end = q_len;
        
        for (uint32_t q = q_start; q < chunk_end; ++q) {
            // Calculate PSSM address: pssm_base + (q * 21)
            // Using shifts for x21: (q<<4) + (q<<2) + q
            uint32_t q_x_21 = (q << 4) + (q << 2) + q;
            uintptr_t row_addr = pssm_mram_base + q_x_21;
            
            mram_read((__mram_ptr void *)(row_addr & ~7U), temp_read_buf, 32);
            int8_t *pssm_vals = &temp_read_buf[row_addr & 7U];
            
            for (uint32_t t = 0; t < t_len; ++t) {
                uint8_t aa = target_seq[t];
                if (aa >= ALPHA_SIZE) aa = 20;
                
                // Diagonal Index: t - q + (q_len - 1)
                int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
                
                if (diag_idx >= 0 && diag_idx < (int32_t)num_diags) {
                    int16_t curr = diag_buffer[diag_idx] + pssm_vals[aa];
                    if (curr < 0) curr = 0;
                    diag_buffer[diag_idx] = curr;
                    
                    if (curr > global_max_score) {
                        global_max_score = curr;
                        global_best_diag = diag_idx;
                    }
                }
            }
        }
    }
    *out_score = global_max_score;
    /* diagonal index encoding: t - q + (q_len - 1) */
    *out_diag = (int16_t)global_best_diag; 
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(UngappedBatchDescriptor)));
        g_hit_count = 0;
        // Hit write offset usually starts after the counter (first 8 bytes)
        g_hit_write_offset = 8; 
    }
    barrier_wait(&my_barrier);

    /* DYNAMIC TASKLET CHECK: exit immediately if this tasklet is not active */
    if (!is_tasklet_active(g_bd.header.num_active_tasklets)) return 0;
    
    // Allocations
    uint8_t *task_target_seq = (uint8_t *)mem_alloc(MAX_TARGET_WRAM_LEN);
    // Buffer for diagonals: needs to hold t_len + q_len
    // Max requirement approx MAX_TARGET + MAX_QUERY. 
    // Allocating a safe upper bound e.g. 8KB or similar
    int16_t *diag_buffer = (int16_t *)mem_alloc(8192 * sizeof(int16_t));
    
    if (!task_target_seq || !diag_buffer) return 0;
    
    uint32_t query_len = g_bd.header.query_len;
    // For Ungapped mode, we use min_score from the batch descriptor
    int16_t min_score = g_bd.min_score; 
    uintptr_t pssm_base = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;
    
    for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += NR_TASKLETS) {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) continue;
        
        uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, MRAM_ALIGN_SIZE(meta.target_len));
        
        int16_t score = 0;
        int16_t diagonal = 0;
        
        compute_ungapped_diagonal_with_diag(
            task_target_seq, meta.target_len, query_len, pssm_base, 
            diag_buffer, &score, &diagonal
        );
        
        if (score >= min_score) {
            __dma_aligned Hit hit;
            hit.target_id = meta.target_id;
            hit.query_id = 0;      // Typically 0 for single-query batches
            hit.score = score;     // Correct score field for Hit struct
            hit.diagonal = diagonal;
            hit.pad1 = 0;
            hit.pad2 = 0;
            
            mutex_lock(hit_mutex);
            uint32_t offset = g_hit_write_offset;
            g_hit_write_offset += sizeof(Hit);
            g_hit_count++;
            mutex_unlock(hit_mutex);
            
            // Ensure we write to 8-byte aligned offset if MRAM requires it, 
            // but Hit is 16 bytes, so straightforward increment is fine.
            mram_write(&hit, (__mram_ptr void*)(results_base + offset), sizeof(Hit));
        }
    }
    
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = 0;
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}