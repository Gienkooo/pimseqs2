#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>
#include <stdio.h>

#include "dpu_common.h"

#define MAX_TARGET_WRAM_LEN 4096
#define MAX_DIAG_COUNTERS 4096

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

__dma_aligned KmerBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

/* Hit counter - written to MRAM after results for host to read */
__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;

typedef struct {
    uint16_t diagonal;
    uint16_t count;
} DiagCounter;

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(KmerBatchDescriptor)));
        g_hit_count = 0;
        g_hit_write_offset = 8; // Standardized: start writing hits after 8-byte count header
    }
    barrier_wait(&my_barrier);

    /* DYNAMIC TASKLET CHECK: exit immediately if this tasklet is not active */
    if (!is_tasklet_active(g_bd.header.num_active_tasklets)) return 0;

    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
    DiagCounter* diag_counters = (DiagCounter*)mem_alloc(MAX_DIAG_COUNTERS * sizeof(DiagCounter));
    
    if (!task_target_seq || !diag_counters) {
        return 0;
    }

    uint32_t ksize = g_bd.kmer_size;
    if (ksize == 0) ksize = 6;
    
    // Get threshold from host (default to 2 if not set)
    int16_t min_score_thr = g_bd.min_score;
    if (min_score_thr < 2) min_score_thr = 2;
    
    uint32_t hash_table_size = g_bd.header.pssm_total_size;
    if (hash_table_size == 0) hash_table_size = 1;
    uint32_t hash_mask = hash_table_size - 1;
    uintptr_t hash_table_addr = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;

    
    for (uint32_t i = tasklet_id; i < g_bd.header.num_targets; i += NR_TASKLETS) {
        TargetMetadata meta;
        mram_read((__mram_ptr void*)(mram_base + g_bd.header.targets_metadata_offset + i*sizeof(TargetMetadata)), 
                  &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        if (meta.target_len < ksize || meta.target_len > MAX_TARGET_WRAM_LEN) {
            continue;
        }

        uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = MRAM_ALIGN_SIZE(meta.target_len);
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

        for (int d = 0; d < MAX_DIAG_COUNTERS; d++) {
            diag_counters[d].count = 0;
        }
        int num_diag_slots_used = 0;

        uint8_t use_spaced = g_bd.use_spaced_kmers;
        uint8_t pattern_span = g_bd.spaced_pattern_span;
        if (pattern_span == 0) pattern_span = ksize;
        
        int num_positions = (int)meta.target_len - (int)pattern_span + 1;
        for (int t_pos = 0; t_pos < num_positions; t_pos++) {
            uint32_t target_kmer = 0;
            uint32_t power = 1;
            
            if (use_spaced) {
                for (int j = 0; j < (int)ksize; j++) {
                    uint8_t offset = g_bd.spaced_pattern[j];
                    uint8_t aa = task_target_seq[t_pos + offset];
                    if (aa >= ALPHA_SIZE) aa = 20;
                    target_kmer += aa * power;
                    power *= ALPHA_SIZE;
                }
            } else {
                for (int j = 0; j < (int)ksize; j++) {
                    uint8_t aa = task_target_seq[t_pos + j];
                    if (aa >= ALPHA_SIZE) aa = 20;
                    target_kmer += aa * power;
                    power *= ALPHA_SIZE;
                }
            }
            
            uint32_t hash_idx = target_kmer & hash_mask;
            
            for (int probe = 0; probe < 512; probe++) {
                uint32_t slot = (hash_idx + probe) & hash_mask;
                uintptr_t entry_addr = hash_table_addr + slot * sizeof(KmerEntry);
                
                KmerEntry entry;
                mram_read((__mram_ptr void*)entry_addr, &entry, sizeof(KmerEntry));
                
                if (entry.kmer == 0) break;
                
                if (entry.kmer == target_kmer) {
                    uint16_t diag = (uint16_t)t_pos - (uint16_t)entry.query_pos;
                    
                    int found_slot = -1;
                    for (int d = 0; d < num_diag_slots_used; d++) {
                        if (diag_counters[d].count > 0 && diag_counters[d].diagonal == diag) {
                            found_slot = d;
                            break;
                        }
                    }
                    
                    if (found_slot >= 0) {
                        if (diag_counters[found_slot].count < 65535) {
                            diag_counters[found_slot].count++;
                        }
                    } else if (num_diag_slots_used < MAX_DIAG_COUNTERS) {
                        diag_counters[num_diag_slots_used].diagonal = diag;
                        diag_counters[num_diag_slots_used].count = 1;
                        num_diag_slots_used++;
                    }
                }
            }
        }
        
        // Find best diagonal with count >= threshold
        uint16_t best_count = 0;
        uint16_t best_diag = 0;
        for (int d = 0; d < num_diag_slots_used; d++) {
            if (diag_counters[d].count > best_count) {
                best_count = diag_counters[d].count;
                best_diag = diag_counters[d].diagonal;
            }
        }
        
        // Only write hit if count meets threshold
        if (best_count >= (uint16_t)min_score_thr) {
            mutex_lock(hit_mutex);
            uint32_t offset = g_hit_write_offset;
            g_hit_write_offset += MRAM_ALIGN_SIZE(sizeof(Hit));
            g_hit_count++;
            mutex_unlock(hit_mutex);

            Hit h;
            h.target_id = meta.target_id;
            h.query_id = 0;
            h.score = (int16_t)best_count;
            h.diagonal = (int16_t)best_diag;
            h.pad1 = 0;
            h.pad2 = 0;
            mram_write(&h, (__mram_ptr void*)(results_base + offset), MRAM_ALIGN_SIZE(sizeof(Hit)));
        }
    }
    
    /* Write hit count to MRAM after all results */
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
