#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>

#include "DpuSharedTypes.h"

#define MAX_TARGET_WRAM_LEN 2048
#define MAX_DIAG_COUNTERS 512

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

typedef struct {
    uint16_t diagonal;
    uint8_t count;
    uint8_t valid;
} DiagCounter;

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
    }
    barrier_wait(&my_barrier);

    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
    DiagCounter* diag_counters = (DiagCounter*)mem_alloc(MAX_DIAG_COUNTERS * sizeof(DiagCounter));
    
    if (!task_target_seq || !diag_counters) {
        return 0;
    }

    uint32_t ksize = g_bd.kmer_size;
    if (ksize == 0) ksize = 6;
    
    uint32_t hash_table_size = g_bd.pssm_total_size;
    if (hash_table_size == 0) hash_table_size = 1;
    uint32_t hash_mask = hash_table_size - 1;
    uintptr_t hash_table_addr = mram_base + g_bd.pssm_data_offset;

    const uint32_t ALPHA_SIZE = 21;
    
    for (uint32_t i = tasklet_id; i < g_bd.num_targets; i += NR_TASKLETS) {
        TargetMetadata meta;
        mram_read((__mram_ptr void*)(mram_base + g_bd.targets_metadata_offset + i*sizeof(TargetMetadata)), 
                  &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

        Hit h;
        h.target_id = meta.target_id;
        h.query_id = 0;
        h.score = 0;
        h.diagonal = 0;
        h.pad1 = 0;
        h.pad2 = 0;
        uintptr_t res_addr = mram_base + g_bd.results_offset + (i * sizeof(Hit));
        
        if (meta.target_len < ksize || meta.target_len > MAX_TARGET_WRAM_LEN) {
            mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
            continue;
        }

        uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = MRAM_ALIGN_SIZE(meta.target_len);
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

        for (int d = 0; d < MAX_DIAG_COUNTERS; d++) {
            diag_counters[d].valid = 0;
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
            
            for (int probe = 0; probe < 8; probe++) {
                uint32_t slot = (hash_idx + probe) & hash_mask;
                uintptr_t entry_addr = hash_table_addr + slot * sizeof(KmerEntry);
                
                KmerEntry entry;
                mram_read((__mram_ptr void*)entry_addr, &entry, sizeof(KmerEntry));
                
                if (entry.kmer == 0) break;
                
                if (entry.kmer == target_kmer) {
                    uint16_t diag = (uint16_t)t_pos - (uint16_t)entry.query_pos;
                    
                    int found_slot = -1;
                    for (int d = 0; d < num_diag_slots_used; d++) {
                        if (diag_counters[d].valid && diag_counters[d].diagonal == diag) {
                            found_slot = d;
                            break;
                        }
                    }
                    
                    if (found_slot >= 0) {
                        if (diag_counters[found_slot].count < 255) {
                            diag_counters[found_slot].count++;
                        }
                    } else if (num_diag_slots_used < MAX_DIAG_COUNTERS) {
                        diag_counters[num_diag_slots_used].diagonal = diag;
                        diag_counters[num_diag_slots_used].count = 1;
                        diag_counters[num_diag_slots_used].valid = 1;
                        num_diag_slots_used++;
                    }
                }
            }
        }
        
        uint8_t best_count = 0;
        uint16_t best_diag = 0;
        for (int d = 0; d < num_diag_slots_used; d++) {
            if (diag_counters[d].valid && diag_counters[d].count > best_count) {
                best_count = diag_counters[d].count;
                best_diag = diag_counters[d].diagonal;
            }
        }
        
        if (best_count >= 2) {
            h.score = best_count;
            h.diagonal = (int16_t)best_diag;
        }
        
        mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
    }
    
    return 0;
}