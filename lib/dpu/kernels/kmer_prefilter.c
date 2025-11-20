#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>

// Data structures (must match host `DpuStructures.h`)
typedef struct {
  uint32_t batch_id;
  uint32_t num_queries;
  uint32_t num_targets;
  uint32_t query_len;
  uint32_t queries_metadata_offset;
  uint32_t pssm_data_offset; // Used for Hash Table Offset
  uint32_t targets_metadata_offset;
  uint32_t targets_data_offset;
  uint32_t results_offset;
  uint32_t pssm_total_size; // Used for Hash Table Size
  uint32_t kmer_size;
  uint32_t targets_total_size;
  uint32_t results_buffer_size;
        uint32_t reserved;
} __attribute__((packed)) BatchDescriptor;

typedef struct {
  uint32_t target_id;
  uint32_t target_len;
  uint32_t offset_in_data;
  uint32_t padding;
} __attribute__((packed)) TargetMetadata;

typedef struct {
    uint32_t target_id;
    uint16_t query_id;
    int16_t score;
    int16_t diagonal;
    uint32_t pad1;
    uint16_t pad2;
} __attribute__((packed)) Hit;

typedef struct {
  uint32_t kmer;
  uint16_t query_id;
  uint16_t query_pos;
} __attribute__((packed)) KmerEntry;

// Constants and globals
#define MAX_TARGET_WRAM_LEN 8192
#define MAX_QUERIES_PER_BATCH 128
// KMER size will be read from the BatchDescriptor at runtime (bd.kmer_size)

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

int main() {
    uint32_t tasklet_id = me();
    // extern uint8_t __sys_used_mram_end[];
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    // 1. Initialization
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
        // Print basic batch info for debugging/log capture
        printf("DPU[%u] Batch: Targets=%u Queries=%u QLen=%u K=%u\n", me(), g_bd.num_targets, g_bd.num_queries, g_bd.query_len, g_bd.kmer_size);
        if (sizeof(Hit) != 16) {
            printf("FATAL: Hit struct size mismatch! Size=%u\n", (unsigned)sizeof(Hit));
            return 0;
        }
    }
    barrier_wait(&my_barrier);

    // Allocation
    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
    
    // Track hits per query in the batch
    // per-query counters and last-diagonal
    int16_t* query_hits = (int16_t*)mem_alloc(MAX_QUERIES_PER_BATCH * sizeof(int16_t));
    int16_t* query_diags = (int16_t*)mem_alloc(MAX_QUERIES_PER_BATCH * sizeof(int16_t));

    if (!task_target_seq || !query_hits || !query_diags) return 0;

    // Hash table setup (power-of-two mask)
    const uint32_t table_mask = (g_bd.pssm_total_size / sizeof(KmerEntry)) - 1; 
    const uintptr_t table_base = mram_base + g_bd.pssm_data_offset;

    // k-mer size for this batch (read from descriptor)
    uint32_t ksize = g_bd.kmer_size;
    if (ksize == 0) ksize = 6;

    // 3. Loop Targets
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

        // Clear Tracking
        // Fast clear for small batch size
        for(int q=0; q<g_bd.num_queries && q<MAX_QUERIES_PER_BATCH; q++) {
            query_hits[q] = 0;
            query_diags[q] = 0;
        }

        // Load Sequence
        uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = (meta.target_len + 7) & ~7U;
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

        // 4. K-mer Scan
        uint32_t current_kmer = 0;

        // Prime k-mer
        for (int j=0; j<(int)ksize-1; j++) {
            current_kmer = (current_kmer << 5) | (task_target_seq[j] & 0x1F);
        }

        for (int t = (int)ksize-1; t < (int)meta.target_len; ++t) {
            // Shift and Add
            current_kmer = ((current_kmer << 5) | (task_target_seq[t] & 0x1F)) & 0x3FFFFFFF; 
            
            // Hash Lookup
            uint32_t idx = current_kmer & table_mask;
            
            // Linear Probing (Check up to 4 slots)
            for (int probe=0; probe<4; probe++) {
                KmerEntry entry;
                uintptr_t entry_addr = table_base + ((idx + probe) & table_mask) * sizeof(KmerEntry);
                
                // Read single entry (8 bytes, aligned)
                mram_read((__mram_ptr void*)entry_addr, &entry, sizeof(KmerEntry));
                
                if (entry.kmer == current_kmer) {
                    uint16_t q_id = entry.query_id;
                    if (q_id < MAX_QUERIES_PER_BATCH) {
                        // Hit Logic
                        // Diagonal = q_pos - t_pos (Standard)
                        // We store: t_pos - q_pos
                        int16_t diag = (int16_t)t - (int16_t)entry.query_pos;
                        
                        // Simple "2 hits on same diagonal" check
                        // If previous hit exists, and diagonal matches (within band of 32), boost score
                        int16_t last = query_diags[q_id];
                        int16_t diff = diag - last;
                        if (query_hits[q_id] == 0 || (diff > -16 && diff < 16)) {
                            query_hits[q_id]++;
                            query_diags[q_id] = diag; // Update last diagonal
                        }
                    }
                }
                
                if (entry.kmer == 0 && entry.query_id == 0) break; // Empty slot
            }
        }

        int16_t max_k_score = 0;
        int16_t best_diag = 0;
        uint16_t best_qid = 0;
        
        for(int q=0; q<g_bd.num_queries && q<MAX_QUERIES_PER_BATCH; q++) {
            if (query_hits[q] > max_k_score) {
                max_k_score = query_hits[q];
                best_diag = query_diags[q];
                best_qid = (uint16_t)q;
            }
        }

        if (max_k_score >= 2) {
            h.query_id = best_qid;
            h.score = max_k_score * 10;
            h.diagonal = -best_diag;
        }
        
        mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
    }
    return 0;
}