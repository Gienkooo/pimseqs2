#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <string.h>

// Data structures (must match host `DpuStructures.h`)
typedef struct {
  uint32_t batch_id;
  uint32_t num_queries;
  uint32_t num_targets;
  uint32_t query_len;
  uint32_t queries_metadata_offset;
  uint32_t pssm_data_offset;
  uint32_t targets_metadata_offset;
  uint32_t targets_data_offset;
  uint32_t results_offset;
  uint32_t pssm_total_size;
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

// Globals
#define KERNEL_AA_SLOTS 21
#define MAX_TARGET_WRAM_LEN 8192
#define NR_TASKLETS_HC 1
__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS_HC);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

/* Compute ungapped score by streaming PSSM rows into WRAM. */
static int16_t compute_score_streaming(
    uint8_t* target_seq, 
    uint32_t t_len,
    uint32_t q_len,
    uintptr_t pssm_mram_base,
    int16_t* diag_buffer,
    int16_t* out_diagonal)
{
    // Reset diagonal buffer. Index range: 0 .. (q_len + t_len - 1)
    uint32_t num_diags = q_len + t_len;
    for (uint32_t i = 0; i < num_diags; ++i) {
        diag_buffer[i] = 0;
    }

    int16_t max_score = 0;
    int16_t best_diag = 0;

    // Temporary WRAM buffer for aligned PSSM reads
    __dma_aligned int8_t pssm_row_cache[24];

    // 2. Iterate Query Positions (Outer Loop)
    for (uint32_t q = 0; q < q_len; ++q) {
        
        // Load PSSM row (aligned read)
        uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
        
        // Handle alignment: mram_read addr must be % 8 == 0.
        uintptr_t aligned_addr = row_addr & ~7U; 
        uint32_t offset = row_addr & 7U;
        
        // We need 21 bytes. Reading 32 bytes (4 longs) from aligned start covers all cases.
        __dma_aligned int8_t temp_read_buf[32]; 
        mram_read((__mram_ptr void*)aligned_addr, temp_read_buf, 32);
        
        // Point to the PSSM values for this query position
        int8_t* pssm_vals = &temp_read_buf[offset];

        // Iterate over target sequence and update diagonals
        for (uint32_t t = 0; t < t_len; ++t) {
            uint8_t aa = target_seq[t];
            if (aa >= KERNEL_AA_SLOTS) aa = 20; // Clamp invalid to 'X'

            // Lookup score for this AA at this Query Pos
            int8_t score_val = pssm_vals[aa];

            // Calculate Diagonal Index
            // k = t - q. We shift by +q_len to make it positive array index.
            // range: 0 to (t_len + q_len)
            int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);
            
            // Safety check
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
    // extern uint8_t __sys_used_mram_end[];
    uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

    // 1. Initialization (Tasklet 0)
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
        printf("DPU[%u] Batch: Targets=%u QLen=%u\n", me(), g_bd.num_targets, g_bd.query_len);
        // Verify struct size matches Host expectation (16 bytes)
        if (sizeof(Hit) != 16) {
            printf("FATAL: Hit struct size mismatch! Size=%u\n", (unsigned)sizeof(Hit));
            return 0;
        }
    }

    barrier_wait(&my_barrier);

    // 2. Allocation
    uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
    
    // Diagonal buffer needs to hold (Q + T) elements.
    // 2048 (Target) + 2048 (Max Query) = 4096 * 2 bytes = 8KB. Fits easily in WRAM.
    uint32_t diag_limit = g_bd.query_len + MAX_TARGET_WRAM_LEN;
    int16_t* task_diag_buf = (int16_t*)mem_alloc(diag_limit * sizeof(int16_t));

    if (task_target_seq == NULL || task_diag_buf == NULL) {
        printf("DPU[%u] OOM Failure\n", me());
        return 0;
    }

    // 3. Processing Loop
    for (uint32_t i = tasklet_id; i < g_bd.num_targets; i += NR_TASKLETS) {
        TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.targets_metadata_offset + (i * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

        Hit h;
        h.target_id = meta.target_id;
        h.query_id = 0; // single-query batches set query_id=0
        h.score = 0;
        h.diagonal = 0;
        h.pad1 = 0;
        h.pad2 = 0;
        uintptr_t res_addr = mram_base + g_bd.results_offset + (i * sizeof(Hit));

        // Skip if invalid or too large for WRAM buffer
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) {
            mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
            continue;
        }

        // Load Target Sequence
        uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
        uint32_t aligned_len = (meta.target_len + 7) & ~7U;
        
        if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN;
        mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

           if (i == 0 && tasklet_id == 0) {
               printf("DPU[%u] Tgt[0] FirstAA=%d Len=%u\n", me(), task_target_seq[0], meta.target_len);
           }

        // Compute Score (Streaming PSSM) and diagonal
        int16_t diag = 0;
        int16_t score = compute_score_streaming(
            task_target_seq, 
            meta.target_len,
            g_bd.query_len,
            mram_base + g_bd.pssm_data_offset,
            task_diag_buf,
            &diag
        );

        h.score = score;
        h.diagonal = diag;
        h.query_id = 0;
        mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));
        
           if (score > 20 && i < 5) {
               printf("DPU[%u] Hit TgtId=%u Score=%d\n", me(), meta.target_id, score);
           }
    }

    return 0;
}