#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <string.h>

// Force compile error if tasklets are too high
#if NR_TASKLETS > 4
#error "NR_TASKLETS must be <= 4 to avoid WRAM overflow"
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================
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
  uint32_t targets_total_size;
  uint32_t results_buffer_size;
} __attribute__((packed)) BatchDescriptor;

typedef struct {
  uint32_t target_id;
  uint32_t target_len;
  uint32_t offset_in_data;
  uint32_t padding;
} __attribute__((packed)) TargetMetadata;

typedef struct {
  uint32_t target_id;
  int16_t score;
  uint16_t q_end;
  uint16_t t_end;
  uint32_t pad1;
  uint16_t pad2;
} __attribute__((packed)) GappedHit;

#define KERNEL_AA_SLOTS 21
#define MAX_TARGET_WRAM_LEN 8192
#define GAP_OPEN 11
#define GAP_EXTEND 1

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

// Ungapped diagonal computation
static void compute_ungapped_diagonal(
    uint8_t* target_seq, uint32_t t_len, uint32_t q_len,
    uintptr_t pssm_mram_base, int16_t* diag_buffer,
    int16_t* out_score, int32_t* out_best_diag_idx) 
{
    uint32_t num_diags = q_len + t_len;
    for (uint32_t i = 0; i < num_diags; ++i) diag_buffer[i] = 0;

    int16_t max_score = 0;
    int32_t best_diag = 0;
    __dma_aligned int8_t temp_read_buf[32];

    for (uint32_t q = 0; q < q_len; ++q) {
        uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
        uintptr_t aligned_addr = row_addr & ~7U; 
        uint32_t offset = row_addr & 7U;
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
                    best_diag = diag_idx;
                }
            }
        }
    }
    *out_score = max_score;
    *out_best_diag_idx = best_diag;
}

// Gapped stage: banded Smith-Waterman around best diagonal
static void compute_gapped_score(
    uint8_t* target_seq, uint32_t t_len, uint32_t q_len,
    uintptr_t pssm_mram_base, int32_t best_diag_idx_ungapped,
    int16_t* out_score, uint16_t* out_q_end, uint16_t* out_t_end)
{
  const int16_t GAP_O = GAP_OPEN;
  const int16_t GAP_E = GAP_EXTEND;

  // Safe allocation for DP arrays per target length
  int16_t *H_prev = (int16_t*)mem_alloc(t_len * sizeof(int16_t));
  int16_t *H_curr = (int16_t*)mem_alloc(t_len * sizeof(int16_t));
  int16_t *E = (int16_t*)mem_alloc(t_len * sizeof(int16_t));
  if (!H_prev || !H_curr || !E) {
    *out_score = 0;
    *out_q_end = 0;
    *out_t_end = 0;
    return;
  }

  // initialize
  for (uint32_t i = 0; i < t_len; ++i) { H_prev[i] = 0; E[i] = 0; }

  int16_t max_score = 0;
  uint16_t best_q = 0;
  uint16_t best_t = 0;

  // iterate over query rows
  __dma_aligned int8_t temp_read_buf[32];
  for (uint32_t q = 0; q < q_len; ++q) {
    // load PSSM row
    uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
    uintptr_t aligned_addr = row_addr & ~7U;
    uint32_t offset = row_addr & 7U;
    mram_read((__mram_ptr void*)aligned_addr, temp_read_buf, 32);
    int8_t* pssm_vals = &temp_read_buf[offset];

    int16_t F = 0;
    int16_t H_left = 0; // H_curr at t-1

    for (uint32_t t = 0; t < t_len; ++t) {
      uint8_t aa = target_seq[t];
      if (aa >= KERNEL_AA_SLOTS) aa = 20;
      int16_t s = pssm_vals[aa];

      int16_t prevH = H_prev[t];

      // update E (gap in target / deletion)
      int16_t e = E[t] - GAP_E;
      int16_t candE = prevH - GAP_O;
      if (candE > e) e = candE;
      E[t] = e;

      // update F (gap in query / insertion) using H_left
      int16_t f = F - GAP_E;
      int16_t candF = H_left - GAP_O;
      if (candF > f) f = candF;
      F = f;

      int16_t diagH = (t > 0) ? H_prev[t-1] : 0;
      int32_t h = (int32_t)diagH + (int32_t)s;
      if (E[t] > h) h = E[t];
      if (F > h) h = F;
      if (h < 0) h = 0;

      H_curr[t] = (int16_t)h;
      H_left = H_curr[t];

      if (H_curr[t] > max_score) {
        max_score = H_curr[t];
        best_q = (uint16_t)q;
        best_t = (uint16_t)t;
      }
    }

    // swap H_prev and H_curr pointers
    int16_t *tmp = H_prev; H_prev = H_curr; H_curr = tmp;
  }

  // write outputs
  *out_score = max_score;
  *out_q_end = best_q;
  *out_t_end = best_t;

  // free allocated memory
  // mem_free may not be necessary as program terminates soon, but try to free
  // Note: not all runtimes provide mem_free; ignore if not present
  // mem_free(H_prev);
  // mem_free(H_curr);
  // mem_free(E);
}

int main() {
  uint32_t tasklet_id = me();
//   extern uint8_t __sys_used_mram_end[];
  uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

  if (tasklet_id == 0) {
    mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
    // Print basic batch info for debugging/log capture
    printf("DPU[%u] Batch: Targets=%u Queries=%u QLen=%u PSSM=%u\n", me(), g_bd.num_targets, g_bd.num_queries, g_bd.query_len, g_bd.pssm_total_size);
    if (sizeof(GappedHit) != 16) {
      printf("FATAL: GappedHit struct size mismatch! Size=%u\n", (unsigned)sizeof(GappedHit));
      return 0;
    }
  }
  barrier_wait(&my_barrier);

  uint8_t* task_target_seq = (uint8_t*)mem_alloc(MAX_TARGET_WRAM_LEN);
  uint32_t diag_buf_size = (g_bd.query_len + MAX_TARGET_WRAM_LEN) * sizeof(int16_t);
  if (diag_buf_size > 12288) return 0; 
  int16_t* task_diag_buf = (int16_t*)mem_alloc(diag_buf_size);

  if (task_target_seq == NULL || task_diag_buf == NULL) return 0;
  for (uint32_t i = tasklet_id; i < g_bd.num_targets; i += NR_TASKLETS) {
    TargetMetadata meta;
    uintptr_t meta_addr = mram_base + g_bd.targets_metadata_offset + (i * sizeof(TargetMetadata));
    mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

    GappedHit h;
    h.target_id = meta.target_id;
    h.score = 0;
    h.q_end = 0;
    h.t_end = 0;
    h.pad1 = 0;
    h.pad2 = 0;
    uintptr_t res_addr = mram_base + g_bd.results_offset + (i * sizeof(GappedHit));

    if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) {
      mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(GappedHit)));
      continue;
    }

    uintptr_t seq_addr = mram_base + g_bd.targets_data_offset + meta.offset_in_data;
    uint32_t aligned_len = (meta.target_len + 7) & ~7U;
    if (aligned_len > MAX_TARGET_WRAM_LEN) aligned_len = MAX_TARGET_WRAM_LEN; 
    mram_read((__mram_ptr void*)seq_addr, task_target_seq, aligned_len);

    // 1. Ungapped Filter
    int16_t ungapped_score = 0;
    int32_t best_diag = 0;
    compute_ungapped_diagonal(task_target_seq, meta.target_len, g_bd.query_len,
                  mram_base + g_bd.pssm_data_offset, task_diag_buf,
                  &ungapped_score, &best_diag);

    // === Instrumentation: log ungapped decision and gapped results for debugging ===
    bool debug_target = (i < 10) || (ungapped_score > 10);

    // 2. Gapped Alignment: run full gapped alignment when ungapped is promising
    if (ungapped_score >= 15) {
      if (debug_target) {
        printf("DPU[%u] Tgt=%u PASS Ungapped=%d -> Computing Gapped...\n", me(), meta.target_id, ungapped_score);
      }

      compute_gapped_score(
        task_target_seq, meta.target_len, g_bd.query_len,
        mram_base + g_bd.pssm_data_offset, best_diag,
        &h.score, &h.q_end, &h.t_end
      );

      if (debug_target) {
            printf("DPU[%u] Tgt=%u DONE Gapped=%d\n", me(), meta.target_id, h.score);
      }
    } else {
      if (debug_target) {
        printf("DPU[%u] Tgt=%u FAIL Ungapped=%d (<15)\n", me(), meta.target_id, ungapped_score);
      }
      h.score = 0;
    }

    mram_write(&h, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(GappedHit)));
  }
  return 0;
}