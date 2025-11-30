#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
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

typedef enum {
  DOWN,
  RIGHT,
} Direction;

#define KERNEL_AA_SLOTS 21
#define MAX_TARGET_WRAM_LEN 8192
#define GAP_OPEN 11
#define GAP_EXTEND 1
#define W 128
#define NEG_INF -32768
#define X 2137

__dma_aligned BatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U)

static inline int16_t max(int16_t a, int16_t b) { return a > b ? a : b; }
static inline int16_t max3(int16_t a, int16_t b, int16_t c) {
  return max(a, max(b, c));
}
static inline int16_t min(int16_t a, int16_t b) { return a < b ? a : b; }
static inline int16_t min3(int16_t a, int16_t b, int16_t c) {
  return min(a, min(b, c));
}

// Ungapped diagonal computation
static void compute_ungapped_diagonal(uint8_t *target_seq, uint32_t t_len,
                                      uint32_t q_len, uintptr_t pssm_mram_base,
                                      int16_t *diag_buffer, int16_t *out_score,
                                      int32_t *out_best_diag_idx) {
  uint32_t num_diags = q_len + t_len;
  for (uint32_t i = 0; i < num_diags; ++i)
    diag_buffer[i] = 0;

  int16_t max_score = 0;
  int32_t best_diag = 0;
  __dma_aligned int8_t temp_read_buf[32];

  for (uint32_t q = 0; q < q_len; ++q) {
    uintptr_t row_addr = pssm_mram_base + (q * KERNEL_AA_SLOTS);
    uintptr_t aligned_addr = row_addr & ~7U;
    uint32_t offset = row_addr & 7U;
    mram_read((__mram_ptr void *)aligned_addr, temp_read_buf, 32);
    int8_t *pssm_vals = &temp_read_buf[offset];

    for (uint32_t t = 0; t < t_len; ++t) {
      uint8_t aa = target_seq[t];
      if (aa >= KERNEL_AA_SLOTS)
        aa = 20;
      int8_t score_val = pssm_vals[aa];
      int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);

      if (diag_idx >= 0 && diag_idx < num_diags) {
        int16_t prev = diag_buffer[diag_idx];
        int16_t curr = prev + score_val;
        if (curr < 0)
          curr = 0;
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

static void calc(uintptr_t pssm_mram_base, uint32_t v_a_start,
                 const uint8_t *target_subseq, uint16_t len, uint8_t *result) {

  __dma_aligned int8_t temp_read_buf[32];

  for (uint16_t k = 0; k < len; k++) {
    uintptr_t row_addr = pssm_mram_base + (v_a_start * KERNEL_AA_SLOTS);
    uintptr_t aligned_addr = row_addr & ~7U;
    uint32_t offset = row_addr & 7U;
    mram_read((__mram_ptr void *)aligned_addr, temp_read_buf, 32);

    int8_t *pssm_vals = &temp_read_buf[offset];

    uint8_t aa = target_subseq[k];
    if (aa >= KERNEL_AA_SLOTS)
      aa = 20;
    int16_t s = pssm_vals[aa];
    result[k] = s;
  }
}

void to_cartesian_coords(int16_t vec_idx, uint32_t center[2],
                         uint32_t *out_coords) {
  int16_t center_idx = W / 2;
  int16_t diff = vec_idx - center_idx;
  out_coords[0] = center[0] - diff;
  out_coords[1] = center[1] + diff;
}

// Gapped stage: banded Smith-Waterman around best diagonal
static void compute_gapped_score(uint8_t *target_seq, uint32_t t_len,
                                 uint32_t q_len, uintptr_t pssm_mram_base,
                                 int32_t best_diag_idx_ungapped,
                                 int16_t *out_score, uint16_t *out_q_end,
                                 uint16_t *out_t_end) {
  const int16_t Gi = GAP_OPEN;
  const int16_t Ge = GAP_EXTEND;

  int16_t ppv[W], pv[W], fv[W], ev[W], cv[W];

  for (uint16_t k = 0; k < W; ++k) {
    ppv[k] = fv[k] = ev[k] = NEG_INF;
    pv[k] = 0;
  }

  ppv[W / 2 + 1] = 0;
  pv[W / 2] = -6;
  pv[W / 2 + 1] = -6;

  int16_t center_max = pv[W / 2];
  Direction direction = DOWN;

  uint16_t max_score_band_idx = 0;
  uint16_t max_score_in_band_idx = 0;
  int16_t max_cv_subvec_elem = NEG_INF;
  uint32_t max_score_center[2];

  uint32_t iteration = 0;
  uint32_t i = 1, j = 0;

  uint32_t A_LEN = q_len;
  uint32_t B_LEN = t_len;

  while ((i < A_LEN && j <= B_LEN) || (i <= A_LEN && j < B_LEN)) {
    if (pv[W / 2] < center_max - X) {
      break;
    }

    Direction prev_direction = direction;
    if (i <= W / 2) {
      direction = (prev_direction == DOWN) ? RIGHT : DOWN;
    } else {
      direction = (pv[W - 1] > pv[0]) ? DOWN : RIGHT;
    }

    int16_t uv[W], lv[W];

    if (direction == DOWN) {
      j++;
      memcpy(uv, pv, W * sizeof(int16_t));

      memmove(lv, &pv[1], (W - 1) * sizeof(int16_t));
      lv[W - 1] = NEG_INF;

      memmove(ev, &ev[1], (W - 1) * sizeof(int16_t));
      ev[W - 1] = NEG_INF;

      if (prev_direction == DOWN) {
        memmove(ppv, &ppv[1], (W - 1) * sizeof(int16_t));
        ppv[W - 1] = NEG_INF;
      }
    } else { // RIGHT
      i++;
      uv[0] = NEG_INF;
      memmove(&uv[1], pv, (W - 1) * sizeof(int16_t));

      memcpy(lv, pv, W * sizeof(int16_t));

      memmove(&fv[1], fv, (W - 1) * sizeof(int16_t));
      fv[0] = NEG_INF;

      if (prev_direction == RIGHT) {
        memmove(&ppv[1], ppv, (W - 1) * sizeof(int16_t));
        ppv[0] = NEG_INF;
      }
    }

    for (uint16_t k = 0; k < W; ++k) {
      ev[k] = max(ev[k] - Ge, lv[k] - Gi - Ge);
      fv[k] = max(fv[k] - Ge, uv[k] - Gi - Ge);
    }

    if (W / 2 - j >= 0) {
      ev[W / 2 - j] = fv[W / 2 - j] = NEG_INF;
    }
    if (W / 2 + i < W) {
      ev[W / 2 + i] = fv[W / 2 + i] = NEG_INF;
    }

    uint32_t ppv_start = W / 2 - min3(j - 1, W / 2, A_LEN - i);
    uint32_t ppv_end = W / 2 + min3(i, W / 2, B_LEN - j + 1);
    ppv_start = max(0, ppv_start);
    ppv_end = min(W, ppv_end);

    uint32_t update_len = ppv_end - ppv_start;

    memcpy(cv, ppv, W * sizeof(int16_t));

    if (update_len > 0) {
      uint32_t v_a_start = max(i - min3(i, W / 2 - 1, B_LEN - j) - 1, 0);

      uint32_t v_b_start = j - min3(j, W / 2 + 1, A_LEN - i + 1);

      uint8_t scv[W];
      calc(pssm_mram_base, v_a_start, &target_seq[v_b_start], update_len, scv);

      int16_t cv_subvec[W];

      for (uint32_t k = 0; k < update_len; k++) {
        uint32_t idx = ppv_start + k;

        int16_t considered_cv_elem = ppv[idx] + scv[k];

        cv_subvec[k] = max3(ev[idx], fv[idx], considered_cv_elem);

        cv_subvec[k] = max(cv_subvec[k], 0);

        if (cv_subvec[k] > max_cv_subvec_elem) {
          max_cv_subvec_elem = cv_subvec[k];
          max_score_in_band_idx = k + ppv_start;
          max_score_band_idx = iteration;
          max_score_center[0] = i;
          max_score_center[1] = j;
        }
      }

      memcpy(cv, ppv, W * sizeof(int16_t));
      memcpy(&cv[ppv_start], cv_subvec,
             (ppv_end - ppv_start) * sizeof(int16_t));
    }

    if (W / 2 - j >= 0)
      cv[W / 2 - j] = -Gi - (i + j) * Ge;
    if (W / 2 + i < W)
      cv[W / 2 + i] = -Gi - (i + j) * Ge;

    memcpy(ppv, pv, W * sizeof(int16_t));
    memcpy(pv, cv, W * sizeof(int16_t));
    center_max = max(center_max, pv[W / 2]);
    iteration++;
  }

  // write outputs
  uint32_t best_coords[2];
  to_cartesian_coords(max_score_in_band_idx, max_score_center, best_coords);
  *out_score = max_cv_subvec_elem;
  *out_q_end = best_coords[0];
  *out_t_end = best_coords[1];
}

int main() {
  uint32_t tasklet_id = me();
  //   extern uint8_t __sys_used_mram_end[];
  uintptr_t mram_base = (uintptr_t)__sys_used_mram_end;

  if (tasklet_id == 0) {
    mram_read((__mram_ptr void *)mram_base, &g_bd,
              MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
    // Print basic batch info for debugging/log capture
    printf("DPU[%u] Batch: Targets=%u Queries=%u QLen=%u PSSM=%u\n", me(),
           g_bd.num_targets, g_bd.num_queries, g_bd.query_len,
           g_bd.pssm_total_size);
    if (sizeof(GappedHit) != 16) {
      printf("FATAL: GappedHit struct size mismatch! Size=%u\n",
             (unsigned)sizeof(GappedHit));
      return 0;
    }
  }
  barrier_wait(&my_barrier);

  uint8_t *task_target_seq = (uint8_t *)mem_alloc(MAX_TARGET_WRAM_LEN);
  uint32_t diag_buf_size =
      (g_bd.query_len + MAX_TARGET_WRAM_LEN) * sizeof(int16_t);
  if (diag_buf_size > 12288)
    return 0;
  int16_t *task_diag_buf = (int16_t *)mem_alloc(diag_buf_size);

  if (task_target_seq == NULL || task_diag_buf == NULL)
    return 0;
  for (uint32_t i = tasklet_id; i < g_bd.num_targets; i += NR_TASKLETS) {
    TargetMetadata meta;
    uintptr_t meta_addr =
        mram_base + g_bd.targets_metadata_offset + (i * sizeof(TargetMetadata));
    mram_read((__mram_ptr void *)meta_addr, &meta,
              MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

    GappedHit h;
    h.target_id = meta.target_id;
    h.score = 0;
    h.q_end = 0;
    h.t_end = 0;
    h.pad1 = 0;
    h.pad2 = 0;
    uintptr_t res_addr =
        mram_base + g_bd.results_offset + (i * sizeof(GappedHit));

    if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN) {
      mram_write(&h, (__mram_ptr void *)res_addr,
                 MRAM_ALIGN_SIZE(sizeof(GappedHit)));
      continue;
    }

    uintptr_t seq_addr =
        mram_base + g_bd.targets_data_offset + meta.offset_in_data;
    uint32_t aligned_len = (meta.target_len + 7) & ~7U;
    if (aligned_len > MAX_TARGET_WRAM_LEN)
      aligned_len = MAX_TARGET_WRAM_LEN;
    mram_read((__mram_ptr void *)seq_addr, task_target_seq, aligned_len);

    // 1. Ungapped Filter
    int16_t ungapped_score = 0;
    int32_t best_diag = 0;
    compute_ungapped_diagonal(task_target_seq, meta.target_len, g_bd.query_len,
                              mram_base + g_bd.pssm_data_offset, task_diag_buf,
                              &ungapped_score, &best_diag);

    // === Instrumentation: log ungapped decision and gapped results for
    // debugging ===
    bool debug_target = (i < 10) || (ungapped_score > 10);

    // 2. Gapped Alignment: run full gapped alignment when ungapped is promising
    if (ungapped_score >= 15) {
      if (debug_target) {
        printf("DPU[%u] Tgt=%u PASS Ungapped=%d -> Computing Gapped...\n", me(),
               meta.target_id, ungapped_score);
      }

      compute_gapped_score(task_target_seq, meta.target_len, g_bd.query_len,
                           mram_base + g_bd.pssm_data_offset, best_diag,
                           &h.score, &h.q_end, &h.t_end);

      if (debug_target) {
        printf("DPU[%u] Tgt=%u DONE Gapped=%d\n", me(), meta.target_id,
               h.score);
      }
    } else {
      if (debug_target) {
        printf("DPU[%u] Tgt=%u FAIL Ungapped=%d (<15)\n", me(), meta.target_id,
               ungapped_score);
      }
      h.score = 0;
    }

    mram_write(&h, (__mram_ptr void *)res_addr,
               MRAM_ALIGN_SIZE(sizeof(GappedHit)));
  }
  return 0;
}