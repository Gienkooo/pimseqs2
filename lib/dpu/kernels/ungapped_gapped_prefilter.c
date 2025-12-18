/* Combined ungapped+gapped DPU kernel (tiled, safe MRAM reads). */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dpu_common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

/* Configuration */
#define ALPHA_SIZE 21
#define MAX_TARGET_WRAM_LEN 6144

#define Q_TILE_SIZE 256
#define T_TILE_SIZE 256

#define GAP_OPEN 11
#define GAP_EXTEND 1

#define SCRATCH_SIZE 16384

/* Macros */
#define ALIGN8(x) (((x) + 7) & ~7U)
#define ALIGN8_PTR(p) ((void*)(((uintptr_t)(p) + 7) & ~((uintptr_t)7)))

/* Globals */
__dma_aligned CombinedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;

/* Safe unaligned MRAM read helper (uses 32-byte temp buffer). */
static inline void mram_read_unaligned_bytes(uintptr_t src, void *dst, uint32_t len)
{
    uint32_t off = (uint32_t)(src & 7U);
    uintptr_t aligned_src = src & ~7U;
    uint32_t need = ALIGN8(len + off);

    // Safety: for this kernel's usage (len=21), need <= 32 always.
    // If you ever reuse this helper for larger len, increase the buffer or assert.
    __dma_aligned uint8_t tmp[32];

    // Read aligned region into temp buffer.
    mram_read((__mram_ptr void *)aligned_src, tmp, need);

    // Copy requested payload into destination.
    memcpy((uint8_t *)dst, tmp + off, len);
}

/* Filters provided by dpu_common.h: can_be_covered, has_coverage, passes_seq_id_threshold */

/* -------------------------------------------------------------------------
 * UNGAPPED HEURISTIC
 * ------------------------------------------------------------------------- */
static void compute_ungapped_diagonal(
    uint8_t *target_seq, uint32_t t_len, uint32_t q_len, uintptr_t pssm_mram_base,
    int16_t *diag_buffer, int16_t *out_score)
{
    int16_t global_max_score = 0;
    uint32_t num_diags = t_len + q_len;
    for (uint32_t i = 0; i < num_diags; ++i)
        diag_buffer[i] = 0;

    __dma_aligned int8_t temp_read_buf[32];
    const uint32_t CHUNK = 512;

    for (uint32_t q_start = 0; q_start < q_len; q_start += CHUNK)
    {
        uint32_t chunk_end = min_u32(q_start + CHUNK, q_len);
        for (uint32_t q = q_start; q < chunk_end; ++q)
        {
            uintptr_t row_addr = pssm_mram_base + (q * ALPHA_SIZE);
            mram_read((__mram_ptr void *)(row_addr & ~7U), temp_read_buf, 32);
            int8_t *pssm_vals = &temp_read_buf[row_addr & 7U];

            for (uint32_t t = 0; t < t_len; ++t)
            {
                uint8_t aa = target_seq[t];
                if (aa >= ALPHA_SIZE)
                    aa = 20;

                int32_t diag_idx = (int32_t)t - (int32_t)q + (int32_t)(q_len - 1);

                if (diag_idx >= 0 && diag_idx < (int32_t)num_diags)
                {
                    int16_t curr = diag_buffer[diag_idx] + pssm_vals[aa];
                    if (curr < 0)
                        curr = 0;
                    diag_buffer[diag_idx] = curr;
                    if (curr > global_max_score)
                        global_max_score = curr;
                }
            }
        }
    }
    *out_score = global_max_score;
}

/* -------------------------------------------------------------------------
 * GAPPED ALIGNMENT (Tiled SW)
 * ------------------------------------------------------------------------- */
typedef struct
{
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
} SwResult;

static void mram_fill_i16(uintptr_t mram_addr, uint32_t num_elems, int16_t value)
{
    __dma_aligned int16_t buf[8];
    for (int i = 0; i < 8; i++)
        buf[i] = value;

    uint32_t total_bytes = ALIGN8(num_elems * sizeof(int16_t));
    for (uint32_t off = 0; off < total_bytes; off += 16)
    {
        uint32_t chunk = (total_bytes - off > 16) ? 16 : (total_bytes - off);
        if (chunk < 8)
            chunk = 8;
        mram_write(buf, (__mram_ptr void *)(mram_addr + off), ALIGN8(chunk));
    }
}

static SwResult compute_sw_tiled(
    uintptr_t mram_base,
    uint32_t target_data_offset,
    uint32_t target_len,
    uint32_t query_len,
    uintptr_t pssm_mram_base,
    uintptr_t mram_scratch_vectors,
    uint8_t *wram_scratch,
    uint32_t wram_scratch_size,
    int16_t gap_open,
    int16_t gap_extend)
{
    SwResult result = (SwResult){0, 0, 0};
    int16_t max_score = 0;
    // MRAM vectors: per-tasklet storage for right boundary between target tiles
    uint32_t col_vec_len_bytes = ALIGN8((query_len + 1) * sizeof(int16_t));
    uintptr_t mram_H_vec = mram_scratch_vectors;
    uintptr_t mram_E_vec = mram_scratch_vectors + col_vec_len_bytes;

    // Initialize MRAM vectors (Column 0, before first target tile)
    mram_fill_i16(mram_H_vec, query_len + 1, 0);
    mram_fill_i16(mram_E_vec, query_len + 1, NEG_INF);

    // WRAM layout
    uint8_t *ptr = wram_scratch;

    // Horizontal boundaries for query-tiling:
    // H_top/F_top: values at row boundary before processing current query tile
    // H_bot/F_bot: produced values at bottom of current query tile (fed into next tile)
    int16_t *H_top = (int16_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int16_t));
    int16_t *F_top = (int16_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int16_t));
    int16_t *H_bot = (int16_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int16_t));
    int16_t *F_bot = (int16_t *)ptr;
    ptr += ALIGN8((T_TILE_SIZE + 1) * sizeof(int16_t));

    // Vertical boundary vectors (left edge of this target tile), size Q_TILE_SIZE+1
    int16_t *H_col = (int16_t *)ptr;
    ptr += ALIGN8((Q_TILE_SIZE + 1) * sizeof(int16_t));
    int16_t *E_col = (int16_t *)ptr;
    ptr += ALIGN8((Q_TILE_SIZE + 1) * sizeof(int16_t));

    uint8_t *target_tile = (uint8_t *)ptr;
    ptr += ALIGN8(T_TILE_SIZE);
    int8_t *pssm_tile = (int8_t *)ptr;
    ptr += ALIGN8(Q_TILE_SIZE * ALPHA_SIZE + 8);

    if ((uint32_t)(ptr - wram_scratch) > wram_scratch_size)
        return result;

    uint32_t num_t_tiles = (target_len + T_TILE_SIZE - 1) / T_TILE_SIZE;

    for (uint32_t t_tile_idx = 0; t_tile_idx < num_t_tiles; t_tile_idx++)
    {
        uint32_t t_start = t_tile_idx * T_TILE_SIZE;
        uint32_t t_size = min_u32(T_TILE_SIZE, target_len - t_start);

        mram_read((__mram_ptr void *)(mram_base + target_data_offset + t_start),
                  target_tile, ALIGN8(t_size));

        // At the start of each target tile, the top boundary (row 0) is the DP row at the
        // query-tile boundary, initialized to 0/NEG_INF for the very first query tile.
        for (uint32_t col = 0; col <= t_size; col++)
        {
            H_top[col] = 0;
            F_top[col] = NEG_INF;
        }

        for (uint32_t q_start = 0; q_start < query_len; q_start += Q_TILE_SIZE)
        {
            uint32_t q_size = min_u32(Q_TILE_SIZE, query_len - q_start);

            // Load left boundary for this (target tile, query tile) from MRAM
            // Includes i=0..q_size
            uint32_t vec_bytes = ALIGN8((q_size + 1) * sizeof(int16_t));
            if (t_tile_idx == 0)
            {
                for (uint32_t i = 0; i <= q_size; i++)
                {
                    H_col[i] = 0;
                    E_col[i] = NEG_INF;
                }
            }
            else
            {
                mram_read((__mram_ptr void *)(mram_H_vec + q_start * sizeof(int16_t)), H_col, vec_bytes);
                mram_read((__mram_ptr void *)(mram_E_vec + q_start * sizeof(int16_t)), E_col, vec_bytes);
            }

            // Load PSSM rows for this query tile: pssm_tile[(i)*ALPHA + aa], i=0..q_size-1
            for (uint32_t i = 0; i < q_size; i++)
            {
                uintptr_t rowaddr = pssm_mram_base + (uintptr_t)(q_start + i) * ALPHA_SIZE;
                mram_read_unaligned_bytes(rowaddr, pssm_tile + i * ALPHA_SIZE, ALPHA_SIZE);
            }

            // Ensure column 0 boundary for this query tile
            H_bot[0] = 0;
            F_bot[0] = NEG_INF;

            // Process columns 1..t_size (col index in tile space)
            for (uint32_t col = 1; col <= t_size; col++)
            {
                uint8_t aa = target_tile[col - 1];
                if (aa >= ALPHA_SIZE)
                    aa = 20;

                // Top-of-tile boundary values for this column (i=0)
                int16_t h_up = H_top[col];       // H(0,col)
                int16_t f_up = F_top[col];       // F(0,col)
                int16_t h_diag = H_top[col - 1]; // H(0,col-1)

                // Sweep i = 1..q_size, update H_col/E_col in place (column-major)
                for (uint32_t i = 1; i <= q_size; i++)
                {
                    int16_t h_left = H_col[i]; // old H(i,col-1)
                    int16_t e_left = E_col[i]; // old E(i,col-1)

                    int8_t sub = pssm_tile[(i - 1) * ALPHA_SIZE + aa];

                    // E(i,col) from left (gap in target / horizontal gap)
                    int16_t e_ext = (e_left == NEG_INF) ? NEG_INF : (e_left - gap_extend);
                    int16_t e_open = (h_left == NEG_INF) ? NEG_INF : (h_left - gap_open);
                    int16_t e_new = max2(e_ext, e_open);

                    // F(i,col) from up (gap in query / vertical gap)
                    int16_t f_ext = (f_up == NEG_INF) ? NEG_INF : (f_up - gap_extend);
                    int16_t f_open = (h_up == NEG_INF) ? NEG_INF : (h_up - gap_open);
                    int16_t f_new = max2(f_ext, f_open);

                    // H(i,col)
                    int16_t diag = (h_diag == NEG_INF) ? NEG_INF : (h_diag + (int16_t)sub);
                    int16_t h_new = max3(diag, e_new, f_new);
                    if (h_new < 0)
                        h_new = 0;

                    // Track global max
                    if (h_new > max_score)
                    {
                        max_score = h_new;
                        result.score = h_new;
                        result.q_end = (uint16_t)(q_start + i);
                        result.t_end = (uint16_t)(t_start + col);
                    }

                    // Shift for next i:
                    // next diag should be old H(i,col-1)
                    h_diag = h_left;
                    h_up = h_new;
                    f_up = f_new;

                    // Store new column values
                    H_col[i] = h_new;
                    E_col[i] = e_new;
                }

                // Bottom boundary at i=q_size for this column -> feed next query tile
                H_bot[col] = h_up;
                F_bot[col] = f_up;
            }

            // Write right boundary (after finishing this target tile) to MRAM for the next target tile
            mram_write(H_col, (__mram_ptr void *)(mram_H_vec + q_start * sizeof(int16_t)), vec_bytes);
            mram_write(E_col, (__mram_ptr void *)(mram_E_vec + q_start * sizeof(int16_t)), vec_bytes);

            // Swap top/bottom boundaries for the next query tile
            int16_t *tmp;
            tmp = H_top;
            H_top = H_bot;
            H_bot = tmp;
            tmp = F_top;
            F_top = F_bot;
            F_bot = tmp;
        }
    }
    result.score = max_score;

    return result;
}

/* -------------------------------------------------------------------------
 * MAIN
 * ------------------------------------------------------------------------- */
int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)DPU_MRAM_HEAP_POINTER;

    printf("[DPU] Tasklet %u started, MRAM base at %p\n", tasklet_id, (void*)mram_base);
    printf("[DPU] NR_TASKLETS=%u, MAX_TARGET_WRAM_LEN=%u, SCRATCH_SIZE=%u\n",
           NR_TASKLETS, MAX_TARGET_WRAM_LEN, SCRATCH_SIZE);

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(CombinedBatchDescriptor)));
    }
    barrier_wait(&my_barrier);

    /* DYNAMIC TASKLET CHECK: exit immediately if this tasklet is not active */
    if (!is_tasklet_active(g_bd.header.num_active_tasklets)) return 0;

    if (tasklet_id == 0) {
        __dma_aligned uint32_t hdr[2] = {0, 0};
        mram_write(hdr, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);

        // Reset per-launch hit bookkeeping so the first write is well-defined
        g_hit_count = 0;
        g_hit_write_offset = 8; // hits start right after the 8-byte [count+pad] header
    }
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        uint32_t hitsArea = MRAM_ALIGN_SIZE(g_bd.header.num_targets * sizeof(GappedHit) + 64);
        uint32_t vecBytes = MRAM_ALIGN_SIZE((g_bd.header.query_len + 1) * sizeof(int16_t));
        uint32_t needed = hitsArea + (2 * vecBytes) * (NR_TASKLETS);

        printf("[DPU] results_buffer_size=%u needed=%u hitsArea=%u vecBytes=%u\n",
               g_bd.header.results_buffer_size, needed, hitsArea, vecBytes);
        if (needed > g_bd.header.results_buffer_size) {
            // hitCount already set to 0, so host gather is safe
            return 0;
        }
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        // Reset WRAM allocator so per-launch allocations start from a clean state
        mem_reset();
        printf("[DPU] Tasklet %u mem_reset done; allocating WRAM: MAX_TARGET_WRAM_LEN=%u SCRATCH_SIZE=%u\n",
               tasklet_id, MAX_TARGET_WRAM_LEN, SCRATCH_SIZE);
    }
    barrier_wait(&my_barrier);

    uint8_t *task_target_seq_raw = (uint8_t *)mem_alloc(MAX_TARGET_WRAM_LEN + 8);
    uint8_t *scratch_buffer_raw  = (uint8_t *)mem_alloc(SCRATCH_SIZE + 8);
    uint8_t *task_target_seq     = (uint8_t *)ALIGN8_PTR(task_target_seq_raw);
    uint8_t *scratch_buffer      = (uint8_t *)ALIGN8_PTR(scratch_buffer_raw);

    if (tasklet_id == 0) {
        if (!task_target_seq_raw || !scratch_buffer_raw) {
            printf("[DPU] Tasklet %u mem_alloc failed: seq=%p scratch=%p\n",
                   tasklet_id, task_target_seq_raw, scratch_buffer_raw);
        } else {
            printf("[DPU] Tasklet %u mem_alloc OK\n", tasklet_id);
        }
    }

    if (!task_target_seq_raw || !scratch_buffer_raw) return 0;

    uint32_t query_len = g_bd.header.query_len;
    int16_t min_ungapped_score = g_bd.min_ungapped_score;
    int16_t min_score = g_bd.min_score;
    bool force_gapped = (g_bd.header.flags & 1);

    uintptr_t pssm_base = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;

    uint32_t hits_area_size = MRAM_ALIGN_SIZE(g_bd.header.num_targets * sizeof(GappedHit) + 64);
    uintptr_t scratch_base = results_base + hits_area_size;

    uint32_t vec_size_bytes = MRAM_ALIGN_SIZE((g_bd.header.query_len + 1) * sizeof(int16_t));
    uint32_t task_offset = tasklet_id * (2 * vec_size_bytes);
    uintptr_t mram_scratch_vectors = scratch_base + task_offset;

    const uint32_t HIT_STRIDE = MRAM_ALIGN_SIZE(sizeof(GappedHit));

    for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += NR_TASKLETS)
    {
        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void *)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_WRAM_LEN)
            continue;
        if (!can_be_covered(query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct))
            continue;

        uint32_t diag_buf_size = ALIGN8((meta.target_len + query_len + 1) * sizeof(int16_t));
        int16_t ungapped_score = 0;

        if (diag_buf_size <= SCRATCH_SIZE)
        {
            uintptr_t seq_addr = mram_base + g_bd.header.targets_data_offset + meta.offset_in_data;
            mram_read((__mram_ptr void *)seq_addr, task_target_seq, MRAM_ALIGN_SIZE(meta.target_len));

            int16_t *diag_buffer = (int16_t *)scratch_buffer;
            compute_ungapped_diagonal(task_target_seq, meta.target_len, query_len, pssm_base, diag_buffer, &ungapped_score);

            if (!force_gapped && ungapped_score < min_ungapped_score)
                continue;
        }

        SwResult sw = compute_sw_tiled(
            mram_base,
            g_bd.header.targets_data_offset + meta.offset_in_data,
            meta.target_len,
            query_len,
            pssm_base,
            mram_scratch_vectors,
            scratch_buffer,
            SCRATCH_SIZE,
            g_bd.gap_open_cost,
            g_bd.gap_extend_cost);

        if (sw.score < min_score)
            continue;

        uint16_t aln_len = max_u32((uint32_t)sw.q_end, (uint32_t)sw.t_end);
        if (g_bd.min_aln_len > 0 && aln_len < g_bd.min_aln_len)
            continue;
        if (!has_coverage(sw.q_end, sw.t_end, g_bd.header.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct))
            continue;
        if (!passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, g_bd.seq_id_thr_pct))
            continue;

        __dma_aligned GappedHit hit;
        hit.target_id = meta.target_id;
        hit.score = sw.score;
        hit.q_end = sw.q_end;
        hit.t_end = sw.t_end;
        hit.padding[0] = 0;
        hit.padding[1] = 0;
        hit.padding[2] = 0;

        mutex_lock(hit_mutex);
        if (g_hit_write_offset + HIT_STRIDE > hits_area_size) {
            // No more space for hits in the reserved region; drop hit safely.
            // Print a short notice to help debugging (avoid flooding by printing minimal info).
            mutex_unlock(hit_mutex);
            printf("[DPU] Tasklet %u dropping hit: ghitwriteoffset=%u hitsArea=%u\n",
                   tasklet_id, g_hit_write_offset, hits_area_size);
        } else {
            uint32_t offset = g_hit_write_offset;
            g_hit_write_offset += HIT_STRIDE;
            g_hit_count++;
            mutex_unlock(hit_mutex);

            mram_write(&hit, (__mram_ptr void *)(results_base + offset), HIT_STRIDE);
        }
    }

    barrier_wait(&my_barrier);
    if (tasklet_id == 0)
    {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = 0;
        mram_write(count_buf, (__mram_ptr void *)(results_base), 8);
    }

    return 0;
}
