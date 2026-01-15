/* Gapped DPU kernel — Smith-Waterman prefilter (no FP). */

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

#define ALPHA_SIZE 21
#define MAX_TARGET_LEN 5000

#define Q_TILE_SIZE 64
#define T_TILE_SIZE 64

#define GAP_OPEN 11
#define GAP_EXTEND 1

#define SCRATCH_SIZE 2560
#define MAX_SAFE_TASKLETS 8  /* Maximum tasklets that fit safely in WRAM */

/* Macros */
#define ALIGN8(x) (((x) + 7) & ~7U)
#define ALIGN8_PTR(p) ((void*)(((uintptr_t)(p) + 7) & ~((uintptr_t)7)))

__dma_aligned GappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);
MUTEX_INIT(job_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;
__host uint32_t g_next_target_idx;
__host uint32_t g_overflow;

#define MAX_BATCH_QUERIES 128
__dma_aligned QueryMetadata g_query_meta[MAX_BATCH_QUERIES];

typedef struct {
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
} SwResult;

/* Safe unaligned MRAM read helper (uses 32-byte temp buffer). */
static inline void mram_read_unaligned_bytes(uintptr_t src, void *dst, uint32_t len)
{
    uint32_t off = (uint32_t)(src & 7U);
    uintptr_t aligned_src = src & ~7U;
    uint32_t need = ALIGN8(len + off);

    __dma_aligned uint8_t tmp[32];

    // Read aligned region into temp buffer.
    mram_read((__mram_ptr void *)aligned_src, tmp, need);

    // Copy requested payload into destination.
    memcpy((uint8_t *)dst, tmp + off, len);
}

/* Unaligned MRAM read for larger payloads without oversized WRAM temps. */
static inline void mram_read_unaligned_bulk(uintptr_t src, void *dst, uint32_t len)
{
    uint32_t off = (uint32_t)(src & 7U);
    uintptr_t aligned = src & ~7U;
    uint8_t *out = (uint8_t *)dst;

    if (off != 0)
    {
        uint32_t head = 8U - off;
        if (head > len)
            head = len;
        __dma_aligned uint8_t tmp[8];
        mram_read((__mram_ptr void *)aligned, tmp, 8);
        memcpy(out, tmp + off, head);
        aligned += 8;
        out += head;
        len -= head;
    }

    uint32_t mid = len & ~7U;
    if (mid)
    {
        mram_read((__mram_ptr void *)aligned, out, mid);
        aligned += mid;
        out += mid;
        len -= mid;
    }

    if (len)
    {
        __dma_aligned uint8_t tmp[8];
        mram_read((__mram_ptr void *)aligned, tmp, 8);
        memcpy(out, tmp, len);
    }
}

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

            // Load PSSM rows for this query tile in a single block read
            uintptr_t tile_addr = pssm_mram_base + (uintptr_t)q_start * ALPHA_SIZE;
            uint32_t tile_bytes = q_size * ALPHA_SIZE;
            mram_read_unaligned_bulk(tile_addr, pssm_tile, tile_bytes);

            // Ensure column 0 boundary for this query tile
            H_bot[0] = H_col[q_size];
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

                    // E(i,col) from left (gap in target / horizontal gap) — saturating
                    int16_t e_ext = sat_sub(e_left, gap_extend);
                    int16_t e_open = sat_sub(h_left, gap_open);
                    int16_t e_new = max2(e_ext, e_open);

                    // F(i,col) from up (gap in query / vertical gap) — saturating
                    int16_t f_ext = sat_sub(f_up, gap_extend);
                    int16_t f_open = sat_sub(h_up, gap_open);
                    int16_t f_new = max2(f_ext, f_open);

                    // H(i,col) — saturating add for diagonal
                    int16_t diag = sat_add(h_diag, (int16_t)sub);
                    int16_t h_new = max3(diag, e_new, f_new);
                    if (h_new < 0)
                        h_new = 0;

                    // Track global max (update on ties to make endpoint deterministic)
                    if (h_new >= max_score)
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

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)DPU_MRAM_HEAP_POINTER;
    
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(GappedBatchDescriptor)));
    }
    barrier_wait(&my_barrier);

    /* Respect WRAM-safe limit: clamp num_active_tasklets to MAX_SAFE_TASKLETS */
    uint8_t effective_tasklets = g_bd.header.num_active_tasklets;
    if (effective_tasklets == 0 || effective_tasklets > MAX_SAFE_TASKLETS) {
        effective_tasklets = MAX_SAFE_TASKLETS;
    }
    
    /* DYNAMIC TASKLET CHECK: exit immediately if this tasklet is not active */
    if (tasklet_id >= effective_tasklets) return 0;

    if (tasklet_id == 0) {
        __dma_aligned uint32_t hdr[2] = {0, 0};
        mram_write(hdr, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);

        // Reset per-launch hit bookkeeping so the first write is well-defined
        g_hit_count = 0;
        g_hit_write_offset = 8; // hits start right after the 8-byte [count+pad] header
        g_overflow = 0;
        g_next_target_idx = 0;

        // Cache Query Metadata
        uint32_t num_q = g_bd.header.num_queries;
        if (num_q > MAX_BATCH_QUERIES) num_q = MAX_BATCH_QUERIES; // Safety
        uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
        mram_read((__mram_ptr void*)qmeta_base, g_query_meta, MRAM_ALIGN_SIZE(num_q * sizeof(QueryMetadata)));
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        // Reset WRAM allocator so per-launch allocations start from a clean state
        mem_reset();
    }
    barrier_wait(&my_barrier);

    uint8_t *scratch_buffer_raw  = (uint8_t *)mem_alloc(SCRATCH_SIZE + 8);
    uint8_t *scratch_buffer      = (uint8_t *)ALIGN8_PTR(scratch_buffer_raw);

    if (!scratch_buffer_raw) return 0;

    // header.query_len is now MAX query len in batch
    uint32_t max_query_len = g_bd.header.query_len;
    int16_t min_score = g_bd.min_score;
    
    uintptr_t pssm_base_start = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;
    
    uint32_t hits_area_size = MRAM_ALIGN_SIZE(g_bd.header.num_targets * g_bd.header.num_queries * sizeof(GappedHit) + 64);
    uintptr_t scratch_base = results_base + hits_area_size;

    uint32_t vec_size_bytes = MRAM_ALIGN_SIZE((max_query_len + 1) * sizeof(int16_t));
    uint32_t task_offset = tasklet_id * (2 * vec_size_bytes);
    uintptr_t mram_scratch_vectors = scratch_base + task_offset;

    const uint32_t HIT_STRIDE = MRAM_ALIGN_SIZE(sizeof(GappedHit));
    GappedHit local_hits[8];
    uint32_t local_count = 0;

#define FLUSH_LOCAL_HITS()                                                                 \
    do {                                                                                   \
        if (local_count) {                                                                 \
            uint32_t bytes = local_count * HIT_STRIDE;                                     \
            mutex_lock(hit_mutex);                                                         \
            if (g_hit_write_offset + bytes > hits_area_size) {                             \
                g_overflow = 1;                                                            \
                mutex_unlock(hit_mutex);                                                   \
                local_count = 0;                                                           \
            } else {                                                                       \
                uint32_t offset = g_hit_write_offset;                                      \
                g_hit_write_offset += bytes;                                               \
                g_hit_count += local_count;                                                \
                mutex_unlock(hit_mutex);                                                   \
                mram_write(local_hits, (__mram_ptr void *)(results_base + offset), bytes); \
                local_count = 0;                                                           \
            }                                                                              \
        }                                                                                  \
    } while (0)
    
    while (1) {
        mutex_lock(job_mutex);
        uint32_t t = g_next_target_idx++;
        mutex_unlock(job_mutex);

        if (t >= g_bd.header.num_targets) break;

        __dma_aligned TargetMetadata meta;
        uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
        mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
        
        if (meta.target_len == 0 || meta.target_len > MAX_TARGET_LEN) {
            continue;
        }
        
        /* FILTER 1: canBeCovered - skip pairs that can never achieve coverage */
        if (!can_be_covered(max_query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) {
            continue;
        }
        
        // Loop over all queries in the batch
        for (uint32_t q_idx = 0; q_idx < g_bd.header.num_queries; ++q_idx) {
            // Use cached metadata if available, else read from MRAM
            QueryMetadata qmeta;
            if (q_idx < MAX_BATCH_QUERIES) {
                qmeta = g_query_meta[q_idx];
            } else {
                uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
                mram_read((__mram_ptr void*)(qmeta_base + q_idx * sizeof(QueryMetadata)), &qmeta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
            }

            uintptr_t pssm_addr = pssm_base_start + qmeta.pssm_offset_in_batch;

            SwResult sw = compute_sw_tiled(
                mram_base,
                g_bd.header.targets_data_offset + meta.offset_in_data, 
                meta.target_len,
                qmeta.query_len,
                pssm_addr,
                mram_scratch_vectors,
                scratch_buffer,
                SCRATCH_SIZE,
                g_bd.gap_open_cost,
                g_bd.gap_extend_cost);
            
            /* FILTER 2: min_score threshold */
            if (sw.score < min_score) {
                continue;
            }
            
            /* FILTER 3: minimum alignment length */
            uint16_t aln_len = (sw.q_end > sw.t_end) ? sw.q_end : sw.t_end;
            if (g_bd.min_aln_len > 0 && aln_len < g_bd.min_aln_len) {
                continue;
            }
            
            /* FILTER 4: hasCoverage - check actual coverage from alignment */
            if (!has_coverage(sw.q_end, sw.t_end, qmeta.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) {
                continue;
            }
            
            /* FILTER 5: estimated sequence identity (NO DIVISION!) */
            if (!passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, g_bd.seq_id_thr_pct)) {
                continue;
            }
            
            /* All filters passed - write hit */
            __dma_aligned GappedHit hit;
            hit.target_id = meta.target_id;
            hit.score = sw.score;
            hit.q_end = sw.q_end;
            hit.t_end = sw.t_end;
            hit.padding[0] = (uint16_t)q_idx;
            hit.padding[1] = 0;
            hit.padding[2] = 0;

            local_hits[local_count++] = hit;
            if (local_count == (sizeof(local_hits) / sizeof(local_hits[0])))
                FLUSH_LOCAL_HITS();
        }
    }
    
    FLUSH_LOCAL_HITS();

#undef FLUSH_LOCAL_HITS

    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = g_overflow;
        /* Write count at offset 0 of results buffer */
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}
