/* Gapped DPU kernel — Smith-Waterman prefilter (Safe Struct Refactor) */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "dpu_common.h"

/* --- Configuration & Constants --- */
#define NEG_INF_32 -1000000000

/* * SAFE MEMORY LAYOUT DEFINITION
 * This struct defines the exact WRAM usage per tasklet.
 * All DMA buffers are explicitly aligned to 8 bytes.
 */
typedef struct {
    /* 1. Vertical Vectors (Read/Write to MRAM) */
    int32_t H_col[Q_TILE_SIZE + 2] __attribute__((aligned(8)));
    int32_t E_col[Q_TILE_SIZE + 2] __attribute__((aligned(8)));

    /* 2. Horizontal Vectors (Internal SW State) */
    int32_t H_top[T_TILE_SIZE + 2] __attribute__((aligned(8)));
    int32_t F_top[T_TILE_SIZE + 2] __attribute__((aligned(8)));
    int32_t H_bot[T_TILE_SIZE + 2] __attribute__((aligned(8)));
    int32_t F_bot[T_TILE_SIZE + 2] __attribute__((aligned(8)));

    /* 3. Data Caches */
    /* Target tile needs to be aligned for MRAM read */
    uint8_t target_tile[T_TILE_SIZE + 8] __attribute__((aligned(8)));
    
    /* PSSM tile: Size = Q_TILE * 21. aligned for safety */
    int8_t pssm_tile[Q_TILE_SIZE * ALPHA_SIZE + 32] __attribute__((aligned(8)));

} TaskletScratch;

/* Global State */
__dma_aligned GappedBatchDescriptor g_bd;
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(hit_mutex);

__host uint32_t g_hit_count;
__host uint32_t g_hit_write_offset;
__host uint32_t g_overflow;
__host uint8_t g_effective_tasklets;

/* Global scratch pool management */
uint8_t *g_scratch_pool_base = NULL;

/* Cache for Query Metadata (shared across tasklets) */
__dma_aligned QueryMetadata g_query_meta[MAX_BATCH_QUERIES];

/* * Initialize MRAM Scratch Vectors (H and E columns) 
 * Uses the tasklet's own WRAM scratch as a temporary buffer to push zeros/inf to MRAM.
 */
static void init_mram_vectors(
    uintptr_t mram_H_vec,
    uintptr_t mram_E_vec,
    uint32_t num_q_tiles,
    uint32_t vec_stride,
    TaskletScratch *scratch)
{
    // Reuse H_col and E_col in WRAM as temporary init buffers
    int32_t *buf = scratch->H_col;
    uint32_t count = Q_TILE_SIZE + 1;
    uint32_t bytes = ALIGN8(count * sizeof(int32_t));

    // 1. Clear H vector (0)
    for(uint32_t i=0; i<count; i++) buf[i] = 0;
    for(uint32_t t=0; t<num_q_tiles; t++) {
         mram_write(buf, (__mram_ptr void*)(mram_H_vec + t*vec_stride), bytes);
    }

    // 2. Clear E vector (NEG_INF)
    for(uint32_t i=0; i<count; i++) buf[i] = NEG_INF_32;
    for(uint32_t t=0; t<num_q_tiles; t++) {
         mram_write(buf, (__mram_ptr void*)(mram_E_vec + t*vec_stride), bytes);
    }
}

static SwResult compute_sw_tiled(
    uintptr_t mram_base,
    uint32_t target_data_offset,
    uint32_t target_len,
    uint32_t query_len,
    uintptr_t pssm_mram_base,
    uintptr_t mram_scratch_vectors,
    TaskletScratch *scratch, 
    int32_t gap_open,    
    int32_t gap_extend)  
{
    SwResult result = (SwResult){0, 0, 0};
    int32_t max_score = 0; 
    
    // MRAM Vector Layout
    uint32_t vec_stride = ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));
    uint32_t num_q_tiles = (query_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE;
    uintptr_t mram_H_vec = mram_scratch_vectors;
    uintptr_t mram_E_vec = mram_scratch_vectors + (num_q_tiles * vec_stride);

    // Initialize MRAM vectors (Column 0 logic)
    init_mram_vectors(mram_H_vec, mram_E_vec, num_q_tiles, vec_stride, scratch);

    // Unpack scratch pointers for cleaner code
    int32_t *H_top = scratch->H_top;
    int32_t *F_top = scratch->F_top;
    int32_t *H_bot = scratch->H_bot;
    int32_t *F_bot = scratch->F_bot;
    int32_t *H_col = scratch->H_col;
    int32_t *E_col = scratch->E_col;
    uint8_t *target_tile = scratch->target_tile;
    int8_t  *pssm_tile = scratch->pssm_tile;

    uint32_t num_t_tiles = (target_len + T_TILE_SIZE - 1) / T_TILE_SIZE;

    // --- Tiling Loop ---
    for (uint32_t t_tile_idx = 0; t_tile_idx < num_t_tiles; t_tile_idx++)
    {
        uint32_t t_start = t_tile_idx * T_TILE_SIZE;
        uint32_t t_size = min_u32(T_TILE_SIZE, target_len - t_start);

        // Load Target Tile
        mram_read((__mram_ptr void *)(mram_base + target_data_offset + t_start),
                  target_tile, ALIGN8(t_size));

        // Initialize Horizontal vectors (Row 0 logic for this tile strip)
        for (uint32_t col = 0; col <= t_size; col++) {
            H_top[col] = 0;
            F_top[col] = NEG_INF_32;
        }

        uint32_t q_tile_idx = 0;
        for (uint32_t q_start = 0; q_start < query_len; q_start += Q_TILE_SIZE, q_tile_idx++)
        {
            uint32_t q_size = min_u32(Q_TILE_SIZE, query_len - q_start);
            uint32_t vec_bytes = ALIGN8((q_size + 1) * sizeof(int32_t));
            uint32_t mram_offset = q_tile_idx * vec_stride;

            // Load Vertical State (from previous target tile or init)
            if (t_tile_idx == 0) {
                // First target strip: implicit zeros (already done via init_mram_vectors, 
                // but local H_col needs reset if we want to be safe, though MRAM read is better)
                // Optimization: Just read back what we wrote (0 and NEG_INF) or set locally.
                // Reading back ensures consistency.
                mram_read((__mram_ptr void *)(mram_H_vec + mram_offset), H_col, vec_bytes);
                mram_read((__mram_ptr void *)(mram_E_vec + mram_offset), E_col, vec_bytes);
            } else {
                mram_read((__mram_ptr void *)(mram_H_vec + mram_offset), H_col, vec_bytes);
                mram_read((__mram_ptr void *)(mram_E_vec + mram_offset), E_col, vec_bytes);
            }

            // Load PSSM Tile
            uintptr_t tile_addr = pssm_mram_base + (uintptr_t)q_start * ALPHA_SIZE;
            uint32_t tile_bytes = q_size * ALPHA_SIZE;
            // Handle PSSM alignment
            if ((tile_addr & 7U) == 0) {
                mram_read_aligned_bulk(tile_addr, pssm_tile, tile_bytes);
            } else {
                mram_read_unaligned_bulk(tile_addr, pssm_tile, tile_bytes);
            }

            // Corner Case: top-left of this tile
            H_bot[0] = H_col[q_size];
            F_bot[0] = NEG_INF_32;

            // --- Inner DP Core ---
            for (uint32_t col = 1; col <= t_size; col++)
            {
                uint8_t aa = target_tile[col - 1];
                if (aa >= ALPHA_SIZE) aa = 20; // Sentinel/X

                int32_t h_up   = H_top[col];     
                int32_t f_up   = F_top[col];     
                int32_t h_diag = H_top[col - 1]; 
                
                for (uint32_t i = 1; i <= q_size; i++)
                {
                    int32_t h_left = H_col[i]; 
                    int32_t e_left = E_col[i]; 

                    // PSSM Access
                    int32_t sub = (int32_t)pssm_tile[(i - 1) * ALPHA_SIZE + aa];

                    // E(i,j) = max(H(i, j-1) - gap_open, E(i, j-1) - gap_ext)
                    int32_t e_ext  = e_left - gap_extend;
                    int32_t e_open = h_left - gap_open;
                    int32_t e_new  = (e_ext > e_open) ? e_ext : e_open;

                    // F(i,j) = max(H(i-1, j) - gap_open, F(i-1, j) - gap_ext)
                    int32_t f_ext  = f_up - gap_extend;
                    int32_t f_open = h_up - gap_open;
                    int32_t f_new  = (f_ext > f_open) ? f_ext : f_open;

                    // H(i,j) = max(0, H(i-1, j-1) + sub, E(i,j), F(i,j))
                    int32_t diag_score = h_diag + sub;
                    int32_t h_new = diag_score;
                    if (e_new > h_new) h_new = e_new;
                    if (f_new > h_new) h_new = f_new;
                    if (h_new < 0)     h_new = 0;

                    // Track Max
                    if (h_new > max_score) {
                        max_score = h_new;
                        result.score = h_new;
                        result.q_end = (uint16_t)(q_start + i);
                        result.t_end = (uint16_t)(t_start + col);
                    }

                    // Rotate
                    h_diag = h_left;
                    h_up   = h_new;
                    f_up   = f_new;

                    // Write Back Column
                    H_col[i] = h_new;
                    E_col[i] = e_new;
                }

                H_bot[col] = h_up;
                F_bot[col] = f_up;
            }

            // Save Vertical State to MRAM
            mram_write(H_col, (__mram_ptr void *)(mram_H_vec + mram_offset), vec_bytes);
            mram_write(E_col, (__mram_ptr void *)(mram_E_vec + mram_offset), vec_bytes);

            // Swap Row Buffers (Pointer swap)
            int32_t *tmp;
            tmp = H_top; H_top = H_bot; H_bot = tmp;
            tmp = F_top; F_top = F_bot; F_bot = tmp;
        }
    }
    result.score = max_score;
    return result;
}

int main() {
    uint32_t tasklet_id = me();
    uintptr_t mram_base = (uintptr_t)DPU_MRAM_HEAP_POINTER;
    
    /* 1. Load Batch Descriptor (Tasklet 0) */
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void*)mram_base, &g_bd, MRAM_ALIGN_SIZE(sizeof(GappedBatchDescriptor)));
        
        // Safety: Clamp requested tasklets
        uint8_t req = g_bd.header.num_active_tasklets;
        if (req == 0 || req > MAX_SAFE_TASKLETS) req = MAX_SAFE_TASKLETS;
        
        // Double Check WRAM Capacity via Struct Size
        uint32_t total_wram_needed = req * sizeof(TaskletScratch);
        uint32_t available_wram = 60 * 1024; // Conservative 60KB
        
        if (total_wram_needed > available_wram) {
            // Downscale dynamically if struct is too big
            req = available_wram / sizeof(TaskletScratch);
        }
        g_effective_tasklets = req;
    }
    barrier_wait(&my_barrier);

    bool is_active = (tasklet_id < g_effective_tasklets);

    /* 2. Global Init (Tasklet 0) */
    if (tasklet_id == 0) {
        // Reset Result Header
        __dma_aligned uint32_t hdr[2] = {0, 0};
        mram_write(hdr, (__mram_ptr void *)(mram_base + g_bd.header.results_offset), 8);
        g_hit_count = 0;
        g_hit_write_offset = 8;
        g_overflow = 0;

        // Load Query Metadata
        uint32_t num_q = g_bd.header.num_queries;
        if (num_q > MAX_BATCH_QUERIES) num_q = MAX_BATCH_QUERIES;
        uintptr_t qmeta_base = mram_base + g_bd.header.queries_metadata_offset;
        mram_read((__mram_ptr void*)qmeta_base, g_query_meta, MRAM_ALIGN_SIZE(num_q * sizeof(QueryMetadata)));
        
        // Allocate WRAM Scratch Pool
        mem_reset();
        uint32_t alloc_size = g_effective_tasklets * sizeof(TaskletScratch);
        g_scratch_pool_base = (uint8_t*)mem_alloc(alloc_size); // Single block alloc
    }
    barrier_wait(&my_barrier);

    /* 3. Pointer Setup */
    if (!g_scratch_pool_base || g_effective_tasklets == 0) return 0;
    
    // Assign strongly typed scratch pointer
    TaskletScratch *my_scratch = NULL;
    if (is_active) {
        my_scratch = (TaskletScratch*)(g_scratch_pool_base + (tasklet_id * sizeof(TaskletScratch)));
    }

    uint32_t max_query_len = g_bd.header.query_len;
    uintptr_t pssm_base_start = mram_base + g_bd.header.pssm_data_offset;
    uintptr_t results_base = mram_base + g_bd.header.results_offset;
    uint32_t total_buffer_size = g_bd.header.results_buffer_size;

    /* 4. MRAM Scratch Calculation (Safe Stride) */
    // Size = 2 vectors (H, E) per tasklet
    uint32_t vec_stride = ALIGN8((Q_TILE_SIZE + 1) * sizeof(int32_t));
    uint32_t num_q_tiles = (max_query_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE;
    uint32_t tasklet_mram_need = 2 * (num_q_tiles * vec_stride);
    uint32_t total_scratch_mram = g_effective_tasklets * tasklet_mram_need;
    
    // Scratch sits at the END of the results buffer
    uint32_t limit_for_hits = 0;
    if (total_buffer_size > total_scratch_mram) {
        limit_for_hits = total_buffer_size - total_scratch_mram;
    } else {
        is_active = false; // Error: No MRAM for scratch
    }

    uintptr_t my_mram_vectors = results_base + limit_for_hits + (tasklet_id * tasklet_mram_need);

    /* 5. Execution Loop */
    GappedHit local_hits[8];
    uint32_t local_count = 0;

    // Helper macro to flush hits
    #define FLUSH_LOCAL_HITS() do { \
        if (local_count) { \
            uint32_t bytes = local_count * sizeof(GappedHit); \
            mutex_lock(hit_mutex); \
            if (g_hit_write_offset + bytes > limit_for_hits) { \
                g_overflow = 1; \
                mutex_unlock(hit_mutex); \
                local_count = 0; \
            } else { \
                uint32_t offset = g_hit_write_offset; \
                g_hit_write_offset += bytes; \
                g_hit_count += local_count; \
                mutex_unlock(hit_mutex); \
                mram_write(local_hits, (__mram_ptr void *)(results_base + offset), bytes); \
                local_count = 0; \
            } \
        } \
    } while (0)

    if (is_active) {
        for (uint32_t t = tasklet_id; t < g_bd.header.num_targets; t += g_effective_tasklets)
        {
            __dma_aligned TargetMetadata meta;
            uintptr_t meta_addr = mram_base + g_bd.header.targets_metadata_offset + (t * sizeof(TargetMetadata));
            mram_read((__mram_ptr void*)meta_addr, &meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));

            if (!can_be_covered(max_query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) {
                continue;
            }

            for (uint32_t q_idx = 0; q_idx < g_bd.header.num_queries; ++q_idx) {
                QueryMetadata qmeta = g_query_meta[q_idx];
                
                SwResult sw = compute_sw_tiled(
                    mram_base,
                    g_bd.header.targets_data_offset + meta.offset_in_data, 
                    meta.target_len,
                    qmeta.query_len,
                    pssm_base_start + qmeta.pssm_offset_in_batch,
                    my_mram_vectors,
                    my_scratch, 
                    (int32_t)g_bd.gap_open_cost,   
                    (int32_t)g_bd.gap_extend_cost
                );
                
                if (sw.score < qmeta.min_score) {
                    continue;
                }

                uint16_t aln_len = (sw.q_end > sw.t_end) ? sw.q_end : sw.t_end;
                if (g_bd.min_aln_len > 0 && aln_len < g_bd.min_aln_len) 
                {
                    continue;
                }
                
                if (!has_coverage(sw.q_end, sw.t_end, qmeta.query_len, meta.target_len, g_bd.cov_mode, g_bd.cov_thr_pct)) 
                {
                    continue;
                }
                
                if (!passes_seq_id_threshold(sw.score, sw.q_end, sw.t_end, g_bd.seq_id_thr_pct)) 
                {
                    continue;
                }

                __dma_aligned GappedHit hit;
                hit.target_id = meta.target_id;
                hit.score = sw.score;
                hit.q_end = sw.q_end;
                hit.t_end = sw.t_end;
                hit.padding[0] = (uint16_t)q_idx;
                hit.padding[1] = 0;

                local_hits[local_count++] = hit;
                if (local_count == 8) FLUSH_LOCAL_HITS();
            }
        }
        FLUSH_LOCAL_HITS();
    }

    barrier_wait(&my_barrier);

    /* 6. Write Result Header (Tasklet 0) */
    if (tasklet_id == 0) {
        __dma_aligned uint32_t count_buf[2];
        count_buf[0] = g_hit_count;
        count_buf[1] = (uint32_t)g_overflow << 31 | (uint32_t)g_effective_tasklets;
        mram_write(count_buf, (__mram_ptr void*)results_base, 8);
    }
    
    return 0;
}