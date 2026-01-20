#ifndef DPU_COMMON_H
#define DPU_COMMON_H

#include <stdint.h>
#include <string.h>
#include <defs.h>
#include <mram.h>
#include "../shared/DpuSharedTypes.h"

/* --- Macros & Constants --- */

#define ALIGN8(x) (((x) + 7) & ~7U)
#define ALIGN8_PTR(p) ((void*)(((uintptr_t)(p) + 7) & ~((uintptr_t)7)))
#define MRAM_ALIGN_SIZE(x) ALIGN8(x)

/* Debug print control - disable for production and benchmark builds */
#ifndef DPU_DEBUG
#define DPU_DEBUG 0
#endif

#if DPU_DEBUG
#include <stdio.h>
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) do {} while(0)
#endif

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

/* --- Common Constants --- */
#define NEG_INF (-10000)  /* Safe value for int16 math */

/* =============================================================================
 * WRAM BUDGET CALCULATION (64KB total):
 * 
 *   - Stack per tasklet:         ~2KB   (DPU runtime requirement)
 *   - Global descriptors:        ~256B  (g_bd, hit counters)
 *   - Global query cache:        ~1KB   (g_query_meta[128] * 8B)
 *   - Per-tasklet scratch heap:  variable (allocated via mem_alloc)
 *
 * For N tasklets, available heap = 64KB - (N * 2KB) - 1.5KB globals = ~62.5KB - N*2KB
 *   - 8 tasklets:  62.5 - 16 = 46.5KB heap → 5.8KB per tasklet scratch
 *   - 11 tasklets: 62.5 - 22 = 40.5KB heap → 3.7KB per tasklet scratch  
 *   - 14 tasklets: 62.5 - 28 = 34.5KB heap → 2.5KB per tasklet scratch
 *   - 16 tasklets: 62.5 - 32 = 30.5KB heap → 1.9KB per tasklet scratch
 *
 * Scratch requirements per tasklet:
 *   - Ungapped: diag_buffer + pssm_cache + target_chunk ≈ 2-4KB depending on seq len
 *   - Gapped SW: H_top/F_top/H_bot/F_bot/H_col/E_col + target_tile + pssm_tile ≈ 2KB
 *   - Combined: max(ungapped, gapped) ≈ 3-4KB
 *
 * Safe defaults: 8-11 tasklets with 3-4KB scratch each.
 * ============================================================================= */

/* Tasklet limits - WRAM is 64KB shared by all tasklets */
#define MAX_SAFE_TASKLETS 11    /* Conservative limit for combined kernel */
#define MAX_BATCH_QUERIES 128   /* Maximum queries cached in WRAM (global, shared) */

/* Calculate per-tasklet scratch budget based on tasklet count.
 * Formula: (60KB available - globals) / num_tasklets with safety margin */
#define WRAM_AVAILABLE_HEAP(n)  (60 * 1024 - (n) * 2048)
#define SCRATCH_PER_TASKLET(n)  (WRAM_AVAILABLE_HEAP(n) / (n))

/* Default gap costs (can be overridden by batch descriptor) */
#define DEFAULT_GAP_OPEN 11
#define DEFAULT_GAP_EXTEND 1

/* --- Smith-Waterman Result Type --- */
typedef struct {
    int32_t score;
    uint16_t q_end;
    uint16_t t_end;
} SwResult;

/* Compile-time shift-based multiplication helpers */
#define MUL_2(x)   ((x) << 1)
#define MUL_3(x)   (((x) << 1) + (x))
#define MUL_4(x)   ((x) << 2)
#define MUL_10(x)   (((x) << 3) + ((x) << 1))
#define MUL_21(x)   (((x) << 4) + ((x) << 2) + (x))
#define MUL_100(x)  (((x) << 6) + ((x) << 5) + ((x) << 2))
#define MUL_1656(x) (((x) << 10) + ((x) << 9) + ((x) << 7) - ((x) << 3))

/* --- Utilities --- */

static inline int16_t max2(int16_t a, int16_t b) { return (a > b) ? a : b; }
static inline int16_t max3(int16_t a, int16_t b, int16_t c) { return max2(a, max2(b, c)); }
static inline uint32_t min_u32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline uint32_t max_u32(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

static inline int16_t sat_sub(int16_t a, int16_t b) { return (a <= NEG_INF + b) ? NEG_INF : a - b; }
static inline int16_t sat_add(int16_t a, int16_t b) {
    int32_t res = (int32_t)a + (int32_t)b;
    if (res < NEG_INF) return NEG_INF;
    if (res > 32767) return 32767;
    return (int16_t)res;
}

/* --- Common Filter Logic --- */

static inline int can_be_covered(uint32_t query_len, uint32_t target_len, uint8_t cov_mode, uint8_t cov_thr_pct) {
    if (cov_thr_pct == 0) return 1;
    uint32_t q100 = MUL_100(query_len);
    uint32_t t100 = MUL_100(target_len);
    uint32_t thr_q = (uint32_t)cov_thr_pct * query_len;
    uint32_t thr_t = (uint32_t)cov_thr_pct * target_len;
    
    switch (cov_mode) {
        case DPU_COV_MODE_BIDIRECTIONAL: return (q100 >= thr_t) && (t100 >= thr_q);
        case DPU_COV_MODE_TARGET: return (q100 >= thr_t);
        case DPU_COV_MODE_QUERY: return (t100 >= thr_q);
        case DPU_COV_MODE_LENGTH_QUERY: return (t100 >= thr_q) && (target_len <= query_len);
        case DPU_COV_MODE_LENGTH_TARGET: return (q100 >= thr_t) && (query_len <= target_len);
        case DPU_COV_MODE_LENGTH_SHORTER: {
            uint32_t shorter = min_u32(query_len, target_len);
            uint32_t longer = max_u32(query_len, target_len);
            return (MUL_100(shorter) >= (uint32_t)cov_thr_pct * longer);
        }
        default: return 1;
    }
}

static inline int has_coverage(uint16_t q_end, uint16_t t_end, uint32_t query_len, uint32_t target_len,
                        uint8_t cov_mode, uint8_t cov_thr_pct) {
    if (cov_thr_pct == 0) return 1;
    uint32_t q_cov_100 = MUL_100((uint32_t)q_end);
    uint32_t t_cov_100 = MUL_100((uint32_t)t_end);
    uint32_t thr_q = (uint32_t)cov_thr_pct * query_len;
    uint32_t thr_t = (uint32_t)cov_thr_pct * target_len;
    
    switch (cov_mode) {
        case DPU_COV_MODE_BIDIRECTIONAL: return (q_cov_100 >= thr_q) && (t_cov_100 >= thr_t);
        case DPU_COV_MODE_TARGET: return (t_cov_100 >= thr_t);
        case DPU_COV_MODE_QUERY: return (q_cov_100 >= thr_q);
        default: return 1;
    }
}

static inline int passes_seq_id_threshold(int32_t score, uint16_t q_end, uint16_t t_end, uint8_t seq_id_thr_pct) {
    if (seq_id_thr_pct == 0) return 1;
    if (score <= 0) return 0;
    uint32_t aln_len = max_u32(q_end, t_end);
    if (aln_len == 0) return 0;
    
    /* seqId = (score/aln_len)*0.1656 + 0.1141 */
    int32_t lhs = MUL_1656(score);
    int32_t rhs_factor = (int32_t)MUL_100((uint32_t)seq_id_thr_pct) - 1141; // Using *100 for % scale
    if (rhs_factor <= 0) return 1;
    int32_t rhs = (int32_t)aln_len * rhs_factor;
    return (lhs >= rhs);
}

/* Optimized aligned bulk read - use when host guarantees alignment.
 * Skips all unaligned path overhead. Use for PSSM data (host pads to 32 bytes). */
static inline void mram_read_aligned_bulk(uintptr_t src, void *dst, uint32_t len)
{
    /* Fast path: source is aligned, use direct DMA */
    uint32_t aligned_len = ALIGN8(len);
    if (aligned_len < 8) aligned_len = 8;
    
    /* For large reads, process in chunks up to 2048 bytes (DMA limit) */
    uint8_t *out = (uint8_t *)dst;
    while (aligned_len >= 2048)
    {
        mram_read((__mram_ptr void *)src, out, 2048);
        src += 2048;
        out += 2048;
        aligned_len -= 2048;
    }
    if (aligned_len > 0)
    {
        mram_read((__mram_ptr void *)src, out, aligned_len);
    }
}

/* Safe unaligned MRAM read helper for SMALL payloads (up to 24 bytes).
 * For larger payloads, use mram_read_unaligned_bulk() instead. */
static inline void mram_read_unaligned_bytes(uintptr_t src, void *dst, uint32_t len)
{
    uint32_t off = (uint32_t)(src & 7U);
    uintptr_t aligned_src = src & ~7U;
    uint32_t need = ALIGN8(len + off);
    
    /* Enforce DMA constraints: min 8 bytes, max 2048 bytes */
    if (need < 8) need = 8;
    if (need > 32) need = 32;  /* Buffer size limit */

    __dma_aligned uint8_t tmp[32];

    /* Read aligned region into temp buffer */
    mram_read((__mram_ptr void *)aligned_src, tmp, need);

    /* Copy requested payload into destination */
    memcpy((uint8_t *)dst, tmp + off, len);
}

/* Bulk unaligned MRAM read for larger payloads - more efficient than multiple small reads */
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

    /* Middle section: process aligned chunks of at least 8 bytes.
     * Note: Direct MRAM read to 'out' requires 'out' to be aligned,
     * which is guaranteed since we handled the head above. */
    uint32_t mid = len & ~7U;
    if (mid >= 8)  /* Enforce minimum 8-byte DMA constraint */
    {
        mram_read((__mram_ptr void *)aligned, out, mid);
        aligned += mid;
        out += mid;
        len -= mid;
    }
    else if (mid > 0)
    {
        /* Less than 8 bytes aligned: use temp buffer */
        __dma_aligned uint8_t tmp[8];
        mram_read((__mram_ptr void *)aligned, tmp, 8);
        memcpy(out, tmp, mid);
        aligned += 8;
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

/* Optimized MRAM fill using 64-byte buffer to balance efficiency vs stack usage.
 * With 11+ tasklets, stack space is precious. 64 bytes = 32 int16 elements.
 * Trade-off: 2x more writes than 128-byte buf, but safer stack footprint. */
static void mram_fill_i16(uintptr_t mram_addr, uint32_t num_elems, int16_t value)
{
    /* Use 64-byte buffer (32 int16 elements) - keeps stack usage low */
    __dma_aligned int16_t buf[32];
    for (int i = 0; i < 32; i++)
        buf[i] = value;

    uint32_t total_bytes = ALIGN8(num_elems * sizeof(int16_t));
    uint32_t off = 0;
    
    /* Write 64-byte chunks */
    for (; off + 64 <= total_bytes; off += 64)
    {
        mram_write(buf, (__mram_ptr void *)(mram_addr + off), 64);
    }
    
    /* Handle remainder with appropriately sized writes */
    if (off < total_bytes)
    {
        uint32_t remaining = total_bytes - off;
        uint32_t chunk = ALIGN8(remaining);
        if (chunk < 8) chunk = 8;
        if (chunk > 64) chunk = 64;
        mram_write(buf, (__mram_ptr void *)(mram_addr + off), chunk);
    }
}

#endif /* DPU_COMMON_H */