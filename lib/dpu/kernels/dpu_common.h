#ifndef DPU_COMMON_H
#define DPU_COMMON_H

#include <stdint.h>
#include <defs.h>
#include "../shared/DpuSharedTypes.h"

/* --- Macros & Constants --- */

#define ALIGN8(x) (((x) + 7) & ~7U)
#define MRAM_ALIGN_SIZE(x) ALIGN8(x)

#define MAX_MRAM_TRANSFER_SIZE 2048

#define ALPHA_SIZE 21
#define NEG_INF (-10000) // Safe value for int16 math

/* Compile-time shift-based multiplication helpers */
#define MUL_10(x)   (((x) << 3) + ((x) << 1))
#define MUL_21(x)   (((x) << 4) + ((x) << 2) + (x))
#define MUL_100(x)  (((x) << 6) + ((x) << 5) + ((x) << 2))
#define MUL_1656(x) (((x) << 10) + ((x) << 9) + ((x) << 7) - ((x) << 3))

/* --- Utilities --- */

static inline int16_t max2(int16_t a, int16_t b) { return (a > b) ? a : b; }
static inline int16_t max3(int16_t a, int16_t b, int16_t c) { return max2(a, max2(b, c)); }
static inline uint32_t min_u32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline uint32_t max_u32(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

/* Tasklet activation helper: 0 means "all tasklets active" */
static inline int is_tasklet_active(uint8_t limit) {
    if (limit == 0) return 1;
    return me() < limit;
}

static inline int16_t sat_sub(int16_t a, int16_t b) { return (a <= NEG_INF + b) ? NEG_INF : a - b; }
static inline int16_t sat_add(int16_t a, int16_t b) {
    int32_t res = (int32_t)a + (int32_t)b;
    if (res < NEG_INF) return NEG_INF;
    if (res > 32767) return 32767;
    return (int16_t)res;
}

/**
 * Safe MRAM write for transfers larger than 2KB.
 * PRECONDITIONS:
 * - wram_src must be 8-byte aligned
 * - mram_dst must be 8-byte aligned  
 * - size must be multiple of 8
 */
static void mram_write_safe(const void *wram_src, __mram_ptr void *mram_dst, uint32_t size) {
    uint32_t offset = 0;
    const uint8_t *src_ptr = (const uint8_t *)wram_src;
    __mram_ptr uint8_t *dst_ptr = (__mram_ptr uint8_t *)mram_dst;

    while (offset < size) {
        uint32_t chunk = (size - offset > MAX_MRAM_TRANSFER_SIZE) ? MAX_MRAM_TRANSFER_SIZE : (size - offset);
        mram_write(&src_ptr[offset], &dst_ptr[offset], chunk);
        offset += chunk;
    }
}

/**
 * Safe MRAM read for transfers larger than 2KB.
 * PRECONDITIONS:
 * - mram_src must be 8-byte aligned
 * - wram_dst must be 8-byte aligned
 * - size must be multiple of 8
 */
static void mram_read_safe(__mram_ptr const void *mram_src, void *wram_dst, uint32_t size) {
    uint32_t offset = 0;
    __mram_ptr const uint8_t *src_ptr = (__mram_ptr const uint8_t *)mram_src;
    uint8_t *dst_ptr = (uint8_t *)wram_dst;

    while (offset < size) {
        uint32_t chunk = (size - offset > MAX_MRAM_TRANSFER_SIZE) ? MAX_MRAM_TRANSFER_SIZE : (size - offset);
        mram_read(&src_ptr[offset], &dst_ptr[offset], chunk);
        offset += chunk;
    }
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

static inline int passes_seq_id_threshold(int16_t score, uint16_t q_end, uint16_t t_end, uint8_t seq_id_thr_pct) {
    if (seq_id_thr_pct == 0) return 1;
    if (score <= 0) return 0;
    uint32_t aln_len = max_u32(q_end, t_end);
    if (aln_len == 0) return 0;
    
    /* seqId = (score/aln_len)*0.1656 + 0.1141 */
    int32_t lhs = MUL_1656((int32_t)score);
    int32_t rhs_factor = (int32_t)MUL_100((uint32_t)seq_id_thr_pct) - 1141; // Using *100 for % scale
    if (rhs_factor <= 0) return 1;
    int32_t rhs = (int32_t)aln_len * rhs_factor;
    return (lhs >= rhs);
}

#endif /* DPU_COMMON_H */