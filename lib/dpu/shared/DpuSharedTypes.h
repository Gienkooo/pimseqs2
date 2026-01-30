#ifndef DPU_SHARED_TYPES_H
#define DPU_SHARED_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define DPU_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define DPU_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* Alignment & Constants */
#define DPU_MRAM_ALIGN 8
#define DPU_ALIGN_SIZE(x) (((x) + (DPU_MRAM_ALIGN - 1)) & ~(DPU_MRAM_ALIGN - 1))
#define MAX_BATCH_QUERIES 128
#define MAX_BATCH_TARGETS 2048

/* Smith-Waterman tiling constants */
#ifndef Q_TILE_SIZE
#define Q_TILE_SIZE 32
#endif
#ifndef T_TILE_SIZE
#define T_TILE_SIZE 32
#endif

#define ALPHA_SIZE 21

/* Coverage Modes */
#define DPU_COV_MODE_BIDIRECTIONAL  0
#define DPU_COV_MODE_TARGET         1
#define DPU_COV_MODE_QUERY          2
#define DPU_COV_MODE_LENGTH_QUERY   3
#define DPU_COV_MODE_LENGTH_TARGET  4
#define DPU_COV_MODE_LENGTH_SHORTER 5

// Kernel contract: when adding new DPU kernels, follow these rules:
// - Include `dpu_common.h` for shared helpers/macros.
// - Use a descriptor struct whose first field is `DpuBatchHeader`.
// - Honor `header.num_active_tasklets` (use `is_tasklet_active()` in kernels).
// - Results MRAM layout: write a 32-bit hit count at results_offset (first 4 bytes)
//   and start hits at `results_offset + 8` (8-byte aligned header).

#define DPU_MRAM_TOTAL_SIZE (64 * 1024 * 1024) 

/* ==================== K-mer Matching Specific Limits and Structures ==================== 
 * MRAM Layout (Bucketed Hash Index):
 * [0x000000] Descriptor         (~96 B)   ← Batch metadata
 * [STATIC]   Checkpoint         (16 B)    ← Recovery state (packet_idx, valid flag)
 * [STATIC]   State Table        (32 KB)   ← Per-sequence diagonal tracking (8192 * 4B)
 * [STATIC]   Query Buffer       (12 MB)   ← Input packets, fixed size
 * [VARIABLE] Bucket Array       (16 MB+)  ← 65536 primary buckets (16MB) + overflow
 * [VARIABLE] Entries Array      (varies)  ← {target_id, pos} pairs
 * [VARIABLE] Output Buffer      (remaining)
 * ↳ [0x00] Result Header (8 B) ← Count + Overflow flag
 * ↳ [0x08] Double Hits...      ← Contiguous hit array
 */

/* Bucketed Index Parameters  */
#define NUM_BUCKETS 65536
#define BUCKET_SIZE 256
#define BUCKET_CAPACITY 20  /* (256 - 2 count - 2 pad - 4 next - 8 pad) / 12 bytes per item = 20 */

/* Sentinels */
#define CHAIN_END_IDX 0xFFFFFFFF

#define KMER_TARGET_ID_PADDING 0xFFFE

#define KMER_PACKET_SENTINEL    0xFFFFFFFF   /* End-of-query marker in packet stream */
#define KMER_RESULT_SENTINEL    0xFFFFFFFF   /* End-of-query marker in result stream */

/* Fixed MRAM Overhead Calculation */
#define DPU_INDEX_BUCKETS_SIZE   (NUM_BUCKETS * BUCKET_SIZE)    /* 16 MB */
#define DPU_STATE_TABLE_SIZE     (MAX_DPU_SEQS * 4)             /* 32 KB */
#define DPU_FIXED_INDEX_OVERHEAD (DPU_INDEX_BUCKETS_SIZE + DPU_STATE_TABLE_SIZE)
#define MAX_DPU_INDEX_SIZE       (44 * 1024 * 1024) 

#define MAX_DPU_SEQS 8192       

/* Buffer Size Configuration */
#define KMER_QUERY_BUFFER_SIZE      (16 * 1024 * 1024)
#define KMER_MIN_OUTPUT_BUFFER_SIZE (1024)

/* ==================== HASH CALCULATION ==================== */
/* MurmurHash3 Finalizer (fast integer hash)
 * Maps k-mer to Bucket Index [0, 65535] */
static inline uint32_t dpu_compute_hash(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (NUM_BUCKETS - 1);
}

/* ==================== BUCKET DATA STRUCTURES ==================== */

/* 12-byte Bucket Item: Key + Offset + Count */
typedef struct {
    uint32_t key;       /* K-mer value */
    uint32_t offset;    /* Start offset in entries array */
    uint16_t count;     /* Number of entries for this k-mer */
    uint16_t pad;       /* Alignment padding */
} __attribute__((packed)) KmerBucketItem;

DPU_STATIC_ASSERT(sizeof(KmerBucketItem) == 12, "KmerBucketItem must be 12 bytes");

/* 256-byte Bucket (Aligned for single MRAM transfer) */
typedef struct {
    uint16_t count;                      /* Number of items in this bucket */
    uint16_t pad1;                       /* Padding for alignment */
    uint32_t next_idx;                   /* Index of overflow bucket (0xFFFFFFFF if none) */
    KmerBucketItem items[BUCKET_CAPACITY]; /* 20 * 12 = 240 bytes */
    uint32_t padding[2];                 /* 8 bytes to reach 256 */
} __attribute__((packed)) KmerBucket;

DPU_STATIC_ASSERT(sizeof(KmerBucket) == 256, "KmerBucket must be 256 bytes");

/* Query Packet */
typedef struct {
    uint32_t kmer_idx;      /* Encoded k-mer index for hash lookup */
    uint16_t bucket_idx;    /* Pre-calculated bucket index (hash result) */
    uint16_t query_pos;     /* Position i in query sequence */
} __attribute__((packed)) KmerQueryPacket;

DPU_STATIC_ASSERT(sizeof(KmerQueryPacket) == 8, "KmerQueryPacket must be 8 bytes");

/* WRAM State Entry 
 * Tracks the last seen diagonal for a sequence. */
typedef struct {
    int16_t diag;       /* Last seen diagonal */
    uint16_t pos;       /* Last query position */
} __attribute__((packed)) KmerDiagonalStateEntry;

DPU_STATIC_ASSERT(sizeof(KmerDiagonalStateEntry) == 4, "KmerDiagonalStateEntry must be 4 bytes");

/* Output Hit - DPU to Host result */
typedef struct {
    uint32_t target_id;     /* Local target ID (0..MAX_DPU_SEQS-1) */
    int16_t diagonal;       /* Diagonal (i - j) */
    uint16_t padding;       /* Alignment padding */
} __attribute__((packed)) KmerDoubleHit;

DPU_STATIC_ASSERT(sizeof(KmerDoubleHit) == 8, "KmerDoubleHit must be 8 bytes");

/* MRAM Index Entry - Compact database index */
typedef struct {
    uint16_t local_target_id;  /* Local target ID within DPU */
    uint16_t pos_j;            /* Position j in target sequence */
} __attribute__((packed)) KmerIndexEntry;

DPU_STATIC_ASSERT(sizeof(KmerIndexEntry) == 4, "KmerIndexEntry must be 4 bytes");

/* Checkpoint Structure - For resuming after output buffer overflow */
typedef struct {
    uint32_t packet_idx;    /* Index of query packet being processed */
    uint32_t valid;         /* 1 if checkpoint active, 0 otherwise */     
} __attribute__((packed)) KmerCheckpoint;

DPU_STATIC_ASSERT(sizeof(KmerCheckpoint) == 8, "KmerCheckpoint must be 8 bytes");

/* Result Header - Stores actual double hit count */
typedef struct {
    uint32_t total_hits;    /* Actual number of double hits written */
    uint32_t overflow;      /* 1 if output buffer overflowed */
} __attribute__((packed)) KmerResultHeader;

DPU_STATIC_ASSERT(sizeof(KmerResultHeader) == 8, "KmerResultHeader must be 8 bytes");

/* K-mer Match Batch Descriptor */
typedef struct {
    uint32_t num_query_packets;     /* Number of query k-mer packets */
    uint32_t num_targets;           /* Number of target sequences */
    uint32_t num_buckets;           /* Total buckets (primary + overflow) */
    uint32_t num_index_entries;     /* Total index entries */
    
    /* MRAM Offsets */
    uint32_t checkpoint_offset;     /* Checkpoint for resuming after overflow */
    uint32_t state_table_offset;    /* State table for diagonal tracking (MRAM backing store) */
    uint32_t query_packets_offset;  /* Query packet buffer */
    uint32_t buckets_offset;        /* Start of bucket array */
    uint32_t index_entries_offset;  /* Start of entries array */
    uint32_t results_offset;        /* Start of results buffer */
    uint32_t results_buffer_size;   /* Actual output buffer size */
    
    uint32_t reserved[1];           /* Padding for 8-byte alignment */
} __attribute__((packed)) KmerBatchDescriptor;

DPU_STATIC_ASSERT(sizeof(KmerBatchDescriptor) % 8 == 0, "KmerBatchDescriptor must be 8-byte aligned");

/* Calculate total static usage */
#define DPU_MRAM_ESTIMATED_USAGE ( \
    sizeof(KmerBatchDescriptor) +  \
    sizeof(KmerCheckpoint) +       \
    sizeof(KmerResultHeader) +     \
    DPU_STATE_TABLE_SIZE +         \
    KMER_QUERY_BUFFER_SIZE +       \
    MAX_DPU_INDEX_SIZE +           \
    KMER_MIN_OUTPUT_BUFFER_SIZE    \
)

DPU_STATIC_ASSERT(
    DPU_MRAM_ESTIMATED_USAGE <= DPU_MRAM_TOTAL_SIZE, 
    "CRITICAL: DPU MRAM Usage exceeds 64MB! Reduce MAX_DPU_INDEX_SIZE or Buffers."
);

/* --- 1. Common Batch Header --- */
struct DpuBatchHeader {
    uint32_t batch_id;
    uint32_t num_queries;
    uint32_t num_targets;
    uint32_t query_len;
    
    /* Offsets */
    uint32_t queries_metadata_offset;
    uint32_t pssm_data_offset;      
    uint32_t targets_metadata_offset;
    uint32_t targets_data_offset;
    uint32_t results_offset;
    
    /* Sizes & Limits */
    uint32_t pssm_total_size;
    uint32_t targets_total_size;
    uint32_t results_buffer_size;
    
    uint16_t flags;
    uint8_t  num_active_tasklets; /* Dynamic tasklet control */
    uint8_t  pad[5];              /* pad to 8 bytes */

#ifdef __cplusplus
    DpuBatchHeader() = default;
    DpuBatchHeader(uint32_t num_queries, uint32_t num_targets, uint32_t query_len,
                   uint32_t queries_metadata_offset, uint32_t pssm_data_offset,
                   uint32_t targets_metadata_offset, uint32_t targets_data_offset,
                   uint32_t results_offset, uint32_t results_buffer_size,
                   uint32_t pssm_total_size, uint32_t targets_total_size,
                   uint16_t flags, uint8_t num_active_tasklets)
        : num_queries(num_queries), num_targets(num_targets), query_len(query_len),
          queries_metadata_offset(queries_metadata_offset), pssm_data_offset(pssm_data_offset),
          targets_metadata_offset(targets_metadata_offset), targets_data_offset(targets_data_offset),
          results_offset(results_offset),
          pssm_total_size(pssm_total_size), targets_total_size(targets_total_size),
          results_buffer_size(results_buffer_size),
          flags(flags), num_active_tasklets(num_active_tasklets) {
        for (int i = 0; i < 5; ++i) pad[i] = 0;
    }
#endif
} __attribute__((packed));
typedef struct DpuBatchHeader DpuBatchHeader;

/* Split contexts to decouple target (static) and query (per-batch) configuration. */
typedef struct {
    uint32_t num_targets;
    uint32_t target_meta_offset;   /* Offset to array of TargetMetadata */
    uint32_t target_data_offset;   /* Offset to packed sequence data */
    uint32_t results_offset;       /* Offset to write results */
    uint32_t results_buffer_size;  /* Size of result buffer */
    uint32_t pad[3];               /* Align to 8 bytes */
} __attribute__((packed)) TargetContext;

/* Query/launch-specific context broadcast once per batch. */
typedef struct {
    uint32_t num_queries;
    uint32_t max_query_len;        /* Maximum query length in batch */
    uint32_t queries_metadata_offset;
    uint32_t pssm_data_offset;
    uint32_t num_active_tasklets;
    int16_t  min_ungapped_score;
    int16_t  min_score;
    int16_t  gap_open_cost;
    int16_t  gap_extend_cost;
    uint8_t  cov_mode;
    uint8_t  cov_thr_pct;
    uint8_t  min_aln_len;
    uint8_t  seq_id_thr_pct;
    uint8_t  flags;                /* Bit 0: force_gapped */
    uint8_t  pad[7];               /* pad to 40 bytes (8-byte aligned) */
} __attribute__((packed)) QueryContext;

/* Fixed MRAM offsets for contexts */
#define TARGET_CTX_OFFSET 0
#define QUERY_CTX_OFFSET  DPU_ALIGN_SIZE(sizeof(TargetContext))

/* Ungapped descriptor */
struct UngappedBatchDescriptor {
    DpuBatchHeader header;
    
    int16_t min_score;
    int16_t gap_open_cost;      // Not used by ungapped, but kept for struct similarity if needed
    int16_t gap_extend_cost;
    int16_t pssm_bias;

#ifdef __cplusplus
    UngappedBatchDescriptor() = default;
    UngappedBatchDescriptor(const DpuBatchHeader& header, int16_t min_score,
                            int16_t gap_open_cost, int16_t gap_extend_cost, int16_t pssm_bias)
        : header(header), min_score(min_score), gap_open_cost(gap_open_cost),
          gap_extend_cost(gap_extend_cost), pssm_bias(pssm_bias) {}
#endif
} __attribute__((packed));
typedef struct UngappedBatchDescriptor UngappedBatchDescriptor;

/* Gapped descriptor */
struct GappedBatchDescriptor {
    DpuBatchHeader header;
    
    int16_t min_score;
    int16_t gap_open_cost;
    int16_t gap_extend_cost;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;
    
    uint8_t padding[2];

#ifdef __cplusplus
    GappedBatchDescriptor() = default;
    GappedBatchDescriptor(const DpuBatchHeader& header,
                          int16_t gap_open_cost, int16_t gap_extend_cost,
                          uint8_t cov_mode, uint8_t cov_thr_pct,
                          uint8_t min_aln_len, uint8_t seq_id_thr_pct)
        : header(header), min_score(min_score), gap_open_cost(gap_open_cost),
          gap_extend_cost(gap_extend_cost),
          cov_mode(cov_mode), cov_thr_pct(cov_thr_pct),
          min_aln_len(min_aln_len), seq_id_thr_pct(seq_id_thr_pct) {
        padding[0] = 0; padding[1] = 0;
    }
#endif
} __attribute__((packed));
typedef struct GappedBatchDescriptor GappedBatchDescriptor;

/* Combined (ungapped+gapped) descriptor */
struct CombinedBatchDescriptor {
    DpuBatchHeader header;
    
    int16_t min_ungapped_score;
    int16_t gap_open_cost;
    int16_t gap_extend_cost;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;

#ifdef __cplusplus
    CombinedBatchDescriptor() = default;
    CombinedBatchDescriptor(const DpuBatchHeader& header, int16_t min_ungapped_score,
                            int16_t gap_open_cost, int16_t gap_extend_cost,
                            uint8_t cov_mode,
                            uint8_t cov_thr_pct, uint8_t min_aln_len, uint8_t seq_id_thr_pct)
        : header(header), min_ungapped_score(min_ungapped_score), 
          gap_open_cost(gap_open_cost), gap_extend_cost(gap_extend_cost),
          cov_mode(cov_mode),
          cov_thr_pct(cov_thr_pct), min_aln_len(min_aln_len), seq_id_thr_pct(seq_id_thr_pct) {}
#endif
} __attribute__((packed));
typedef struct CombinedBatchDescriptor CombinedBatchDescriptor;


/* Data Structures */

typedef struct {
    uint32_t query_id;
    uint32_t query_len;
    uint32_t pssm_offset_in_batch;
    uint8_t  bias;
    uint8_t  padding_byte;
    int16_t  min_score;
} __attribute__((packed)) QueryMetadata;

DPU_STATIC_ASSERT(sizeof(QueryMetadata) == 16, "QueryMetadata must be 16 bytes");

typedef struct {
    uint32_t target_id;
    uint32_t target_len;
    uint32_t offset_in_data;
    uint32_t padding;
} __attribute__((packed)) TargetMetadata;

DPU_STATIC_ASSERT(sizeof(TargetMetadata) == 16, "TargetMetadata must be 16 bytes");

typedef struct {
    uint32_t kmer;
    uint16_t query_id;
    uint16_t query_pos;
} __attribute__((packed)) KmerEntry;

DPU_STATIC_ASSERT(sizeof(KmerEntry) == 8, "KmerEntry must be 8 bytes");
typedef struct {
    uint32_t target_id;
    uint16_t query_id;
    int16_t score;
    int16_t diagonal;
    uint32_t pad1;
    uint16_t pad2;
} __attribute__((packed)) Hit;

DPU_STATIC_ASSERT(sizeof(Hit) == 16, "Hit must be 16 bytes");

typedef struct {
    uint32_t target_id;
    int32_t score;
    uint16_t q_end;
    uint16_t t_end;
    uint16_t padding[2];
} __attribute__((packed)) GappedHit;

DPU_STATIC_ASSERT(sizeof(GappedHit) == 16, "GappedHit must be 16 bytes");

#define DPU_FLAG_FORCE_GAPPED   (1U << 0)

#ifdef __cplusplus
}
#endif

#endif /* DPU_SHARED_TYPES_H */