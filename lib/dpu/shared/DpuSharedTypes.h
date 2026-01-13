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
 * MRAM Layout (default):
 *   [0x000000] Descriptor         (~96 B)
 *   [STATIC]   Result Header      (8 B)     ← Host polls this at fixed address
 *   [STATIC]   Checkpoint         (16 B)    ← Resume state on overflow
 *   [STATIC]   Hint Table         (~1.6 KB) ← 400-entry prefix lookup
 *   [STATIC]   State Table        (32 KB)   ← Per-sequence diagonal tracking
 *   [STATIC]   Query Buffer       (1 MB)    ← Input packets, fixed size
 *   [VARIABLE] Index (Keys/Off/Ent) (varies per chunk)
 *   [VARIABLE] Output Buffer       (remaining MRAM) 
 */

#define HINT_TABLE_SIZE 400     /* 20x20 AA prefix combinations (base-20 encoding) */
#define KMER_TARGET_ID_PADDING 0xFFFF

// The following can be adjusted in order to optimize performance TODO
/* Max number of sequences per DPU database chunk for kmer prefiltering 
   has to fit in WRAM comfortably (8192 fits in 32KB state table with 4-byte entries) */
#define MAX_DPU_SEQS 8192       
#define MAX_DPU_INDEX_SIZE (16 * 1024 * 1024) /* 16 MB max index size per DPU (keys + offsets + entries) */
#define BLOCK_SEARCH_SIZE 16    /* Fetch 16 keys (64 bytes) per MRAM access */

/* Buffer Size Configuration */
#define KMER_QUERY_BUFFER_SIZE (5 * 1024 * 1024)     /* query packet buffer */
#define KMER_MIN_OUTPUT_BUFFER_SIZE (1024 * 1024)    /* 1 MB = 128 hits * 1024  */

#define MAX_QUERY_PACKETS_PER_LAUNCH (KMER_QUERY_BUFFER_SIZE / sizeof(KmerQueryPacket))  /* 131072 packets for 5MB buffer */

/* Query Packet */
typedef struct {
    uint32_t kmer_idx;      /* Encoded k-mer index for binary search */
    uint16_t hint_idx;      /* Pre-calculated hint index */
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
} __attribute__((packed)) KmerCompactIndexEntry;

DPU_STATIC_ASSERT(sizeof(KmerCompactIndexEntry) == 4, "KmerCompactIndexEntry must be 4 bytes");

/* Checkpoint Structure - For resuming after output buffer overflow */
typedef struct {
    uint32_t packet_idx;    /* Index of query packet being processed */
    uint32_t entry_idx;     /* Offset in entries list */
    uint32_t key_idx;       /* Cached binary search result */
    uint32_t valid;         /* 1 if checkpoint active, 0 otherwise */
} __attribute__((packed)) KmerCheckpoint;

DPU_STATIC_ASSERT(sizeof(KmerCheckpoint) == 16, "KmerCheckpoint must be 16 bytes");

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
    uint32_t num_index_keys;        /* Number of unique k-mers in index */
    uint32_t num_index_entries;     /* Total index entries */
    
    /* MRAM Offsets */
    uint32_t hint_table_offset;     
    uint32_t query_packets_offset;
    uint32_t index_keys_offset;
    uint32_t index_offsets_offset;
    uint32_t index_entries_offset;
    uint32_t state_table_offset;    /* MRAM backup of state table for resuming after an overflow */
    uint32_t checkpoint_offset;     /* Checkpoint for resuming after an overflow structure offset */
    uint32_t results_header_offset; 
    uint32_t results_offset;        
    uint32_t results_buffer_size;   /* Actual output buffer size (max(KMER_MIN_OUTPUT_BUFFER_SIZE, remaining_mram)) */
    
    uint32_t packet_start_idx;      /* Start index for this batch (for resume) */
    uint32_t reserved1;             /* Align to 8 bytes */
} __attribute__((packed)) KmerBatchDescriptor;

DPU_STATIC_ASSERT(sizeof(KmerBatchDescriptor) % 8 == 0, "KmerBatchDescriptor must be 8-byte aligned");

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
    int16_t xdrop_threshold;
    int16_t pssm_bias;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;
    
    uint8_t padding[2];

#ifdef __cplusplus
    GappedBatchDescriptor() = default;
    GappedBatchDescriptor(const DpuBatchHeader& header, int16_t min_score,
                          int16_t gap_open_cost, int16_t gap_extend_cost, int16_t xdrop_threshold,
                          int16_t pssm_bias, uint8_t cov_mode, uint8_t cov_thr_pct,
                          uint8_t min_aln_len, uint8_t seq_id_thr_pct)
        : header(header), min_score(min_score), gap_open_cost(gap_open_cost),
          gap_extend_cost(gap_extend_cost), xdrop_threshold(xdrop_threshold),
          pssm_bias(pssm_bias), cov_mode(cov_mode), cov_thr_pct(cov_thr_pct),
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
    int16_t min_score;          // Gapped score threshold
    int16_t gap_open_cost;
    int16_t gap_extend_cost;
    int16_t xdrop_threshold;
    int16_t pssm_bias;
    
    uint8_t cov_mode;
    uint8_t cov_thr_pct;
    uint8_t min_aln_len;
    uint8_t seq_id_thr_pct;

#ifdef __cplusplus
    CombinedBatchDescriptor() = default;
    CombinedBatchDescriptor(const DpuBatchHeader& header, int16_t min_ungapped_score,
                            int16_t min_score, int16_t gap_open_cost, int16_t gap_extend_cost,
                            int16_t xdrop_threshold, int16_t pssm_bias, uint8_t cov_mode,
                            uint8_t cov_thr_pct, uint8_t min_aln_len, uint8_t seq_id_thr_pct)
        : header(header), min_ungapped_score(min_ungapped_score), min_score(min_score),
          gap_open_cost(gap_open_cost), gap_extend_cost(gap_extend_cost),
          xdrop_threshold(xdrop_threshold), pssm_bias(pssm_bias), cov_mode(cov_mode),
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
    uint8_t  pad[3];
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
    int16_t score;
    uint16_t q_end;
    uint16_t t_end;
    uint16_t padding[3];
} __attribute__((packed)) GappedHit;

DPU_STATIC_ASSERT(sizeof(GappedHit) == 16, "GappedHit must be 16 bytes");

#define DPU_FLAG_FORCE_GAPPED   (1U << 0)

#ifdef __cplusplus
}
#endif

#endif /* DPU_SHARED_TYPES_H */