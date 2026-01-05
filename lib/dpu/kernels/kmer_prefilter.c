// does not follow the kernel contract yet
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <defs.h>
#include <barrier.h>

#include "DpuSharedTypes.h"

// Define DEBUG_MODE to enable logging. Can be defined here or via compiler flag -DDEBUG_MODE
// #define DEBUG_MODE

#ifdef DEBUG_MODE
    #define DPU_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DPU_LOG(fmt, ...) ((void)0)
#endif

/**
 * DPU K-mer Prefilter Kernel
 * 
 * This kernel performs k-mer based sequence filtering with checkpoint-based
 * overflow handling for streaming large result sets. Uses WRAM-resident
 * structures for fast access and MRAM for bulk storage.
 * 
 * Memory Architecture:
 * - WRAM: Cache structures for hint table, keys, entries, and result batching
 * - MRAM: Persistent storage for index, query packets, and output buffer
 * 
 * Overflow Strategy:
 * - Batches results in WRAM (128 hits) before writing to MRAM
 * - Per-packet flushing ensures checkpoint resume points are calculable
 * - Checkpoint stores packet_idx, entry_idx, key_idx for stateful resumption
 */

#define NR_TASKLETS 1
#define MAX_WRAM_TRANSFER_SIZE 2048

/* WRAM Allocations - Total usage ~36KB
 * All structures are statically allocated to avoid stack overflow issues.
 * WRAM provides fast access but is limited, so we use it for hot data only.
 * 
 * Budget breakdown:
 *   - State table: 32KB (MAX_DPU_SEQS * 4 bytes)
 *   - Hint table: ~1.6KB (401 * 4 bytes)
 *   - Key cache: 512 bytes (128 * 4 bytes)
 *   - Result batch: 1KB (128 * 8 bytes)
 *   - Entry cache: 128 bytes (32 * 4 bytes)
 *   - Packet cache: 64 bytes (8 * 8 bytes)
 *   - Other: ~200 bytes
 */

__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;

// Hint table: HINT_TABLE_SIZE + 1 elements needed for data
// MRAM read operations align to 8-byte boundaries, requiring padding
// Buffer size must be >= ((HINT_TABLE_SIZE + 1) * 4 + 7) & ~7 bytes
// With HINT_TABLE_SIZE=400: need 1608 bytes, so 402 uint32_t minimum
__host __attribute__((aligned(8))) uint32_t w_hint_table[HINT_TABLE_SIZE + 2];

// Compile-time check: ensure buffer is large enough for aligned MRAM read
_Static_assert(sizeof(w_hint_table) >= (((HINT_TABLE_SIZE + 1) * sizeof(uint32_t) + 7) & ~7),
               "w_hint_table buffer too small for 8-byte aligned MRAM read");

// State Table: 32KB - tracks double-hit state for each target sequence
// Entry: {last_diagonal, last_query_pos} - used for consecutive hit detection
// Initialization: 0xFFFF for both fields means "no previous hit"
static __attribute__((aligned(8))) KmerDiagonalStateEntry w_state_table[MAX_DPU_SEQS];

// Key cache for binary search - 128 keys = 512 bytes
#define KEY_CACHE_SIZE 128
static __attribute__((aligned(8))) uint32_t w_key_cache[KEY_CACHE_SIZE];

// Result batch buffer - 128 results = 1024 bytes  
#define RESULT_BATCH_SIZE 128
static __attribute__((aligned(8))) KmerDoubleHit w_result_batch[RESULT_BATCH_SIZE];

// Packet cache for reading - 8 packets = 64 bytes
#define PACKET_CACHE_SIZE 8
static __attribute__((aligned(8))) KmerQueryPacket w_packet_cache[PACKET_CACHE_SIZE];

// Entry cache for processing index entries
// MRAM reads must be 8-byte aligned, but KmerCompactIndexEntry is 4 bytes.
// We may need to read 1 extra entry at the start for alignment, so cache size = logical + 1
#define ENTRY_CACHE_LOGICAL_SIZE 32
#define ENTRY_CACHE_SIZE (ENTRY_CACHE_LOGICAL_SIZE + 2)  // +2 for alignment padding
static __attribute__((aligned(8))) KmerCompactIndexEntry w_entry_cache[ENTRY_CACHE_SIZE];

// Offset read buffer - 2 uint32_t = 8 bytes (for aligned read)
static __attribute__((aligned(8))) uint32_t w_offset_buf[2];

// Barriers are not required for single-tasklet execution configuration. But would be defined here if needed.

__mram_ptr uint8_t* mram_base = (__mram_ptr uint8_t*)DPU_MRAM_HEAP_POINTER;

/* Dynamic MRAM pointers */
static __mram_ptr uint32_t* m_keys = (__mram_ptr uint32_t*)0;
static __mram_ptr uint32_t* m_offsets = (__mram_ptr uint32_t*)0;
static __mram_ptr KmerCompactIndexEntry* m_entries = (__mram_ptr KmerCompactIndexEntry*)0;

static void safe_mram_read(__mram_ptr const void* src, void* dst, uint32_t size) {
    uint32_t offset = 0;
    while (offset < size) {
        uint32_t chunk = (size - offset > MAX_WRAM_TRANSFER_SIZE) ? MAX_WRAM_TRANSFER_SIZE : (size - offset);
        mram_read((__mram_ptr const uint8_t*)src + offset, (uint8_t*)dst + offset, chunk);
        offset += chunk;
    }
}

static void safe_mram_write(const void* src, __mram_ptr void* dst, uint32_t size) {
    uint32_t offset = 0;
    while (offset < size) {
        uint32_t chunk = (size - offset > MAX_WRAM_TRANSFER_SIZE) ? MAX_WRAM_TRANSFER_SIZE : (size - offset);
        mram_write((const uint8_t*)src + offset, (__mram_ptr uint8_t*)dst + offset, chunk);
        offset += chunk;
    }
}

/**
 * K-mer Binary Search with Offset Retrieval
 * 
 * Performs a two-phase search:
 * 1. Hint table lookup to narrow the search range
 * 2. Blocked binary search within the narrowed range
 * 
 * @param kmer_idx The k-mer value to search for
 * @param hint_idx Pre-computed hint index for this k-mer
 * @param out_offset_start Output: starting offset in entries array
 * @param out_offset_end Output: ending offset in entries array
 * @return Index of found key (>=0), or -1 if not found
 */
static int32_t search_kmer_with_offset(uint32_t kmer_idx, uint16_t hint_idx, 
                                        uint32_t* out_offset_start, uint32_t* out_offset_end) {
    // Phase A: Validate hint and get initial search range
    if (hint_idx >= HINT_TABLE_SIZE) {
        return -1;
    }
    
    uint32_t low = w_hint_table[hint_idx];
    uint32_t high = w_hint_table[hint_idx + 1];
    
    // Empty range check
    if (low >= high) {
        return -1;
    }
    
    // Bounds safety
    if (low >= g_descriptor.num_index_keys) {
        return -1;
    }
    if (high > g_descriptor.num_index_keys) {
        high = g_descriptor.num_index_keys;
    }
    
    // MRAM pointers are initialized at runtime in main() from g_descriptor
    // Use the global aliases: m_keys, m_offsets    
    // Phase B: Blocked Binary Search
    while (low < high) {
        uint32_t mid = low + ((high - low) >> 1);
        
        // Align to KEY_CACHE_SIZE boundary for efficient MRAM access
        uint32_t block_start = mid & ~(KEY_CACHE_SIZE - 1);
        
        // Clamp to valid range
        if (block_start >= g_descriptor.num_index_keys) {
            high = mid;
            continue;
        }
        
        // Calculate how many keys to fetch (may be partial block at end)
        uint32_t block_end = block_start + KEY_CACHE_SIZE;
        if (block_end > g_descriptor.num_index_keys) {
            block_end = g_descriptor.num_index_keys;
        }
        uint32_t fetch_count = block_end - block_start;
        
        // uint32_t fetch_bytes = (fetch_count * sizeof(uint32_t) + 7) & ~7;
        uint32_t fetch_bytes = ((fetch_count << 2) + 7) & ~7;
        mram_read(&m_keys[block_start], w_key_cache, fetch_bytes);
        
        // Phase C: Check if target is within this block's range
        uint32_t block_min = w_key_cache[0];
        uint32_t block_max = w_key_cache[fetch_count - 1];
        
        if (kmer_idx >= block_min && kmer_idx <= block_max) {
            // Target MUST be in this block if it exists (sorted array)
            // Linear scan within WRAM cache
            for (uint32_t i = 0; i < fetch_count; ++i) {
                if (w_key_cache[i] == kmer_idx) {
                    // FOUND! Now fetch the offset value
                    uint32_t found_idx = block_start + i;
                    
                    // Retrieve offset pair for this k-mer from MRAM
                    // MRAM requires 8-byte aligned reads, so we align to uint32_t pairs
                    uint32_t align_idx = found_idx & ~1;
                    
                    // Handle odd vs even indices for proper offset extraction
                    // The offsets array has size (num_keys + 1) to provide [start, end) pairs
                    if (found_idx & 1) {
                        // Odd index: Our offset is in the second position of the aligned pair
                        if (found_idx >= g_descriptor.num_index_keys) {
                            *out_offset_start = 0;
                            *out_offset_end = 0;
                            return -1;
                        }
                        
                        // Read aligned pair and extract start offset from second position
                        mram_read(&m_offsets[align_idx], w_offset_buf, 8);
                        *out_offset_start = w_offset_buf[1];
                        
                        // Read end offset from next pair, checking array bounds
                        if (align_idx + 2 > g_descriptor.num_index_keys) {
                            // Last valid position - read the sentinel with proper alignment
                            // Ensure alignment when reading the sentinel, as num_index_keys may be odd
                            uint32_t sent_align = g_descriptor.num_index_keys & ~1;
                            mram_read(&m_offsets[sent_align], w_offset_buf, 8);
                            *out_offset_end = (g_descriptor.num_index_keys & 1) ? w_offset_buf[1] : w_offset_buf[0];
                        } else {
                            // Read next aligned pair
                            mram_read(&m_offsets[align_idx + 2], w_offset_buf, 8);
                            *out_offset_end = w_offset_buf[0];
                        }
                    } else {
                        // Even index: both offsets available in one aligned read
                        mram_read(&m_offsets[align_idx], w_offset_buf, 8);
                        *out_offset_start = w_offset_buf[0];  // offsets[found_idx]
                        *out_offset_end = w_offset_buf[1];    // offsets[found_idx+1]
                    }
                    
                    return found_idx;
                }
            }
            // In range but not found - k-mer doesn't exist
            return -1;
        }
        
        // Binary search pivot
        if (kmer_idx < block_min) {
            high = block_start;
        } else {
            low = block_start + KEY_CACHE_SIZE;
        }
    }
    
    return -1; 
}

int main() {
    // Ensure execution on the main tasklet only
    if (me() != 0) {
        return 0;
    }
    
    // Load descriptor
    mram_read(mram_base, &g_descriptor, sizeof(KmerBatchDescriptor));
    
    // Load hint table into WRAM (permanent resident)
    __mram_ptr uint32_t* hint_table_mram = (__mram_ptr uint32_t*)(mram_base + g_descriptor.hint_table_offset);
    uint32_t hint_bytes = (HINT_TABLE_SIZE + 1) * sizeof(uint32_t);
    uint32_t hint_bytes_aligned = (hint_bytes + 7) & ~7;
    mram_read(hint_table_mram, w_hint_table, hint_bytes_aligned);
    
    // === PRINT DESCRIPTOR VALUES FOR VERIFICATION ===
    DPU_LOG("\n========================================\n");
    DPU_LOG("[DPU T0] RECEIVED BATCH DESCRIPTOR:\n");
    DPU_LOG("========================================\n");
    DPU_LOG("  num_query_packets:     %u\n", g_descriptor.num_query_packets);
    DPU_LOG("  num_targets:           %u\n", g_descriptor.num_targets);
    DPU_LOG("  num_index_keys:        %u\n", g_descriptor.num_index_keys);
    DPU_LOG("  num_index_entries:     %u\n", g_descriptor.num_index_entries);
    DPU_LOG("\n  MRAM Offsets:\n");
    DPU_LOG("    hint_table_offset:     %u\n", g_descriptor.hint_table_offset);
    DPU_LOG("    query_packets_offset:  %u\n", g_descriptor.query_packets_offset);
    DPU_LOG("    index_keys_offset:     %u\n", g_descriptor.index_keys_offset);
    DPU_LOG("    index_offsets_offset:  %u\n", g_descriptor.index_offsets_offset);
    DPU_LOG("    index_entries_offset:  %u\n", g_descriptor.index_entries_offset);
    DPU_LOG("    state_table_offset:    %u\n", g_descriptor.state_table_offset);
    DPU_LOG("    checkpoint_offset:     %u\n", g_descriptor.checkpoint_offset);
    DPU_LOG("    results_header_offset: %u\n", g_descriptor.results_header_offset);
    DPU_LOG("    results_offset:        %u\n", g_descriptor.results_offset);
    DPU_LOG("\n  Buffer Sizes:\n");
    DPU_LOG("    results_buffer_size:   %u\n", g_descriptor.results_buffer_size);
    DPU_LOG("    packet_start_idx:      %u\n", g_descriptor.packet_start_idx);
    DPU_LOG("========================================\n\n");
    
    // ===== SETUP MRAM POINTERS =====
    __mram_ptr KmerQueryPacket* query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
    __mram_ptr KmerDoubleHit* output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset);
    __mram_ptr KmerResultHeader* result_header_ptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_header_offset);
    __mram_ptr KmerCheckpoint* checkpoint_ptr = (__mram_ptr KmerCheckpoint*)(mram_base + g_descriptor.checkpoint_offset);
    __mram_ptr KmerCompactIndexEntry* index_entries = (__mram_ptr KmerCompactIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
    __mram_ptr KmerDiagonalStateEntry* state_table_mram = (__mram_ptr KmerDiagonalStateEntry*)(mram_base + g_descriptor.state_table_offset);

    /* Initialize runtime MRAM pointer aliases from descriptor to support dynamic chunk layouts */
    m_keys = (__mram_ptr uint32_t*)(mram_base + g_descriptor.index_keys_offset);
    m_offsets = (__mram_ptr uint32_t*)(mram_base + g_descriptor.index_offsets_offset);
    m_entries = index_entries; 

    // Load state table from MRAM into WRAM (32KB)
    // Host initializes to 0xFF (invalid) at start of each query
    // State persists across overflow/resume cycles within a query

    // [ << 2 ] == [ * sizeof(KmerDiagonalStateEntry) ]
    uint32_t state_bytes = MAX_DPU_SEQS << 2;
    
    // Read state table safely in chunks
    safe_mram_read(state_table_mram, w_state_table, state_bytes);

    // [ >> 3 ] == [ / sizeof(KmerDoubleHit) ]
    uint32_t max_results = g_descriptor.results_buffer_size >> 3;
    uint32_t total_packets = g_descriptor.num_query_packets;
    
    DPU_LOG("[DPU T0] Max output hits: %u, Total packets: %u\n", max_results, total_packets);
    DPU_LOG("[DPU T0] State table initialized: %u entries (%u bytes)\n", MAX_DPU_SEQS, state_bytes);
    
    // ===== HIT EMISSION STATE =====
    uint32_t total_hits_written = 0;  // Total double-hits written to MRAM
    uint32_t batch_count = 0;          // Current fill level of w_result_batch
    uint32_t single_hits_count = 0;    // Count of single hits (for stats)
    uint32_t double_hits_count = 0;    // Count of double hits detected (for stats)
    bool overflow_occurred = false;

    // Check for checkpoint - resuming after output buffer overflow
    KmerCheckpoint checkpoint;
    mram_read(checkpoint_ptr, &checkpoint, sizeof(KmerCheckpoint));
    
    uint32_t start_packet = 0;
    uint32_t start_entry = 0;
    int32_t cached_key_idx = -1;
    
    if (checkpoint.valid == 1) {
        start_packet = checkpoint.packet_idx;
        
        // Check for Spillover Resume using the entry index as an indicator
        // This happens when we overflowed at a sentinel (query boundary) 
        //with pending hits that didn't fit
        if (checkpoint.entry_idx == 1) {
            // Load spilled hits from state table MRAM backup area
            uint32_t spill_count = checkpoint.key_idx;
            DPU_LOG("[DPU T0] Resuming with %u spilled hits from state table\n", spill_count);
            
            if (spill_count > 0 && spill_count <= RESULT_BATCH_SIZE) {
                mram_read(state_table_mram, w_result_batch, spill_count << 3);
                batch_count = spill_count;
            }

            start_entry = 0;
            cached_key_idx = -1;
        } else {
            // Normal resume from mid-packet processing
            start_entry = checkpoint.entry_idx;
            cached_key_idx = (int32_t)checkpoint.key_idx;
            DPU_LOG("[DPU T0] Resuming from checkpoint: packet=%u entry=%u key_idx=%d\n", 
                   start_packet, start_entry, cached_key_idx);
        }
    } else {
        DPU_LOG("[DPU T0] Starting from beginning (no checkpoint)\n");
    }
    
    // ===== MAIN PACKET PROCESSING LOOP =====
    for (uint32_t pkt_idx = start_packet; pkt_idx < total_packets; ++pkt_idx) {
        // Read packet from MRAM
        mram_read(&query_packets[pkt_idx], w_packet_cache, 8);
        uint32_t kmer_idx = w_packet_cache[0].kmer_idx;
        uint16_t hint_idx = w_packet_cache[0].hint_idx;
        uint16_t query_pos = w_packet_cache[0].query_pos;
        
        // ===== NEW QUERY DETECTION =====
        // If kmer_idx == KMER_PACKET_SENTINEL_KEY, this marks the end of a query
        // 1) Flush pending hits with spillover protection, 2) Write result delimiter, 3) Reset state table
        if (kmer_idx == KMER_PACKET_SENTINEL_KEY) {
            DPU_LOG("[DPU T0] SENTINEL detected at packet %u - query boundary\n", pkt_idx);
            
            // 1. Flush Pending Hits with Spillover Protection
            if (batch_count > 0) {
                // Check for Overflow
                if (total_hits_written + batch_count > max_results) {
                    DPU_LOG("[DPU T0] OVERFLOW at sentinel flush: batch_count=%u, space=%u\n", 
                           batch_count, max_results - total_hits_written);
                    
                    uint32_t can_write = max_results - total_hits_written;
                    uint32_t spill_count = batch_count - can_write;
                    
                    // A. Write what fits to Output Buffer
                    if (can_write > 0) {
                        mram_write(w_result_batch, &output_buffer[total_hits_written], can_write << 3);
                        total_hits_written += can_write;
                    }

                    // B. SPILL remaining hits to State Table MRAM (Backup)
                    // The state table backup can be safely overwritten because it is no longer needed
                    // because the state is about to be reset anyway. The state table MRAM area is 32KB,
                    // more than enough to hold RESULT_BATCH_SIZE (128) hits (1KB max).
                    if (spill_count > 0) {
                        mram_write(&w_result_batch[can_write], state_table_mram, spill_count << 3);
                        DPU_LOG("[DPU T0] Spilled %u hits to state table MRAM\n", spill_count);
                    }

                    // C. Save Checkpoint with "Spill Flag"
                    // entry_idx = 1 signals "Resume by loading spill from state table"
                    // key_idx stores the spill count
                    KmerCheckpoint save_checkpoint;
                    save_checkpoint.packet_idx = pkt_idx;
                    save_checkpoint.entry_idx = 1;        // FLAG: Load spill on resume
                    save_checkpoint.key_idx = spill_count; // SIZE: How many hits to load
                    save_checkpoint.valid = 1;
                    mram_write(&save_checkpoint, checkpoint_ptr, sizeof(KmerCheckpoint));

                    overflow_occurred = true;
                    goto finish;
                }
                
                // Normal Flush (No Overflow)
                mram_write(w_result_batch, &output_buffer[total_hits_written], batch_count << 3);
                total_hits_written += batch_count;
                batch_count = 0;
            }
            
            // 2. Write Result Sentinel (Delimiter) to MRAM
            // Check if there is space for the delimiter hit
            if (total_hits_written + 1 > max_results) {
                DPU_LOG("[DPU T0] OVERFLOW writing sentinel delimiter\n");
                
                // Standard overflow - no WRAM data to lose at this point
                // Checkpoint simply points to packet_idx, entry_idx=0 (no spill)
                KmerCheckpoint save_checkpoint;
                save_checkpoint.packet_idx = pkt_idx;
                save_checkpoint.entry_idx = 0;  
                save_checkpoint.key_idx = 0;
                save_checkpoint.valid = 1;
                mram_write(&save_checkpoint, checkpoint_ptr, sizeof(KmerCheckpoint));
                
                overflow_occurred = true;
                goto finish;
            }
            
            // Write delimiter directly to MRAM output
            KmerDoubleHit delimiter;
            delimiter.target_id = KMER_RESULT_SENTINEL_TARGET; 
            delimiter.diagonal = 0;
            delimiter.padding = 0;
            mram_write(&delimiter, &output_buffer[total_hits_written], sizeof(KmerDoubleHit));
            total_hits_written++;
            
            DPU_LOG("[DPU T0] Result delimiter written, total_hits=%u\n", total_hits_written);
            
            // 3. Reset State Table for Next Query 
            // The next packet belongs to a new query, so previous diagonals are irrelevant.
            memset(w_state_table, 0xFF, MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
            
            DPU_LOG("[DPU T0] State table reset for next query\n");
            
            // Reset entry tracking for clean start on next query
            start_entry = 0;
            cached_key_idx = -1;
            
            // Skip to next packet (sentinel has no k-mer to search)
            continue;
        }
        
        // Binary search for k-mer (use cached result if resuming same packet)
        int32_t found_idx = -1;
        uint32_t offset_start = 0;
        uint32_t offset_end = 0;
        
        if (pkt_idx == start_packet && cached_key_idx >= 0) {
            // Resume: use cached search result to skip binary search
            found_idx = cached_key_idx;
            
            // Retrieve offsets for the cached key
            __mram_ptr uint32_t* m_offsets = (__mram_ptr uint32_t*)(mram_base + g_descriptor.index_offsets_offset);
            uint32_t align_idx = found_idx & ~1;
            
            if (found_idx & 1) {
                // Odd index: extract from second position of aligned pair
                mram_read(&m_offsets[align_idx], w_offset_buf, 8);
                offset_start = w_offset_buf[1];  // offsets[found_idx]
                
                // Read end offset from next aligned pair
                if (align_idx + 2 > g_descriptor.num_index_keys) {
                    // Can't read next pair, read sentinel at offsets[num_keys]
                    // Ensure alignment when reading the sentinel, as num_index_keys may be odd
                    uint32_t sent_align = g_descriptor.num_index_keys & ~1;
                    mram_read(&m_offsets[sent_align], w_offset_buf, 8);
                    offset_end = (g_descriptor.num_index_keys & 1) ? w_offset_buf[1] : w_offset_buf[0];
                } else {
                    mram_read(&m_offsets[align_idx + 2], w_offset_buf, 8);
                    offset_end = w_offset_buf[0];  // offsets[found_idx+1]
                }
            } else {
                // Even index: read offsets[found_idx] (buf[0]) and offsets[found_idx+1] (buf[1])
                mram_read(&m_offsets[align_idx], w_offset_buf, 8);
                offset_start = w_offset_buf[0];  // offsets[found_idx]
                offset_end = w_offset_buf[1];    // offsets[found_idx+1]
            }
        } else {
            // Normal search
            found_idx = search_kmer_with_offset(kmer_idx, hint_idx, &offset_start, &offset_end);
        }
        
        // If k-mer not found, skip to next packet
        if (found_idx < 0) {
            // Reset entry counter for next packet
            start_entry = 0;
            continue;
        }
        
        // ===== BOUNDS CHECKING =====
        // Validate offset ordering to prevent invalid memory access
        if (offset_start > offset_end) {
            DPU_LOG("[DPU T0] ERROR: Invalid offset range for key_idx=%d! start=%u > end=%u. Skipping packet %u.\n",
                   found_idx, offset_start, offset_end, pkt_idx);
            start_entry = 0;
            continue;
        }
        
        // ===== PROCESS ENTRIES FOR THIS K-MER =====
        uint32_t total_entries = offset_end - offset_start;
        
        // Validate entry count against maximum sequence limit
        // if (total_entries > MAX_DPU_SEQS) {
        //     printf("[DPU T0] WARN: Suspicious entry count %u for key_idx=%d (clamping to %u). Packet %u.\n",
        //            total_entries, found_idx, MAX_DPU_SEQS, pkt_idx);
        //     total_entries = MAX_DPU_SEQS;
        // }
        
        uint32_t entry_base = (pkt_idx == start_packet) ? start_entry : 0;
        
        // Track which aligned block is currently cached to avoid redundant reloads
        uint32_t cached_aligned_start = 0xFFFFFFFF;  // Invalid sentinel
        
        for (uint32_t e = entry_base; e < total_entries; ++e) {
            // Calculate global index in the entries array
            uint32_t global_entry_idx = offset_start + e;
            
            // ===== ENTRY CACHE MANAGEMENT =====
            // KmerCompactIndexEntry is 4 bytes, so we must align to pairs (8 bytes)
            // Same pattern used for uint32_t offset array access
            
            // Determine which logical block this entry belongs to
            // [ >> 5 ] == [ / ENTRY_CACHE_LOGICAL_SIZE ]
            uint32_t logical_block = e >> 5;
            
            // [ << 5 ] == [ * ENTRY_CACHE_LOGICAL_SIZE ]
            uint32_t raw_fetch_start = offset_start + (logical_block << 5);
            
            // Round down to even index to satisfy 8-byte alignment requirements
            uint32_t aligned_fetch_start = raw_fetch_start & ~1;
            
            // Reload cache only when entering a new aligned block
            if (aligned_fetch_start != cached_aligned_start) {
                cached_aligned_start = aligned_fetch_start;
                
                // Calculate how many entries to read (including alignment padding)
                uint32_t alignment_padding = raw_fetch_start - aligned_fetch_start;  // 0 or 1
                uint32_t logical_count = ENTRY_CACHE_LOGICAL_SIZE;
                
                // Clamp to valid range
                if (raw_fetch_start + logical_count > offset_end) {
                    logical_count = offset_end - raw_fetch_start;
                }
                
                uint32_t physical_count = logical_count + alignment_padding;
                
                // Round up to even count for 8-byte transfer alignment
                if (physical_count & 1) {
                    physical_count++;
                }
                
                // Clamp to cache size
                if (physical_count > ENTRY_CACHE_SIZE) {
                    physical_count = ENTRY_CACHE_SIZE;
                }
                
                // Perform 8-byte aligned MRAM read
                // [ << 2 ] == [ * sizeof(KmerCompactIndexEntry) ]
                uint32_t fetch_bytes = physical_count << 2;
                mram_read(&index_entries[aligned_fetch_start], w_entry_cache, fetch_bytes);
            }
            
            // Calculate index into cache: global position minus aligned start
            uint32_t cache_idx = global_entry_idx - cached_aligned_start;
            
            // Bounds check on cache access
            if (cache_idx >= ENTRY_CACHE_SIZE) {
                // FATAL: The cache loading logic above guarantees this fits.
                // If this triggers, the alignment math (lines 339-360) is mathematically wrong.
                DPU_LOG("[DPU T0] FATAL: Cache logic error! cache_idx=%u >= size=%u\n", cache_idx, ENTRY_CACHE_SIZE);
                return -1; 
            }
            
            // Get entry data from cache
            uint16_t target_id = w_entry_cache[cache_idx].local_target_id;
            uint16_t target_pos = w_entry_cache[cache_idx].pos_j;
            
            // Safety check: target_id must be within state table bounds
            if (target_id >= MAX_DPU_SEQS) {
                // FATAL: The IndexBuilder guarantees target_id < MAX_DPU_SEQS.
                // If this triggers, MRAM is corrupt or the IndexBuilder is broken.
                DPU_LOG("[DPU T0] FATAL: MRAM Corruption! target_id=%u >= max=%u\n", target_id, MAX_DPU_SEQS);
                return -1;
            }
            
            // 1. Calculate Full 16-bit Diagonal
            int16_t diagonal = (int16_t)query_pos - (int16_t)target_pos;
            uint8_t diag_u8 = (uint8_t)diagonal;
            
            single_hits_count++;
            
            // ===== STATE MACHINE =====
            KmerDiagonalStateEntry* state = &w_state_table[target_id];
            
            bool is_double_hit = false;

            // Determine "Effective Previous State"
            // If slot is empty (0xFFFF), treat prev diagonal as 0 (CPU Artifact).
            uint8_t prev_diag = (state->pos == 0xFFFF) ? 0 : (uint8_t)state->diag;

            // Check for match
            if (prev_diag == diag_u8) {
                is_double_hit = true;
                double_hits_count++;
                
                // Update state (Extend chain)
                state->pos = query_pos;
                state->diag = diagonal; 
            } else {
                // Mismatch - Overwrite immediately (New seed)
                state->pos = query_pos;
                state->diag = diagonal;
            }
            
            if (!is_double_hit) {
                continue; // Skip single hits - only emit double hits
            }
            
            w_result_batch[batch_count].target_id = target_id;
            w_result_batch[batch_count].diagonal = diagonal;
            w_result_batch[batch_count].padding = 0;
            batch_count++;
            
            // ===== CHECK FOR OVERFLOW BEFORE FLUSHING =====
            if (batch_count >= RESULT_BATCH_SIZE) {
                // Check if we have space in MRAM output buffer
                if (total_hits_written + batch_count > max_results) {
                    DPU_LOG("[DPU T0] OVERFLOW DETECTED at packet %u, entry %u\n", pkt_idx, e);
                    DPU_LOG("[DPU T0]   Would write %u hits, but only %u slots remaining\n",
                           batch_count, max_results - total_hits_written);
                    
                    // Save state table to MRAM before exit
                    // Preserves double-hit detection state for resume
                    safe_mram_write(w_state_table, state_table_mram, state_bytes);
                    
                    // Flush partial batch to MRAM (fill remaining space)
                    uint32_t can_write = max_results - total_hits_written;
                    if (can_write > 0) {
                        // [ << 3 ] == [ * sizeof(KmerDoubleHit) ]
                        mram_write(w_result_batch, &output_buffer[total_hits_written], can_write << 3);
                        total_hits_written += can_write;
                    }
                    
                    // Save checkpoint for resuming from this position
                    // Calculate the next entry index to process after resume
                    // At overflow with full batch: e is current entry, batch contains [e-127..e]
                    // After writing 'can_write' hits, resume at: (e - 127) + can_write
                    KmerCheckpoint save_checkpoint;
                    save_checkpoint.packet_idx = pkt_idx;
                    save_checkpoint.entry_idx = e - (RESULT_BATCH_SIZE - 1) + can_write;
                    save_checkpoint.key_idx = (uint32_t)found_idx;
                    save_checkpoint.valid = 1;
                    mram_write(&save_checkpoint, checkpoint_ptr, sizeof(KmerCheckpoint));
                    
                    DPU_LOG("[DPU T0]   Checkpoint saved: resume at packet=%u entry=%u key=%u\n",
                           save_checkpoint.packet_idx, save_checkpoint.entry_idx, save_checkpoint.key_idx);
                    DPU_LOG("[DPU T0]   (e=%u, batch_count=%u, can_write=%u)\n",
                           e, batch_count, can_write);
                    
                    overflow_occurred = true;
                    goto finish;
                }
                
                // Safe to flush
                // [ << 3 ] == [ * sizeof(KmerDoubleHit) ]
                mram_write(w_result_batch, &output_buffer[total_hits_written], batch_count << 3);
                total_hits_written += batch_count;
                batch_count = 0;
            }
        }
        
        // Flush any remaining hits for this packet before moving to next
        // Per-packet flushing ensures batch_count only tracks current packet
        if (batch_count > 0) {
            if (total_hits_written + batch_count > max_results) {
                DPU_LOG("[DPU T0] OVERFLOW at end of packet %u\n", pkt_idx);
                
                // Save state table to MRAM before exit
                safe_mram_write(w_state_table, state_table_mram, state_bytes);
                
                uint32_t can_write = max_results - total_hits_written;
                if (can_write > 0) {
                    // batch_count << 3 == batch_count * sizeof(KmerDoubleHit))
                    mram_write(w_result_batch, &output_buffer[total_hits_written], can_write << 3);
                    total_hits_written += can_write;
                }
                
                KmerCheckpoint save_checkpoint;
                save_checkpoint.packet_idx = pkt_idx;
                // Calculate resume point: first unwritten entry in current packet
                save_checkpoint.entry_idx = (total_entries - batch_count) + can_write;
                save_checkpoint.key_idx = (uint32_t)found_idx;
                save_checkpoint.valid = 1;
                mram_write(&save_checkpoint, checkpoint_ptr, sizeof(KmerCheckpoint));
                
                DPU_LOG("[DPU T0]   Checkpoint saved: resume at packet=%u entry=%u key=%u\n",
                       save_checkpoint.packet_idx, save_checkpoint.entry_idx, save_checkpoint.key_idx);
                DPU_LOG("[DPU T0]   (total_entries=%u, batch_count=%u, can_write=%u)\n",
                       total_entries, batch_count, can_write);
                
                overflow_occurred = true;
                goto finish;
            }
            
            // [ << 3 ] == [ * sizeof(KmerDoubleHit) ]
            mram_write(w_result_batch, &output_buffer[total_hits_written], batch_count << 3);
            total_hits_written += batch_count;
            batch_count = 0;
        }
        
        // Reset entry counter for next packet
        start_entry = 0;
    }
    
finish:
    // ===== FLUSH REMAINING HITS =====
    if (batch_count > 0 && !overflow_occurred) {
        if (total_hits_written + batch_count <= max_results) {
            // batch_count << 3 == batch_count * sizeof(KmerDoubleHit))
            mram_write(w_result_batch, &output_buffer[total_hits_written], batch_count << 3);
            total_hits_written += batch_count;
        }
    }
    
    // Always save state table back to MRAM (for multi-batch queries)
    safe_mram_write(w_state_table, state_table_mram, state_bytes);
    
    // write result header
    KmerResultHeader result_header;
    result_header.total_hits = total_hits_written;
    result_header.overflow = overflow_occurred ? 1 : 0;
    mram_write(&result_header, result_header_ptr, sizeof(KmerResultHeader));
    
    DPU_LOG("[DPU T0] EXECUTION COMPLETE:\n");
    DPU_LOG("[DPU T0]   Single hits processed: %u\n", single_hits_count);
    DPU_LOG("[DPU T0]   Double hits detected: %u\n", double_hits_count);
    DPU_LOG("[DPU T0]   Double hits written: %u\n", total_hits_written);
    DPU_LOG("[DPU T0]   Overflow: %s\n", overflow_occurred ? "YES" : "NO");
    if (!overflow_occurred) {
        DPU_LOG("[DPU T0]   All packets processed successfully\n");
    }
    
    return 0;
}