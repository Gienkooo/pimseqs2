/**
 * K-mer Prefilter DPU Kernel - Target-Partitioned Leader-Follower Model
 * 
 * ALGORITHM REQUIREMENTS:
 * 1. Packets MUST be processed strictly in order (position index ordering)
 * 2. For diagonal d, two hits are a "double hit" only if no other diagonal 
 *    was seen for that target between them
 * 3. On query boundary (sentinel), state must be reset for all targets
 * 
 * IMPLEMENTATION:
 * - Leader (Tasklet 0): Fetches packets and index entries from MRAM to shared WRAM
 * - Followers (All Tasklets): Process hits only for targets they "own" (tid % NR_TASKLETS == me())
 * - Eliminates spinlocks and guarantees sequential packet processing
 * - Transactional batches with checkpoint/rollback for overflow handling
 * 
 * DPU CONSTRAINTS ENFORCED:
 * - All MRAM transfers are 8-byte aligned (addresses and sizes)
 * - WRAM budget: ~52KB used of 60KB available
 * - Max single MRAM transfer: 2KB (chunked for larger transfers)
 * - No recursion, no large stack allocations
 * - All barriers reached by all tasklets unconditionally
 */

#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#include "dpu_common.h"
#include "DpuSharedTypes.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#define IS_POWER_OF_2(x) ((x) && !((x) & ((x) - 1)))
_Static_assert(IS_POWER_OF_2(NR_TASKLETS), "NR_TASKLETS must be a power of 2!");

// ============================================================================
// CONFIGURATION (All sizes are 8-byte aligned)
// ============================================================================

// Entry buffer capacity calculation:
// - DPU max single MRAM transfer: 2048 bytes
// - Worst case: misaligned read adds 4 bytes, round-up adds up to 7 bytes
// - Max safe: (capacity * 4 + 11) & ~7 <= 2048
// - capacity <= (2048 - 4) / 4 = 511, use 508 for safety margin
// Add +2 entries for alignment padding when misaligned reads occur
#define ENTRY_BUFFER_CAPACITY 508
#define ENTRY_BUFFER_SIZE (ENTRY_BUFFER_CAPACITY + 2)

// Transaction batch size: process this many packets before committing state
#define TRANSACTION_BATCH_SIZE 128

// Maximum MRAM single transfer size (DPU hardware limit)
#define MAX_MRAM_TRANSFER_SIZE 2048

// Local hit buffer per tasklet: 32 hits * 8 bytes = 256 bytes (8-byte aligned)
#define LOCAL_HIT_BUFFER_SIZE 32

/**
 * Safe MRAM write for buffers larger than 2KB.
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
        uint32_t chunk = (size - offset > MAX_MRAM_TRANSFER_SIZE) 
                        ? MAX_MRAM_TRANSFER_SIZE : (size - offset);
        mram_write(&src_ptr[offset], &dst_ptr[offset], chunk);
        offset += chunk;
    }
}

/**
 * Safe MRAM read for buffers larger than 2KB.
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
        uint32_t chunk = (size - offset > MAX_MRAM_TRANSFER_SIZE) 
                        ? MAX_MRAM_TRANSFER_SIZE : (size - offset);
        mram_read(&src_ptr[offset], &dst_ptr[offset], chunk);
        offset += chunk;
    }
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

BARRIER_INIT(g_barrier, NR_TASKLETS);
MUTEX_INIT(g_output_mutex);

// ============================================================================
// GLOBAL STATE (Static WRAM - no stack allocation)
// ============================================================================

// Descriptor from Host (48 bytes, 8-byte aligned)
__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;

// State Table: per-target diagonal tracking (32KB, 8-byte aligned)
// Each entry is 4 bytes, but array is 8-byte aligned for MRAM transfers
__host __attribute__((aligned(8))) KmerDiagonalStateEntry w_state_table[MAX_DPU_SEQS];

// Shared entry buffer for Leader-Follower pattern (2KB + padding, 8-byte aligned)
// Extra 2 entries for alignment padding during misaligned MRAM reads
__attribute__((aligned(8))) KmerCompactIndexEntry w_entry_buffer[ENTRY_BUFFER_SIZE];

// Current packet being processed (8 bytes, naturally aligned)
__attribute__((aligned(8))) KmerQueryPacket w_current_packet;

// Shared control variables (accessed under barrier synchronization)
__attribute__((aligned(8))) struct {
    uint32_t entries_count;       // Valid entries in w_entry_buffer
    uint32_t entries_offset;      // MRAM offset for current k-mer's entries
    uint32_t entries_total;       // Total entries for current k-mer
    uint32_t current_packet_idx;  // Global progress tracker
    uint32_t total_hits_written;  // Hits written to MRAM
    uint32_t overflow_occurred;   // Overflow flag (use uint32_t for alignment)
    uint32_t transaction_aborted; // Soft abort flag
    uint32_t buffer_start_idx;    // 0 or 1: where valid data starts in w_entry_buffer
} g_shared;

// ============================================================================
// MRAM POINTERS (set once during initialization)
// ============================================================================

__mram_ptr uint8_t* mram_base;
__mram_ptr KmerBucket* m_buckets;
__mram_ptr KmerCompactIndexEntry* m_entries;
__mram_ptr KmerQueryPacket* m_query_packets;
__mram_ptr KmerDoubleHit* m_output_buffer;
__mram_ptr KmerCheckpoint* m_checkpoint;
__mram_ptr KmerDiagonalStateEntry* m_state_table;

// ============================================================================
// BUCKET LOOKUP (Leader only, uses per-tasklet cache)
// ============================================================================

/**
 * Look up a k-mer in the hash index, returning offset and count if found.
 * Handles chained overflow buckets.
 * 
 * @param kmer_idx    K-mer value to look up
 * @param bucket_idx  Pre-computed hash bucket index
 * @param out_offset  [out] Start offset in entries array
 * @param out_count   [out] Number of entries for this k-mer
 * @param bucket_cache 8-byte aligned WRAM buffer for bucket reads (256 bytes)
 * @return true if k-mer found, false otherwise
 */
static bool lookup_bucket(uint32_t kmer_idx, uint16_t bucket_idx,
                          uint32_t* out_offset, uint32_t* out_count,
                          KmerBucket* bucket_cache) {
    uint32_t current_bucket = (uint32_t)bucket_idx;
    
    // Bounds check: bucket index must be valid
    if (current_bucket >= g_descriptor.num_buckets) {
        return false;
    }
    
    while (current_bucket != CHAIN_END_IDX) {
        // Bounds check before MRAM access
        if (current_bucket >= g_descriptor.num_buckets) {
            return false;
        }
        
        // KmerBucket is 256 bytes (8-byte aligned), single transfer OK
        mram_read(&m_buckets[current_bucket], bucket_cache, sizeof(KmerBucket));
        
        for (uint16_t i = 0; i < bucket_cache->count && i < BUCKET_CAPACITY; ++i) {
            if (bucket_cache->items[i].key == kmer_idx) {
                *out_offset = bucket_cache->items[i].offset;
                *out_count = bucket_cache->items[i].count;
                return true;
            }
        }
        current_bucket = bucket_cache->next_idx;
    }
    return false;
}

// ============================================================================
// ALIGNED MRAM READ HELPER
// ============================================================================

/**
 * Reads 4-byte KmerCompactIndexEntry structs from MRAM with proper 8-byte alignment.
 * 
 * The DPU MRAM requires 8-byte aligned addresses for all transfers.
 * KmerCompactIndexEntry is only 4 bytes, so when start_idx is odd, the byte
 * address would end in 0x4, violating alignment requirements.
 * 
 * This function:
 * 1. Calculates the 8-byte aligned read address
 * 2. Reads data including any padding bytes needed for alignment
 * 3. Returns an offset (0 or 1) indicating where valid data starts
 * 
 * OPTIMIZATION: Instead of shifting data (expensive), we return an offset.
 * The caller uses pointer arithmetic: valid_entries = &buffer[offset]
 * 
 * @param mram_base  Pointer to the start of the entries array in MRAM
 * @param start_idx  Index of the first entry to read (0-based)
 * @param count      Number of entries to read
 * @param wram_dst   Destination WRAM buffer (must have space for count+2 entries)
 * @return           Offset where valid data starts: 0 (aligned) or 1 (misaligned by 4)
 */
static uint32_t read_entries_aligned(__mram_ptr KmerCompactIndexEntry* mram_base, 
                                     uint32_t start_idx, 
                                     uint32_t count, 
                                     KmerCompactIndexEntry* wram_dst) {
    if (count == 0) return 0;
    
    // 1. Calculate the byte offset of the requested entry
    uint32_t byte_offset = start_idx * sizeof(KmerCompactIndexEntry);  // start_idx * 4
    
    // 2. Align down to 8 bytes
    uint32_t aligned_offset = byte_offset & ~7u;
    uint32_t misalignment = byte_offset & 7u;  // Will be 0 or 4
    
    // 3. Calculate total bytes to read (from aligned start to end, rounded up to 8)
    uint32_t end_offset = byte_offset + (count * sizeof(KmerCompactIndexEntry));
    uint32_t read_len = ((end_offset - aligned_offset) + 7u) & ~7u;
    
    // Safety: Cap read length to avoid buffer overflow
    if (read_len > ENTRY_BUFFER_SIZE * sizeof(KmerCompactIndexEntry)) {
        read_len = ENTRY_BUFFER_SIZE * sizeof(KmerCompactIndexEntry);
    }
    
    // 4. MRAM read from aligned address
    mram_read((__mram_ptr uint8_t*)mram_base + aligned_offset, wram_dst, read_len);
    
    // 5. Return offset: 1 if misaligned (data starts at index 1), 0 if exact
    return (misalignment == 4) ? 1 : 0;
}


int main() {

    // PHASE 1: INITIALIZATION (Leader sets up, all tasklets wait at barrier)
    if (me() == 0) {
        mram_base = (__mram_ptr uint8_t*)DPU_MRAM_HEAP_POINTER;
        mram_read(mram_base, &g_descriptor, sizeof(KmerBatchDescriptor));
        
        // Initialize MRAM pointers (all offsets are 8-byte aligned by Host)
        m_buckets = (__mram_ptr KmerBucket*)(mram_base + g_descriptor.buckets_offset);
        m_entries = (__mram_ptr KmerCompactIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
        m_query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
        m_output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset + sizeof(KmerResultHeader));
        m_checkpoint = (__mram_ptr KmerCheckpoint*)(mram_base + g_descriptor.checkpoint_offset);
        m_state_table = (__mram_ptr KmerDiagonalStateEntry*)(mram_base + g_descriptor.state_table_offset);
        
        // Load checkpoint to resume from last committed position
        // IMPORTANT: Host MUST zero the checkpoint before first launch.
        // We verify the valid flag to catch uninitialized MRAM.
        KmerCheckpoint ckpt;
        mram_read(m_checkpoint, &ckpt, sizeof(KmerCheckpoint));
        
        // Only use checkpoint if valid flag is set (valid=1 means in-progress, valid=0 means complete/fresh)
        // If valid is garbage (e.g., 0xFFFFFFFF), treat as fresh start
        if (ckpt.valid == 1 && ckpt.packet_idx < g_descriptor.num_query_packets) {
            g_shared.current_packet_idx = ckpt.packet_idx;
        } else {
            // Fresh start or completed - start from beginning
            g_shared.current_packet_idx = 0;
        }
        
        // Load state table from MRAM backing store (32KB, chunked reads)
        // On first launch, Host initializes MRAM state to 0xFFFFFFFF (reset state)
        mram_read_safe(m_state_table, w_state_table, MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
        
        // Initialize shared control variables
        g_shared.total_hits_written = 0;
        g_shared.overflow_occurred = 0;
        g_shared.transaction_aborted = 0;
        g_shared.entries_count = 0;
        g_shared.entries_offset = 0;
        g_shared.entries_total = 0;
    }
    
    barrier_wait(&g_barrier);
    
    // For idle DPUs - host sends an empty descriptor 
    if (g_descriptor.num_query_packets == 0) {
        return 0;
    }
    
    // PHASE 2: TRANSACTIONAL BATCH PROCESSING
    __attribute__((aligned(8))) KmerDoubleHit t_local_hits[LOCAL_HIT_BUFFER_SIZE];
    __attribute__((aligned(8))) KmerBucket t_bucket_cache;  // 256 bytes for bucket lookup
    uint32_t t_local_hit_count = 0;
    
    // Calculate max results that fit in output buffer
    uint32_t max_results = (g_descriptor.results_buffer_size - sizeof(KmerResultHeader)) / sizeof(KmerDoubleHit);
    
    // Main processing loop - each iteration is one transaction batch
    while (1) {
        // ----- Transaction Start (Leader captures batch boundaries) -----
        uint32_t batch_start_packet;
        uint32_t batch_start_hits;
        
        if (me() == 0) {
            batch_start_packet = g_shared.current_packet_idx;
            batch_start_hits = g_shared.total_hits_written;
            g_shared.transaction_aborted = 0;
        }
        
        // BARRIER: Ensure all tasklets see consistent batch start state
        barrier_wait(&g_barrier);
        
        // Check completion conditions (all tasklets check, no conditional barrier)
        if (batch_start_packet >= g_descriptor.num_query_packets) {
            break;  // All packets processed
        }
        if (g_shared.overflow_occurred) {
            break;  // Overflow from previous batch
        }
        
        // ----- Process packets in this transaction batch -----
        for (uint32_t batch_offset = 0; batch_offset < TRANSACTION_BATCH_SIZE; ++batch_offset) {
            uint32_t pkt_idx = batch_start_packet + batch_offset;
            
            // Check bounds (all tasklets check consistently)
            if (pkt_idx >= g_descriptor.num_query_packets) {
                break;
            }
            
            // BARRIER: Sync before checking abort flag
            barrier_wait(&g_barrier);
            
            if (g_shared.transaction_aborted) {
                break;
            }
            
            // === LEADER: Fetch current packet from MRAM ===
            if (me() == 0) {
                // Bounds check
                if (pkt_idx < g_descriptor.num_query_packets) {
                    mram_read(&m_query_packets[pkt_idx], &w_current_packet, sizeof(KmerQueryPacket));
                }
                g_shared.entries_count = 0;
                g_shared.entries_total = 0;
            }
            
            // BARRIER: All tasklets wait for packet to be loaded
            barrier_wait(&g_barrier);
            
            // === SENTINEL PACKET: Reset state for all targets ===
            if (w_current_packet.kmer_idx == KMER_PACKET_SENTINEL_KEY) {
                // Owner-Computes Reset: Each tasklet resets its owned targets
                // Divides target sequences evenly among NR_TASKLETS (16) = 512 each
                uint32_t chunk_size = (MAX_DPU_SEQS + NR_TASKLETS - 1) / NR_TASKLETS;
                uint32_t start_t = me() * chunk_size;
                uint32_t end_t = start_t + chunk_size;
                if (end_t > MAX_DPU_SEQS) end_t = MAX_DPU_SEQS;
                
                // Reset state entries to "no previous hit" (0xFFFF means invalid)
                for (uint32_t t = start_t; t < end_t; ++t) {
                    w_state_table[t].pos = 0xFFFF;
                    w_state_table[t].diag = 0;
                }
                
                // BARRIER: All resets complete
                barrier_wait(&g_barrier);
                
                // Leader writes sentinel marker to output
                if (me() == 0) {
                    mutex_lock(g_output_mutex);
                    if (!g_shared.transaction_aborted && g_shared.total_hits_written + 2 <= max_results) {
                        // Write sentinel pair (16 bytes, 8-byte aligned)
                        __attribute__((aligned(8))) KmerDoubleHit sents[2];
                        sents[0].target_id = KMER_RESULT_SENTINEL_TARGET;
                        sents[0].diagonal = 0;
                        sents[0].padding = 0;
                        sents[1].target_id = KMER_TARGET_ID_PADDING;
                        sents[1].diagonal = 0;
                        sents[1].padding = 0;
                        
                        mram_write(sents, &m_output_buffer[g_shared.total_hits_written], 16);
                        g_shared.total_hits_written += 2;
                    } else {
                        g_shared.transaction_aborted = 1;
                    }
                    mutex_unlock(g_output_mutex);
                }
                
                // BARRIER: Before next packet
                barrier_wait(&g_barrier);
                continue;
            }
            
            // === LEADER: Look up k-mer in hash index ===
            if (me() == 0) {
                uint32_t offset = 0;
                uint32_t count = 0;
                
                bool found = lookup_bucket(w_current_packet.kmer_idx, 
                                          w_current_packet.bucket_idx,
                                          &offset, &count,
                                          &t_bucket_cache);
                if (found && count > 0) {
                    // Bounds check: ensure offset + count doesn't exceed index
                    if (offset + count <= g_descriptor.num_index_entries) {
                        g_shared.entries_offset = offset;
                        g_shared.entries_total = (count > MAX_DPU_SEQS) ? MAX_DPU_SEQS : count;
                    }
                }
            }
            
            // BARRIER: All tasklets wait for lookup result
            barrier_wait(&g_barrier);
            
            // === PROCESS ENTRIES IN CHUNKS ===
            // Using aligned read helper to handle 4-byte entry alignment on 8-byte MRAM
            uint32_t processed = 0;
            uint32_t total_entries = g_shared.entries_total;
            uint32_t base_idx = g_shared.entries_offset;  // Entry index (not byte offset)
            
            while (processed < total_entries) {
                // Check abort before each chunk
                if (g_shared.transaction_aborted) {
                    break;
                }
                
                // Leader fetches next chunk of entries using aligned read
                uint32_t chunk_len = 0;
                
                if (me() == 0) {
                    chunk_len = total_entries - processed;
                    if (chunk_len > ENTRY_BUFFER_CAPACITY) {
                        chunk_len = ENTRY_BUFFER_CAPACITY;
                    }
                    
                    // Bounds check before read
                    if (base_idx + processed + chunk_len <= g_descriptor.num_index_entries) {
                        // Use aligned read helper to handle 4-byte entries on 8-byte aligned MRAM
                        // Returns offset (0 or 1) where valid data starts - zero-cost pointer optimization
                        uint32_t offset = read_entries_aligned(m_entries, base_idx + processed, chunk_len, w_entry_buffer);
                        g_shared.buffer_start_idx = offset;
                    } else {
                        g_shared.buffer_start_idx = 0;
                    }
                    g_shared.entries_count = chunk_len;
                }
                
                // BARRIER: Wait for chunk to be loaded
                barrier_wait(&g_barrier);
                
                chunk_len = g_shared.entries_count;
                
                // === FOLLOWERS: Owner-Computes pattern ===
                // Each tasklet scans ALL entries but only acts on owned targets
                // Use pointer arithmetic to access valid data (zero-cost optimization)
                KmerCompactIndexEntry* valid_entries = &w_entry_buffer[g_shared.buffer_start_idx];
                
                for (uint32_t e = 0; e < chunk_len; ++e) {
                    KmerCompactIndexEntry entry = valid_entries[e];
                    uint16_t tid = entry.local_target_id;
                    
                    // Bounds check: skip invalid target IDs
                    // Use num_targets (actual count in this batch) not MAX_DPU_SEQS
                    if (tid >= g_descriptor.num_targets) {
                        continue;
                    }
                    
                    // OWNERSHIP CHECK: Only process targets assigned to this tasklet
                    // This guarantees no two tasklets access the same state entry
                    if ((tid & (NR_TASKLETS - 1)) != (uint16_t)me()) {
                        continue;
                    }
                    
                    // Access state (exclusive - no lock needed due to ownership)
                    KmerDiagonalStateEntry* state = &w_state_table[tid];
                    
                    // Calculate diagonal: query_pos - target_pos
                    int16_t diag = (int16_t)w_current_packet.query_pos - (int16_t)entry.pos_j;
                    
                    // Check for double hit: same diagonal as previous hit for this target
                    bool is_double = (state->pos != 0xFFFF && state->diag == diag);
                    
                    // Update state to last seen diagonal (always, regardless of double hit)
                    state->pos = w_current_packet.query_pos;
                    state->diag = diag;
                    
                    if (is_double) {
                        // Buffer the hit locally
                        t_local_hits[t_local_hit_count].target_id = tid;
                        t_local_hits[t_local_hit_count].diagonal = diag;
                        t_local_hits[t_local_hit_count].padding = 0;
                        t_local_hit_count++;
                        
                        // Flush if local buffer is full (8 hits = 64 bytes, 8-byte aligned)
                        if (t_local_hit_count >= LOCAL_HIT_BUFFER_SIZE) {
                            mutex_lock(g_output_mutex);
                            bool success = (!g_shared.transaction_aborted && 
                                            g_shared.total_hits_written + t_local_hit_count <= max_results);
                            uint32_t my_write_offset = g_shared.total_hits_written;
                            
                            if (success) {
                                g_shared.total_hits_written += t_local_hit_count;
                            } else {
                                g_shared.transaction_aborted = 1;
                            }
                            mutex_unlock(g_output_mutex);
                            
                            if (success) {
                                mram_write(t_local_hits, 
                                          &m_output_buffer[my_write_offset],
                                          t_local_hit_count * sizeof(KmerDoubleHit));
                            }
                            t_local_hit_count = 0;
                        }
                    }
                }
                
                processed += chunk_len;
                
                // BARRIER: All tasklets sync after processing chunk
                barrier_wait(&g_barrier);
            }
            
            // Flush remaining local hits after each packet
            if (t_local_hit_count > 0) {
                mutex_lock(g_output_mutex);
                bool success = (!g_shared.transaction_aborted && 
                                g_shared.total_hits_written + t_local_hit_count <= max_results);
                uint32_t my_write_offset = g_shared.total_hits_written;
                
                if (success) {
                    g_shared.total_hits_written += t_local_hit_count;
                } else {
                    g_shared.transaction_aborted = 1;
                }
                mutex_unlock(g_output_mutex);
                
                if (success) {
                    mram_write(t_local_hits, 
                              &m_output_buffer[my_write_offset],
                              t_local_hit_count * sizeof(KmerDoubleHit));
                }
                t_local_hit_count = 0;
            }
            
            // BARRIER: Before next packet in batch
            barrier_wait(&g_barrier);
        }
        
        // ----- Transaction Commit or Rollback (Leader handles, all wait) -----
        if (me() == 0) {
            if (g_shared.transaction_aborted) {
                // ROLLBACK: Revert to batch start state
                g_shared.total_hits_written = batch_start_hits;
                g_shared.overflow_occurred = 1;
                
                // Write header with overflow flag
                __attribute__((aligned(8))) KmerResultHeader hdr;
                hdr.total_hits = g_shared.total_hits_written;
                hdr.overflow = 1;
                
                __mram_ptr KmerResultHeader* hdr_ptr = 
                    (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
                mram_write(&hdr, hdr_ptr, sizeof(KmerResultHeader));
                
                // DO NOT update checkpoint - leave at batch_start_packet for resume
                // State table in MRAM remains valid from previous commit
                
            } else {
                // COMMIT: Update progress
                g_shared.current_packet_idx = batch_start_packet + TRANSACTION_BATCH_SIZE;
                if (g_shared.current_packet_idx > g_descriptor.num_query_packets) {
                    g_shared.current_packet_idx = g_descriptor.num_query_packets;
                }
                
                // Save checkpoint (8-byte aligned struct)
                __attribute__((aligned(8))) KmerCheckpoint ckpt;
                ckpt.packet_idx = g_shared.current_packet_idx;
                ckpt.entry_idx = 0;
                ckpt.key_idx = 0;
                ckpt.valid = 1;
                mram_write(&ckpt, m_checkpoint, sizeof(KmerCheckpoint));
                
                // Save state table to MRAM backing store (32KB, chunked writes)
                mram_write_safe(w_state_table, m_state_table, 
                               MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
            }
        }
        
        // BARRIER: All tasklets wait for commit/rollback to complete
        barrier_wait(&g_barrier);
        
        if (g_shared.overflow_occurred) {
            break;
        }
    }
    
    // =========================================================================
    // PHASE 3: FINALIZE (Leader writes final header)
    // =========================================================================
    
    if (me() == 0 && !g_shared.overflow_occurred) {
        // Write final header (success case)
        __attribute__((aligned(8))) KmerResultHeader hdr;
        hdr.total_hits = g_shared.total_hits_written;
        hdr.overflow = 0;
        
        __mram_ptr KmerResultHeader* hdr_ptr = 
            (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
        mram_write(&hdr, hdr_ptr, sizeof(KmerResultHeader));
        
        // Final checkpoint (all packets processed)
        __attribute__((aligned(8))) KmerCheckpoint ckpt;
        ckpt.packet_idx = g_descriptor.num_query_packets;
        ckpt.entry_idx = 0;
        ckpt.key_idx = 0;
        ckpt.valid = 0;  // Mark as complete
        mram_write(&ckpt, m_checkpoint, sizeof(KmerCheckpoint));
    }
    
    return 0;
}