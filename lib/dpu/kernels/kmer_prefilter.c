#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
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

#define ENTRY_BUFFER_CAPACITY 508
#define ENTRY_BUFFER_SIZE (ENTRY_BUFFER_CAPACITY + 2)

#define TRANSACTION_BATCH_SIZE 128
#define MAX_MRAM_TRANSFER_SIZE 2048
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

BARRIER_INIT(g_barrier, NR_TASKLETS);
MUTEX_INIT(g_output_mutex);

// Descriptor from Host (48 bytes, 8-byte aligned)
__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;

// State Table: per-target diagonal tracking (32KB, 8-byte aligned)
// Each entry is 4 bytes, but array is 8-byte aligned for MRAM transfers
__host __attribute__((aligned(8))) KmerDiagonalStateEntry wram_state_table[MAX_DPU_SEQS];

// Shared entry buffer for Leader-Follower pattern (2KB + padding, 8-byte aligned)
// Extra 2 entries for alignment padding during misaligned MRAM reads
__attribute__((aligned(8))) KmerCompactIndexEntry wram_entry_buffer[ENTRY_BUFFER_SIZE];

// Current packet being processed (8 bytes, naturally aligned)
__attribute__((aligned(8))) KmerQueryPacket wram_current_packet;

// Shared control variables accessed under barrier synchronization
__attribute__((aligned(8))) struct {
    uint32_t entries_buffer_count;              // Valid entries in wram_entry_buffer
    uint32_t mram_kmer_entry_start_index;       // MRAM offset for current k-mer's entries
    uint32_t entries_for_kmer_total;            // Total entries for current k-mer
    uint32_t current_packet_idx;                // Global progress tracker
    uint32_t total_mram_hits_written;           // Hits written to MRAM
    uint32_t overflow_occurred;                 // Overflow flag (use uint32_t for alignment)
    uint32_t transaction_aborted;               // Soft abort flag
    uint32_t wram_buffer_valid_start_offset;    // 0 or 1: where valid data starts in wram_entry_buffer
} g_shared;

__mram_ptr uint8_t* mram_base;
__mram_ptr KmerBucket* mram_buckets;
__mram_ptr KmerCompactIndexEntry* mram_entries;
__mram_ptr KmerQueryPacket* mram_query_packets;
__mram_ptr KmerDoubleHit* mram_output_buffer;
__mram_ptr KmerCheckpoint* mram_checkpoint;
__mram_ptr KmerDiagonalStateEntry* mram_state_table;

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
    
    if (current_bucket >= g_descriptor.num_buckets) {
        g_shared.overflow_occurred = 2;
        g_shared.transaction_aborted = 1;
        return false;
    }
    
    while (current_bucket != CHAIN_END_IDX) {
        if (current_bucket >= g_descriptor.num_buckets) {
            g_shared.overflow_occurred = 2;
            g_shared.transaction_aborted = 1;
            return false;
        }
        
        mram_read(&mram_buckets[current_bucket], bucket_cache, sizeof(KmerBucket));
        
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
    
    if (read_len > ENTRY_BUFFER_SIZE * sizeof(KmerCompactIndexEntry)) {
        g_shared.overflow_occurred = 2;
        g_shared.transaction_aborted = 1;
        return 0;
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
        mram_buckets = (__mram_ptr KmerBucket*)(mram_base + g_descriptor.buckets_offset);
        mram_entries = (__mram_ptr KmerCompactIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
        mram_query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
        mram_output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset + sizeof(KmerResultHeader));
        mram_checkpoint = (__mram_ptr KmerCheckpoint*)(mram_base + g_descriptor.checkpoint_offset);
        mram_state_table = (__mram_ptr KmerDiagonalStateEntry*)(mram_base + g_descriptor.state_table_offset);
        
        // Load checkpoint to resume from last committed position
        // Host MUST zero the checkpoint before first launch.
        __attribute__((aligned(8))) KmerCheckpoint checkpoint;
        mram_read(mram_checkpoint, &checkpoint, sizeof(KmerCheckpoint));
        
        // Only use checkpoint if valid flag is set (valid=1 means in-progress, valid=0 means complete/fresh)
        // If valid is garbage (e.g., 0xFFFFFFFF), treat as fresh start
        if (checkpoint.valid == 1 && checkpoint.packet_idx < g_descriptor.num_query_packets) {
            g_shared.current_packet_idx = checkpoint.packet_idx;
        } else {
            g_shared.current_packet_idx = 0;
        }
        
        if (g_descriptor.num_targets > MAX_DPU_SEQS) {
            g_shared.overflow_occurred = 2;
            g_shared.transaction_aborted = 1;
        } else {
            uint32_t state_size = DPU_ALIGN_SIZE(g_descriptor.num_targets * sizeof(KmerDiagonalStateEntry));
            mram_read_safe(mram_state_table, wram_state_table, state_size);
        }
        
        g_shared.total_mram_hits_written = 0;
        g_shared.overflow_occurred = 0;
        g_shared.transaction_aborted = 0;
        g_shared.entries_buffer_count = 0;
        g_shared.mram_kmer_entry_start_index = 0;
        g_shared.entries_for_kmer_total = 0;
    }
    barrier_wait(&g_barrier);
    
    // For idle DPUs - host sends an empty descriptor 
    if (g_descriptor.num_query_packets == 0) {
        return 0;
    }
    
    // PHASE 2: TRANSACTIONAL BATCH PROCESSING
    __attribute__((aligned(8))) KmerDoubleHit tasklet_hit_buffer[LOCAL_HIT_BUFFER_SIZE];
    __attribute__((aligned(8))) KmerBucket tasklet_bucket_cache; 
    uint32_t tasklet_hit_count = 0;
    
    // Calculate max results that fit in output buffer
    uint32_t max_results = (g_descriptor.results_buffer_size - sizeof(KmerResultHeader)) / sizeof(KmerDoubleHit);
    
    // Main processing loop - each iteration is one transaction batch
    while (1) {
        uint32_t batch_start_packet;
        uint32_t batch_start_hits;
        
        if (me() == 0) {
            batch_start_packet = g_shared.current_packet_idx;
            batch_start_hits = g_shared.total_mram_hits_written;
            g_shared.transaction_aborted = 0;
        }
        barrier_wait(&g_barrier);
        
        // Exit if all packets processed or overflow occurred
        if (batch_start_packet >= g_descriptor.num_query_packets || g_shared.overflow_occurred) {
            break; 
        }
        
        // Process packets in this transaction batch
        for (uint32_t batch_offset = 0; batch_offset < TRANSACTION_BATCH_SIZE; ++batch_offset) {
            uint32_t packet_index = batch_start_packet + batch_offset;
            
            if (packet_index >= g_descriptor.num_query_packets || g_shared.transaction_aborted) {
                break;
            }
            
            // Fetch current packet from MRAM
            if (me() == 0) {
                if (packet_index < g_descriptor.num_query_packets) {
                    mram_read(&mram_query_packets[packet_index], &wram_current_packet, sizeof(KmerQueryPacket));
                }
                g_shared.entries_buffer_count = 0;
                g_shared.entries_for_kmer_total = 0;
            }
            barrier_wait(&g_barrier);
            
            // Reset state for all targets for a new query
            if (wram_current_packet.kmer_idx == KMER_PACKET_SENTINEL_KEY) {
                // Each tasklet resets it's owned targets
                uint32_t chunk_size = (g_descriptor.num_targets + NR_TASKLETS - 1) / NR_TASKLETS;
                uint32_t start_target_idx = me() * chunk_size;
                uint32_t end_target_idx = start_target_idx + chunk_size;
                if (end_target_idx > g_descriptor.num_targets) end_target_idx = g_descriptor.num_targets;
                
                // Reset state entries to "no previous hit"
                for (uint32_t target_idx = start_target_idx; target_idx < end_target_idx; ++target_idx) {
                    wram_state_table[target_idx].pos = 0xFFFF;
                    wram_state_table[target_idx].diag = 0;
                }
                
                // Leader writes sentinel marker to output
                if (me() == 0) {
                    mutex_lock(g_output_mutex);
                    if (!g_shared.transaction_aborted && g_shared.total_mram_hits_written + 2 <= max_results) {
                        __attribute__((aligned(8))) KmerDoubleHit sentinel_hits[2];
                        sentinel_hits[0].target_id = KMER_RESULT_SENTINEL_TARGET;
                        sentinel_hits[0].diagonal = 0;
                        sentinel_hits[0].padding = 0;
                        sentinel_hits[1].target_id = KMER_TARGET_ID_PADDING;
                        sentinel_hits[1].diagonal = 0;
                        sentinel_hits[1].padding = 0;
                        
                        mram_write(sentinel_hits, &mram_output_buffer[g_shared.total_mram_hits_written], 16);
                        g_shared.total_mram_hits_written += 2;
                    } else {
                        g_shared.transaction_aborted = 1;
                    }
                    mutex_unlock(g_output_mutex);
                }
                barrier_wait(&g_barrier);

                continue;
            }
            
            // Leader looks up k-mer in hash index
            if (me() == 0) {
                uint32_t offset = 0;
                uint32_t count = 0;
                
                bool found = lookup_bucket(wram_current_packet.kmer_idx, wram_current_packet.bucket_idx, &offset, &count, &tasklet_bucket_cache);

                if (found && count > 0) {
                    if (offset + count <= g_descriptor.num_index_entries) {
                        g_shared.mram_kmer_entry_start_index = offset;
                        g_shared.entries_for_kmer_total = count;
                    } else {
                        g_shared.overflow_occurred = 2;
                        g_shared.transaction_aborted = 1;
                    }
                }
            }
            barrier_wait(&g_barrier);
            
            uint32_t processed = 0;
            uint32_t total_entries = g_shared.entries_for_kmer_total;
            uint32_t mram_entry_start_index = g_shared.mram_kmer_entry_start_index; 
            
            while (processed < total_entries) {
                if (g_shared.transaction_aborted) {
                    break;
                }
                
                uint32_t entries_chunk_len = 0;
                
                if (me() == 0) {
                    entries_chunk_len = total_entries - processed;
                    if (entries_chunk_len > ENTRY_BUFFER_CAPACITY) {
                        entries_chunk_len = ENTRY_BUFFER_CAPACITY;
                    }
                    
                    if (mram_entry_start_index + processed + entries_chunk_len <= g_descriptor.num_index_entries) {
                        // Returns offset (0 or 1) where valid data starts 
                        uint32_t offset = read_entries_aligned(mram_entries, mram_entry_start_index + processed, entries_chunk_len, wram_entry_buffer);
                        g_shared.wram_buffer_valid_start_offset = offset;
                        
                        // Check if the read failed (buffer overflow detected above)
                        if (g_shared.transaction_aborted) {
                            entries_chunk_len = 0;
                        }
                    } else {
                        g_shared.overflow_occurred = 2;
                        g_shared.transaction_aborted = 1;
                        entries_chunk_len = 0; 
                    }
                    g_shared.entries_buffer_count = entries_chunk_len;
                }
                barrier_wait(&g_barrier);
                
                entries_chunk_len = g_shared.entries_buffer_count;
                
                // Each tasklet scans all entries but only acts on owned targets
                KmerCompactIndexEntry* valid_entries = &wram_entry_buffer[g_shared.wram_buffer_valid_start_offset];
                
                for (uint32_t entry_idx = 0; entry_idx < entries_chunk_len; ++entry_idx) {
                    KmerCompactIndexEntry entry = valid_entries[entry_idx];
                    uint16_t target_id = entry.local_target_id;
                    
                    if (target_id >= g_descriptor.num_targets) {
                        g_shared.overflow_occurred = 2;
                        g_shared.transaction_aborted = 1;
                        break;
                    }

                    if ((target_id & (NR_TASKLETS - 1)) != (uint16_t)me()) {
                        continue;
                    }
                    
                    KmerDiagonalStateEntry* state = &wram_state_table[target_id];
                    
                    int16_t diag = (int16_t)wram_current_packet.query_pos - (int16_t)entry.pos_j;
                    bool is_double = (state->pos != 0xFFFF && state->diag == diag);
                    
                    // Update state regardless of double hit    
                    state->pos = wram_current_packet.query_pos;
                    state->diag = diag;
                    
                    if (is_double) {
                        tasklet_hit_buffer[tasklet_hit_count].target_id = target_id;
                        tasklet_hit_buffer[tasklet_hit_count].diagonal = diag;
                        tasklet_hit_buffer[tasklet_hit_count].padding = 0;
                        tasklet_hit_count++;
                        
                        if (tasklet_hit_count >= LOCAL_HIT_BUFFER_SIZE) {
                            mutex_lock(g_output_mutex);
                            bool success = (!g_shared.transaction_aborted && g_shared.total_mram_hits_written + tasklet_hit_count <= max_results);
                            uint32_t my_write_offset = g_shared.total_mram_hits_written;
                            
                            if (success) {
                                g_shared.total_mram_hits_written += tasklet_hit_count;
                            } else {
                                g_shared.transaction_aborted = 1;
                            }
                            mutex_unlock(g_output_mutex);
                            
                            if (success) {
                                mram_write(tasklet_hit_buffer, &mram_output_buffer[my_write_offset], tasklet_hit_count * sizeof(KmerDoubleHit));
                            }
                            tasklet_hit_count = 0;
                        }
                    }
                }
                
                processed += entries_chunk_len;
                
                barrier_wait(&g_barrier);
            }
            
            // Flush remaining local hits after each packet
            if (tasklet_hit_count > 0) {
                mutex_lock(g_output_mutex);
                bool success = (!g_shared.transaction_aborted && g_shared.total_mram_hits_written + tasklet_hit_count <= max_results);
                uint32_t my_write_offset = g_shared.total_mram_hits_written;
                
                if (success) {
                    g_shared.total_mram_hits_written += tasklet_hit_count;
                } else {
                    g_shared.transaction_aborted = 1;
                }
                mutex_unlock(g_output_mutex);
                
                if (success) {
                    mram_write(tasklet_hit_buffer, &mram_output_buffer[my_write_offset], tasklet_hit_count * sizeof(KmerDoubleHit));
                }
                tasklet_hit_count = 0;
            }
            barrier_wait(&g_barrier);
        }
        
        // Transaction Commit or Rollback 
        if (me() == 0) {
            if (g_shared.transaction_aborted) {
                g_shared.total_mram_hits_written = batch_start_hits;
                if (g_shared.overflow_occurred != 2) g_shared.overflow_occurred = 1;
                
                __attribute__((aligned(8))) KmerResultHeader result_header;
                result_header.total_hits = g_shared.total_mram_hits_written;
                result_header.overflow = g_shared.overflow_occurred;
                
                __mram_ptr KmerResultHeader* hdr_ptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
                mram_write(&result_header, hdr_ptr, sizeof(KmerResultHeader));
            } 
            else {
                g_shared.current_packet_idx = batch_start_packet + TRANSACTION_BATCH_SIZE;
                if (g_shared.current_packet_idx > g_descriptor.num_query_packets) {
                    g_shared.current_packet_idx = g_descriptor.num_query_packets;
                }
                
                __attribute__((aligned(8))) KmerCheckpoint checkpoint;
                checkpoint.packet_idx = g_shared.current_packet_idx;
                checkpoint.entry_idx = 0;
                checkpoint.padding = 0;
                checkpoint.valid = 1;
                mram_write(&checkpoint, mram_checkpoint, sizeof(KmerCheckpoint));
                
                uint32_t state_size = DPU_ALIGN_SIZE(g_descriptor.num_targets * sizeof(KmerDiagonalStateEntry));
                if (state_size > MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry)) {
                    state_size = MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry);
                }
                mram_write_safe(wram_state_table, mram_state_table, state_size);
            }
        }
        barrier_wait(&g_barrier);
        
        if (g_shared.overflow_occurred) {
            break;
        }
    }
    
    // PHASE 3: Finalize processing if no overflow occurred
    if (me() == 0 && !g_shared.overflow_occurred) {
        __attribute__((aligned(8))) KmerResultHeader result_header;
        result_header.total_hits = g_shared.total_mram_hits_written;
        result_header.overflow = 0;
        
        __mram_ptr KmerResultHeader* hdr_ptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
        mram_write(&result_header, hdr_ptr, sizeof(KmerResultHeader));
        
        __attribute__((aligned(8))) KmerCheckpoint checkpoint;
        checkpoint.packet_idx = g_descriptor.num_query_packets;
        checkpoint.entry_idx = 0;
        checkpoint.padding = 0;
        checkpoint.valid = 0;
        mram_write(&checkpoint, mram_checkpoint, sizeof(KmerCheckpoint));
    }
    
    return 0;
}