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

#define ENTRY_BUFFER_CAPACITY 510
#define ENTRY_BUFFER_SIZE (ENTRY_BUFFER_CAPACITY + 2)

#define TRANSACTION_BATCH_SIZE 131072
#define LOCAL_HIT_BUFFER_SIZE 32

BARRIER_INIT(g_barrier, NR_TASKLETS);
MUTEX_INIT(g_output_mutex);

// Descriptor from Host (48 bytes, 8-byte aligned)
__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;

// State Table: per-target diagonal tracking (32KB, 8-byte aligned)
// Each entry is 4 bytes, but array is 8-byte aligned for MRAM transfers
__dma_aligned KmerDiagonalStateEntry wram_state_table[MAX_DPU_SEQS];

// Shared entry buffer for Leader-Follower pattern (2KB + padding, 8-byte aligned)
// Extra 2 entries for alignment padding during misaligned MRAM reads
__dma_aligned KmerIndexEntry wram_entry_buffer[ENTRY_BUFFER_SIZE];

// Current packet being processed (8 bytes, naturally aligned)
__dma_aligned KmerQueryPacket wram_current_packet;
volatile uint32_t g_do_state_table_reset;

__dma_aligned KmerDoubleHit g_hit_buffers[NR_TASKLETS][LOCAL_HIT_BUFFER_SIZE];
__dma_aligned KmerBucket g_bucket_caches[NR_TASKLETS];

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
__mram_ptr KmerIndexEntry* mram_entries;
__mram_ptr KmerQueryPacket* mram_query_packets;
__mram_ptr KmerDoubleHit* mram_output_buffer;
__mram_ptr KmerCheckpoint* mram_checkpoint;
__mram_ptr KmerDiagonalStateEntry* mram_state_table;

/**
 * Look up a k-mer in the hash index, returning offset and count if found.
 * Handles chained overflow buckets.
 */
static bool lookup_bucket(uint32_t kmer_idx, uint16_t bucket_idx,
                          uint32_t* out_offset, uint32_t* out_count,
                          KmerBucket* bucket_cache) {
    uint32_t current_bucket = (uint32_t)bucket_idx;
                          
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
 * Reads 4-byte KmerIndexEntry structs from MRAM with proper 8-byte alignment.
 */
static uint32_t read_entries_aligned(__mram_ptr KmerIndexEntry* mram_base, 
                                     uint32_t start_idx, 
                                     uint32_t count, 
                                     KmerIndexEntry* wram_dst) {
    if (count == 0) return 0;
    
    // 1. Calculate the byte offset of the requested entry
    uint32_t byte_offset = start_idx * sizeof(KmerIndexEntry);  // start_idx * 4
    
    // 2. Align down to 8 bytes
    uint32_t aligned_offset = ALIGN8_DOWN(byte_offset);
    uint32_t misalignment = byte_offset & 7u;  // Will be 0 or 4
    
    // 3. Calculate total bytes to read (from aligned start to end, rounded up to 8)
    uint32_t end_offset = byte_offset + (count * sizeof(KmerIndexEntry));
    uint32_t read_len = ALIGN8(end_offset - aligned_offset);
    
    if (read_len > ENTRY_BUFFER_SIZE * sizeof(KmerIndexEntry)) {
        g_shared.overflow_occurred = 2;
        g_shared.transaction_aborted = 1;
        return 0;
    }
    
    // 4. MRAM read from aligned address
    mram_read((__mram_ptr uint8_t*)mram_base + aligned_offset, wram_dst, read_len);
    
    // 5. Return offset: 1 if misaligned (data starts at index 1), 0 if exact
    return (misalignment == 4) ? 1 : 0;
}

/**
 * Parallel MRAM read using all active tasklets.
 */
static void mram_read_parallel(__mram_ptr const void* src, void* dst, uint32_t size) {
    uint32_t chunk_size = (size + NR_TASKLETS - 1) / NR_TASKLETS;
    chunk_size = ALIGN8(chunk_size);

    uint32_t offset = me() * chunk_size;

    if (offset < size) {
        uint32_t len = size - offset;
        if (len > chunk_size) len = chunk_size;
        
        mram_read_safe((__mram_ptr uint8_t*)src + offset, (uint8_t*)dst + offset, len);
    }
}

/**
 * Parallel MRAM write using all active tasklets.
 */
static void mram_write_parallel(const void* src, __mram_ptr void* dst, uint32_t size) {
    uint32_t chunk_size = (size + NR_TASKLETS - 1) / NR_TASKLETS;
    chunk_size = ALIGN8(chunk_size);

    uint32_t offset = me() * chunk_size;

    if (offset < size) {
        uint32_t len = size - offset;
        if (len > chunk_size) len = chunk_size;
        
        mram_write_safe((const uint8_t*)src + offset, (__mram_ptr uint8_t*)dst + offset, len);
    }
}

static inline void flush_hit_buffer(KmerDoubleHit* buffer, uint32_t* count, uint32_t max_results) {
    if (*count == 0) return;

    mutex_lock(g_output_mutex);
    
    bool success = (!g_shared.transaction_aborted && g_shared.total_mram_hits_written + *count <= max_results);
    
    uint32_t write_offset = g_shared.total_mram_hits_written;
    
    if (success) {
        g_shared.total_mram_hits_written += *count;
    } else {
        g_shared.transaction_aborted = 1;
    }
    
    mutex_unlock(g_output_mutex);
    
    if (success) {
        mram_write(buffer, &mram_output_buffer[write_offset], *count * sizeof(KmerDoubleHit));
    }

    *count = 0;
}

int main() {
    // PHASE 1: INITIALIZATION (Leader sets up, all tasklets wait at barrier)
    if (me() == 0) {
        mram_base = DPU_MRAM_HEAP_POINTER;
        mram_read(mram_base, &g_descriptor, sizeof(KmerBatchDescriptor));
        
        // Initialize MRAM pointers (all offsets are 8-byte aligned by Host)
        mram_checkpoint = (__mram_ptr KmerCheckpoint*)(mram_base + g_descriptor.checkpoint_offset);
        mram_state_table = (__mram_ptr KmerDiagonalStateEntry*)(mram_base + g_descriptor.state_table_offset);
        mram_query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
        mram_buckets = (__mram_ptr KmerBucket*)(mram_base + g_descriptor.buckets_offset);
        mram_entries = (__mram_ptr KmerIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
        mram_output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset + sizeof(KmerResultHeader));
        
        // Load checkpoint to resume from last committed position
        // Host MUST zero the checkpoint before first launch.
        __dma_aligned KmerCheckpoint checkpoint;
        mram_read(mram_checkpoint, &checkpoint, sizeof(KmerCheckpoint));
        
        // Only use checkpoint if valid flag is set (valid=1 means in-progress, valid=0 means complete/fresh)
        uint32_t state_size = ALIGN8(g_descriptor.num_targets * sizeof(KmerDiagonalStateEntry));

        if (checkpoint.valid == 1 && checkpoint.packet_idx < g_descriptor.num_query_packets) {
            g_shared.current_packet_idx = checkpoint.packet_idx;
            g_do_state_table_reset = 0;
        } 
        else {
            g_shared.current_packet_idx = 0;
            g_do_state_table_reset = 1;
        }
        
        g_shared.total_mram_hits_written = 0;
        g_shared.mram_kmer_entry_start_index = 0;
        g_shared.entries_for_kmer_total = 0;
        g_shared.entries_buffer_count = 0;
        g_shared.overflow_occurred = 0;
        g_shared.transaction_aborted = 0;
    }
    barrier_wait(&g_barrier);

    // For idle DPUs - host sends an empty descriptor 
    if (g_descriptor.num_query_packets == 0) {
        return 0;
    }

    uint32_t total_state_table_entries = g_descriptor.num_targets;

    if (g_do_state_table_reset) {
        uint32_t chunk = (total_state_table_entries + NR_TASKLETS - 1) / NR_TASKLETS;
        uint32_t start_idx = me() * chunk;
        uint32_t end_idx = start_idx + chunk;
        if (end_idx > total_state_table_entries) end_idx = total_state_table_entries;

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            wram_state_table[i].pos = 0xFFFF;
        }
    } else {
        uint32_t state_size = ALIGN8(total_state_table_entries * sizeof(KmerDiagonalStateEntry));
        mram_read_parallel(mram_state_table, wram_state_table, state_size);
    }
    barrier_wait(&g_barrier);
    
    // PHASE 2: TRANSACTIONAL BATCH PROCESSING
    KmerDoubleHit* tasklet_hit_buffer = g_hit_buffers[me()];
    KmerBucket* tasklet_bucket_cache = &g_bucket_caches[me()];
    
    // Calculate max results that fit in output buffer
    uint32_t max_results = (g_descriptor.results_buffer_size - sizeof(KmerResultHeader)) / sizeof(KmerDoubleHit);
    
    // Main processing loop - each iteration is one transaction batch
    while (1) {
        uint32_t batch_start_packet;
        uint32_t batch_start_hits;
        uint32_t tasklet_hit_count = 0;
        
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
            if (wram_current_packet.kmer_idx == KMER_PACKET_SENTINEL) {
                uint32_t chunk_size = (g_descriptor.num_targets + NR_TASKLETS - 1) / NR_TASKLETS;
                uint32_t start_target_idx = me() * chunk_size;
                uint32_t end_target_idx = start_target_idx + chunk_size;
                if (end_target_idx > g_descriptor.num_targets) end_target_idx = g_descriptor.num_targets;
                
                // Reset state entries to "no previous hit"
                for (uint32_t target_idx = start_target_idx; target_idx < end_target_idx; ++target_idx) {
                    wram_state_table[target_idx].pos = 0xFFFF;
                }
                
                flush_hit_buffer(tasklet_hit_buffer, &tasklet_hit_count, max_results);
                barrier_wait(&g_barrier);
                
                // Leader writes sentinel marker to output
                if (me() == 0) {
                    mutex_lock(g_output_mutex);
                    if (!g_shared.transaction_aborted && g_shared.total_mram_hits_written + 2 <= max_results) {
                        __dma_aligned KmerDoubleHit sentinel_hits[2];
                        sentinel_hits[0].target_id = KMER_RESULT_SENTINEL;
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
                
                bool found = lookup_bucket(wram_current_packet.kmer_idx, wram_current_packet.bucket_idx, &offset, &count, tasklet_bucket_cache);

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
                KmerIndexEntry* valid_entries = &wram_entry_buffer[g_shared.wram_buffer_valid_start_offset];
                
                for (uint32_t entry_idx = 0; entry_idx < entries_chunk_len; ++entry_idx) {
                    KmerIndexEntry entry = valid_entries[entry_idx];
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
                            flush_hit_buffer(tasklet_hit_buffer, &tasklet_hit_count, max_results);
                        }
                    }
                }
                
                processed += entries_chunk_len;
                
                barrier_wait(&g_barrier);
            }
            
            // Flush remaining local hits after each packet
            barrier_wait(&g_barrier);
        }

        flush_hit_buffer(tasklet_hit_buffer, &tasklet_hit_count, max_results);
        barrier_wait(&g_barrier);

        // Transaction Commit or Rollback 
        // 1. LEADER: Update Global State / Headers / Checkpoints
        if (me() == 0) {
            g_shared.current_packet_idx = batch_start_packet + TRANSACTION_BATCH_SIZE;
            if (g_shared.current_packet_idx > g_descriptor.num_query_packets) {
                g_shared.current_packet_idx = g_descriptor.num_query_packets;
            }

            if (g_shared.transaction_aborted) {
                // --- ROLLBACK LOGIC DISABLED ---
                g_shared.total_mram_hits_written = batch_start_hits;
                if (g_shared.overflow_occurred != 2) g_shared.overflow_occurred = 1;
                
                __dma_aligned KmerResultHeader result_header;
                result_header.total_hits = g_shared.total_mram_hits_written;
                result_header.overflow = g_shared.overflow_occurred;
                
                __mram_ptr KmerResultHeader* hdr_ptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
                mram_write(&result_header, hdr_ptr, sizeof(KmerResultHeader));
            } 
            else {
                // --- CHECKPOINT WRITE DISABLED ---
                __dma_aligned KmerCheckpoint checkpoint;
                checkpoint.packet_idx = g_shared.current_packet_idx;
                checkpoint.valid = 1;
                mram_write(&checkpoint, mram_checkpoint, sizeof(KmerCheckpoint));
            }
        }
        barrier_wait(&g_barrier);
        
        if (!g_shared.transaction_aborted) {
            uint32_t state_size = DPU_ALIGN_SIZE(g_descriptor.num_targets * sizeof(KmerDiagonalStateEntry));
            if (state_size > MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry)) {
                state_size = MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry);
            }

            mram_write_parallel(wram_state_table, mram_state_table, state_size);
        }
        barrier_wait(&g_barrier);
        
        if (g_shared.overflow_occurred) {
            break;
        }
    }
    
    // PHASE 3: Finalize processing
    if (me() == 0) {
        // Write header if successful OR if overflow occurred (to notify host)
        // Original logic was "&& !g_shared.overflow_occurred", but since we disabled 
        // intermediate overflow reporting, we must write the final header state here.
        
        __dma_aligned KmerResultHeader result_header;
        result_header.total_hits = g_shared.total_mram_hits_written;
        result_header.overflow = g_shared.overflow_occurred; // Reports 0, 1, or 2
        
        __mram_ptr KmerResultHeader* hdr_ptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
        mram_write(&result_header, hdr_ptr, sizeof(KmerResultHeader));
        
        // Reset Checkpoint (Finish)
        __dma_aligned KmerCheckpoint checkpoint;
        checkpoint.packet_idx = g_descriptor.num_query_packets;
        checkpoint.valid = 0;
        mram_write(&checkpoint, mram_checkpoint, sizeof(KmerCheckpoint));
    }
    
    return 0;
}