#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#include "DpuSharedTypes.h"

// ================= TUNING PARAMETERS =================
#ifndef NR_TASKLETS
#define NR_TASKLETS 16           // Increased to 16 to hide more latency
#endif

// REDUCED CACHE SIZE: 
// 128 keys (512B) was too large for random access. 
// 32 keys (128B) is the sweet spot for DPU DMA.
#define KEY_CACHE_SIZE 32        
#define RESULT_BATCH_SIZE 64 
#define ENTRY_CACHE_LOGICAL_SIZE 32
#define ENTRY_CACHE_SIZE (ENTRY_CACHE_LOGICAL_SIZE + 2)

// ================= SPINLOCK HELPERS =================
typedef uint32_t spinlock_t;

static inline void spin_lock(volatile spinlock_t *lock) {
    while (__sync_lock_test_and_set(lock, 1)) {
        // Busy wait (WRAM only)
    }
}

static inline void spin_unlock(volatile spinlock_t *lock) {
    __sync_lock_release(lock);
}

// ================= GLOBAL STRUCTURES =================
__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;
__host __attribute__((aligned(8))) uint32_t w_hint_table[HINT_TABLE_SIZE + 2];
__host __attribute__((aligned(8))) KmerDiagonalStateEntry w_state_table[MAX_DPU_SEQS];

// ================= SHARED SYNCHRONIZATION =================
BARRIER_INIT(g_barrier, NR_TASKLETS);
MUTEX_INIT(g_output_mutex);

#define STATE_LOCK_COUNT 64
volatile spinlock_t g_state_locks[STATE_LOCK_COUNT]; 

volatile uint32_t g_next_packet_idx;
uint32_t g_batch_limit_idx;
uint32_t g_total_hits_written;
bool g_overflow_occurred;

__mram_ptr uint8_t* mram_base;
__mram_ptr uint32_t* m_keys;
__mram_ptr uint32_t* m_offsets;
__mram_ptr KmerCompactIndexEntry* m_entries;
__mram_ptr KmerQueryPacket* m_query_packets;
__mram_ptr KmerDoubleHit* m_output_buffer;

static uint32_t find_next_sentinel(uint32_t start_idx, uint32_t max_idx) {
    uint32_t current = start_idx;
    KmerQueryPacket tmp;
    while (current < max_idx) {
        mram_read(&m_query_packets[current], &tmp, 8);
        if (tmp.kmer_idx == KMER_PACKET_SENTINEL_KEY) {
            return current;
        }
        current++;
    }
    return max_idx;
}

static int32_t search_kmer_with_offset(uint32_t kmer_idx, uint16_t hint_idx, 
                                      uint32_t* out_offset_start, uint32_t* out_offset_end,
                                      uint32_t* key_cache, uint32_t* offset_buf) {
    if (hint_idx >= HINT_TABLE_SIZE) return -1;
    
    uint32_t low = w_hint_table[hint_idx];
    uint32_t high = w_hint_table[hint_idx + 1];
    
    if (low >= high) return -1;
    if (low >= g_descriptor.num_index_keys) return -1;
    if (high > g_descriptor.num_index_keys) high = g_descriptor.num_index_keys;
    
    // Binary Search Block Loop
    while (low < high) {
        uint32_t mid = low + ((high - low) >> 1);
        // Align to cache block size
        uint32_t block_start = mid & ~(KEY_CACHE_SIZE - 1);
        
        if (block_start >= g_descriptor.num_index_keys) { high = mid; continue; }
        
        uint32_t block_end = block_start + KEY_CACHE_SIZE;
        if (block_end > g_descriptor.num_index_keys) block_end = g_descriptor.num_index_keys;
        uint32_t fetch_count = block_end - block_start;
        
        // Fetch 128 bytes (32 keys) instead of 512 bytes
        uint32_t fetch_bytes = ((fetch_count << 2) + 7) & ~7;
        mram_read(&m_keys[block_start], key_cache, fetch_bytes);
        
        uint32_t block_min = key_cache[0];
        uint32_t block_max = key_cache[fetch_count - 1];
        
        if (kmer_idx >= block_min && kmer_idx <= block_max) {
            // Found the block, linear scan inside WRAM cache
            for (uint32_t i = 0; i < fetch_count; ++i) {
                if (key_cache[i] == kmer_idx) {
                    uint32_t found_idx = block_start + i;
                    uint32_t align_idx = found_idx & ~1;
                    
                    if (found_idx & 1) {
                        mram_read(&m_offsets[align_idx], offset_buf, 8);
                        *out_offset_start = offset_buf[1];
                        if (align_idx + 2 > g_descriptor.num_index_keys) {
                            // Boundary check for last odd key
                             uint32_t last_offset_idx = g_descriptor.num_index_keys; // sentinel index
                             // We need to read the offset at 'last_offset_idx'
                             // Align it down
                             uint32_t sent_align = last_offset_idx & ~1;
                             mram_read(&m_offsets[sent_align], offset_buf, 8);
                             *out_offset_end = (last_offset_idx & 1) ? offset_buf[1] : offset_buf[0];
                        } else {
                            mram_read(&m_offsets[align_idx + 2], offset_buf, 8);
                            *out_offset_end = offset_buf[0];
                        }
                    } else {
                        mram_read(&m_offsets[align_idx], offset_buf, 8);
                        *out_offset_start = offset_buf[0];
                        *out_offset_end = offset_buf[1];
                    }
                    return found_idx;
                }
            }
            return -1; // In block range but not found
        }
        
        if (kmer_idx < block_min) high = block_start;
        else low = block_start + KEY_CACHE_SIZE;
    }
    return -1; 
}

int main() {
    if (me() == 0) {
        mram_base = (__mram_ptr uint8_t*)DPU_MRAM_HEAP_POINTER;
        mram_read(mram_base, &g_descriptor, sizeof(KmerBatchDescriptor));
        
        __mram_ptr uint32_t* hint_mram = (__mram_ptr uint32_t*)(mram_base + g_descriptor.hint_table_offset);
        uint32_t hint_bytes = ((HINT_TABLE_SIZE + 1) * sizeof(uint32_t) + 7) & ~7;
        mram_read(hint_mram, w_hint_table, hint_bytes);
        
        m_keys = (__mram_ptr uint32_t*)(mram_base + g_descriptor.index_keys_offset);
        m_offsets = (__mram_ptr uint32_t*)(mram_base + g_descriptor.index_offsets_offset);
        m_entries = (__mram_ptr KmerCompactIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
        m_query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
        m_output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset + sizeof(KmerResultHeader));
        
        g_next_packet_idx = 0;
        g_batch_limit_idx = 0;
        g_total_hits_written = 0;
        g_overflow_occurred = false;
        
        for(int i=0; i<STATE_LOCK_COUNT; i++) g_state_locks[i] = 0;
    }
    
    barrier_wait(&g_barrier);
    
    uint32_t t_key_cache[KEY_CACHE_SIZE];
    KmerDoubleHit t_result_batch[RESULT_BATCH_SIZE];
    KmerQueryPacket t_packet_cache;
    KmerCompactIndexEntry t_entry_cache[ENTRY_CACHE_SIZE];
    uint32_t t_offset_buf[2];
    uint32_t t_batch_count = 0;
    
    uint32_t max_results = (g_descriptor.results_buffer_size - sizeof(KmerResultHeader)) >> 3;
    
    while (true) {
        // PHASE 1: Scan
        if (me() == 0) {
            uint32_t start = g_next_packet_idx;
            g_batch_limit_idx = (start < g_descriptor.num_query_packets) 
                ? find_next_sentinel(start, g_descriptor.num_query_packets) 
                : start;
        }
        barrier_wait(&g_barrier);
        
        if (g_next_packet_idx >= g_descriptor.num_query_packets) break;
        if (g_overflow_occurred) break;
        
        // PHASE 2: Process
        while (true) {
            uint32_t my_pkt_idx;
            spin_lock(&g_state_locks[0]); 
            if (g_next_packet_idx >= g_batch_limit_idx) {
                spin_unlock(&g_state_locks[0]);
                break; 
            }
            my_pkt_idx = g_next_packet_idx++;
            spin_unlock(&g_state_locks[0]);
            
            mram_read(&m_query_packets[my_pkt_idx], &t_packet_cache, 8);
            
            uint32_t offset_start, offset_end;
            int32_t found_idx = search_kmer_with_offset(t_packet_cache.kmer_idx, t_packet_cache.hint_idx,
                                                       &offset_start, &offset_end, 
                                                       t_key_cache, t_offset_buf);
            
            if (found_idx >= 0 && offset_start < offset_end) {
                uint32_t total_entries = offset_end - offset_start;
                if (total_entries > MAX_DPU_SEQS) total_entries = MAX_DPU_SEQS;
                
                uint32_t cached_aligned_start = 0xFFFFFFFF;
                
                for (uint32_t e = 0; e < total_entries; ++e) {
                     uint32_t global_idx = offset_start + e;
                     uint32_t raw_fetch = offset_start + (e >> 5 << 5); 
                     uint32_t aligned_fetch = raw_fetch & ~1;
                     
                     if (aligned_fetch != cached_aligned_start) {
                         cached_aligned_start = aligned_fetch;
                         uint32_t count = ENTRY_CACHE_SIZE;
                         if (aligned_fetch + count > offset_end) count = offset_end - aligned_fetch + 1;
                         if (count & 1) count++;
                         mram_read(&m_entries[aligned_fetch], t_entry_cache, count << 2);
                     }
                     
                     uint32_t c_idx = global_idx - cached_aligned_start;
                     uint16_t tid = t_entry_cache[c_idx].local_target_id;
                     
                     if (tid >= MAX_DPU_SEQS) continue;

                     // [OPTIMIZATION] Calculate diagonal before locking to reduce critical section
                     int16_t tpos = t_entry_cache[c_idx].pos_j;
                     int16_t diag = (int16_t)t_packet_cache.query_pos - (int16_t)tpos;
                     uint8_t diag_u8 = (uint8_t)diag;
                     
                     // === STATE UPDATE ===
                     uint8_t lock_id = tid % STATE_LOCK_COUNT;
                     spin_lock(&g_state_locks[lock_id]);
                     
                     KmerDiagonalStateEntry* state = &w_state_table[tid];
                     bool is_double = false;
                     if (state->pos != 0xFFFF && (uint8_t)state->diag == diag_u8) {
                         is_double = true;
                     }
                     state->pos = t_packet_cache.query_pos;
                     state->diag = diag;
                     
                     spin_unlock(&g_state_locks[lock_id]);
                     
                     if (is_double) {
                         t_result_batch[t_batch_count].target_id = tid;
                         t_result_batch[t_batch_count].diagonal = diag;
                         t_result_batch[t_batch_count].padding = 0;
                         t_batch_count++;
                         
                         if (t_batch_count >= RESULT_BATCH_SIZE) {
                             mutex_lock(g_output_mutex);
                             if (g_total_hits_written + t_batch_count <= max_results) {
                                 mram_write(t_result_batch, &m_output_buffer[g_total_hits_written], t_batch_count << 3);
                                 g_total_hits_written += t_batch_count;
                             } else {
                                 g_overflow_occurred = true;
                             }
                             mutex_unlock(g_output_mutex);
                             t_batch_count = 0;
                         }
                     }
                }
            }
            if (g_overflow_occurred) break;
        }
        
        barrier_wait(&g_barrier);
        
        // PHASE 2.5: INTERMEDIATE FLUSH
        if (t_batch_count > 0) {
            mutex_lock(g_output_mutex);
            if (!g_overflow_occurred && g_total_hits_written + t_batch_count <= max_results) {
                 mram_write(t_result_batch, &m_output_buffer[g_total_hits_written], t_batch_count << 3);
                 g_total_hits_written += t_batch_count;
            } else if (!g_overflow_occurred) {
                 g_overflow_occurred = true;
            }
            mutex_unlock(g_output_mutex);
            t_batch_count = 0; 
        }

        barrier_wait(&g_barrier);
        
        // PHASE 3: Sentinel (Tasklet 0)
        if (me() == 0 && g_batch_limit_idx < g_descriptor.num_query_packets) {
             mutex_lock(g_output_mutex);
             if (g_total_hits_written + 1 <= max_results) {
                 KmerDoubleHit sent = { .target_id = KMER_RESULT_SENTINEL_TARGET, .diagonal = 0, .padding = 0 };
                 mram_write(&sent, &m_output_buffer[g_total_hits_written], 8);
                 g_total_hits_written++;
             } else {
                 g_overflow_occurred = true;
             }
             mutex_unlock(g_output_mutex);
             memset(w_state_table, 0xFF, MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
             g_next_packet_idx = g_batch_limit_idx + 1;
        }
        
        barrier_wait(&g_barrier);
    }
    
    if (me() == 0) {
        KmerResultHeader hdr = { .total_hits = g_total_hits_written, .overflow = g_overflow_occurred ? 1 : 0 };
        __mram_ptr KmerResultHeader* hptr = (__mram_ptr KmerResultHeader*)(mram_base + g_descriptor.results_offset);
        mram_write(&hdr, hptr, sizeof(KmerResultHeader));
    }
    
    return 0;
}