#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#include "DpuSharedTypes.h"

// ================= CONFIGURATION =================
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

// Limits
#define RESULT_BATCH_SIZE 64 
#define ENTRY_CACHE_LOGICAL_SIZE 32
#define ENTRY_CACHE_SIZE (ENTRY_CACHE_LOGICAL_SIZE + 2)

// ================= SYNCHRONIZATION =================
typedef uint32_t spinlock_t;

static inline void spin_lock(volatile spinlock_t *lock) {
    while (__sync_lock_test_and_set(lock, 1)) { }
}

static inline void spin_unlock(volatile spinlock_t *lock) {
    __sync_lock_release(lock);
}

// ================= GLOBAL STRUCTURES =================
__host __attribute__((aligned(8))) KmerBatchDescriptor g_descriptor;
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
__mram_ptr KmerBucket* m_buckets;
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

// O(1) Hash Lookup Logic
static bool lookup_bucket(uint32_t kmer_idx, uint16_t bucket_idx,
                          uint32_t* out_offset, uint16_t* out_count,
                          KmerBucket* bucket_cache) {
    // Cast to 32-bit to handle overflow buckets
    uint32_t current_bucket = (uint32_t)bucket_idx;
    
    // Traverse Chained Buckets (typically 1 iteration)
    while (current_bucket != CHAIN_END_IDX) {
        // Fetch 256-byte bucket
        mram_read(&m_buckets[current_bucket], bucket_cache, sizeof(KmerBucket));
        
        // Linear scan in WRAM (Zero Latency)
        for (uint16_t i = 0; i < bucket_cache->count; ++i) {
            if (bucket_cache->items[i].key == kmer_idx) {
                *out_offset = bucket_cache->items[i].offset;
                *out_count = bucket_cache->items[i].count;
                return true; // Match found
            }
        }
        
        // Follow link to next bucket
        current_bucket = bucket_cache->next_idx;
    }
    
    return false; // Not found
}

// ================= MAIN KERNEL =================

int main() {
    if (me() == 0) {
        mram_base = (__mram_ptr uint8_t*)DPU_MRAM_HEAP_POINTER;
        mram_read(mram_base, &g_descriptor, sizeof(KmerBatchDescriptor));
        
        // Setup Pointers using Bucketed Descriptor Fields
        m_buckets = (__mram_ptr KmerBucket*)(mram_base + g_descriptor.buckets_offset);
        m_entries = (__mram_ptr KmerCompactIndexEntry*)(mram_base + g_descriptor.index_entries_offset);
        m_query_packets = (__mram_ptr KmerQueryPacket*)(mram_base + g_descriptor.query_packets_offset);
        m_output_buffer = (__mram_ptr KmerDoubleHit*)(mram_base + g_descriptor.results_offset + sizeof(KmerResultHeader));
        
        // Initialize State Table
        __mram_ptr KmerDiagonalStateEntry* state_mram = 
            (__mram_ptr KmerDiagonalStateEntry*)(mram_base + g_descriptor.state_table_offset);
        
        // 8-byte aligned size
        uint32_t state_bytes = ((MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry)) + 7) & ~7;
        
        // Note: We don't read state from MRAM (it's initialized by host reset or previous run)
        // For simple prefilter, we can just zero it in WRAM.
        memset(w_state_table, 0xFF, MAX_DPU_SEQS * sizeof(KmerDiagonalStateEntry));
        
        g_next_packet_idx = 0;
        g_batch_limit_idx = 0;
        g_total_hits_written = 0;
        g_overflow_occurred = false;
        
        for(int i=0; i<STATE_LOCK_COUNT; i++) g_state_locks[i] = 0;
    }
    
    barrier_wait(&g_barrier);
    if (g_overflow_occurred) return 0;
    
    // --- Local Tasklet Cache ---
    __attribute__((aligned(8))) KmerBucket t_bucket_cache; 
    KmerDoubleHit t_result_batch[RESULT_BATCH_SIZE];
    KmerQueryPacket t_packet_cache;
    KmerCompactIndexEntry t_entry_cache[ENTRY_CACHE_SIZE];
    uint32_t t_batch_count = 0;
    uint32_t max_results = (g_descriptor.results_buffer_size - sizeof(KmerResultHeader)) >> 3;
    
    while (true) {
        // PHASE 1: Scan for Sentinel
        if (me() == 0) {
            uint32_t start = g_next_packet_idx;
            g_batch_limit_idx = (start < g_descriptor.num_query_packets) 
                ? find_next_sentinel(start, g_descriptor.num_query_packets) 
                : start;
        }
        barrier_wait(&g_barrier);
        
        if (g_next_packet_idx >= g_descriptor.num_query_packets) break;
        if (g_overflow_occurred) break;
        
        // PHASE 2: Parallel Process
        while (true) {
            uint32_t my_idx;
            spin_lock(&g_state_locks[0]); 
            if (g_next_packet_idx >= g_batch_limit_idx) {
                spin_unlock(&g_state_locks[0]);
                break; 
            }
            my_idx = g_next_packet_idx++;
            spin_unlock(&g_state_locks[0]);
            
            mram_read(&m_query_packets[my_idx], &t_packet_cache, 8);
            
            // === O(1) BUCKET LOOKUP ===
            uint32_t offset_start;
            uint16_t entry_count;
            bool found = lookup_bucket(t_packet_cache.kmer_idx, t_packet_cache.bucket_idx,
                                       &offset_start, &entry_count, &t_bucket_cache);
            
            if (found && entry_count > 0) {
                uint32_t total_entries = entry_count;
                if (total_entries > MAX_DPU_SEQS) total_entries = MAX_DPU_SEQS;
                
                uint32_t cached_aligned_start = 0xFFFFFFFF;
                
                for (uint32_t e = 0; e < total_entries; ++e) {
                     uint32_t global_idx = offset_start + e;
                     // 8-byte aligned fetch
                     uint32_t raw_fetch = offset_start + (e >> 5 << 5); 
                     uint32_t aligned_fetch = raw_fetch & ~1;
                     
                     if (aligned_fetch != cached_aligned_start) {
                         cached_aligned_start = aligned_fetch;
                         uint32_t count = ENTRY_CACHE_SIZE;
                         // Don't read past end if entries are contiguous
                         // But for simplicity/safety with alignment, fixed size is fine.
                         mram_read(&m_entries[aligned_fetch], t_entry_cache, count << 2);
                     }
                     
                     uint32_t c_idx = global_idx - cached_aligned_start;
                     uint16_t tid = t_entry_cache[c_idx].local_target_id;
                     if (tid >= MAX_DPU_SEQS) continue;

                     int16_t tpos = t_entry_cache[c_idx].pos_j;
                     int16_t diag = (int16_t)t_packet_cache.query_pos - (int16_t)tpos;
                     uint8_t diag_u8 = (uint8_t)diag;
                     
                     // State Update
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