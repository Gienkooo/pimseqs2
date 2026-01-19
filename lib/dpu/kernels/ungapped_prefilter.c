#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include <stdio.h>
#include <string.h>

#include "DpuSharedTypes.h"

#define AA_NUMBER 21
#define MRAM_MAX 2048
#define LONG_TARGET_LEN 4096
#define LONG_QUERY_LEN 2048
#define WRAM_CACHE 57344

#define MRAM_ALIGN_SIZE(x) (((x) + 7) & ~7U) // 8 * ceil(x/8)
#define XAA(x) (((x) << 4) + ((x) << 2) + (x))

BARRIER_INIT(barrier, NR_TASKLETS);

__dma_aligned BatchDescriptor batch_descriptor;
__dma_aligned QueryMetadata query_meta;
__dma_aligned TargetMetadata target_meta;
__dma_aligned TargetMetadata next_meta;

__dma_aligned uint8_t wram[WRAM_CACHE];
uintptr_t mram_base;

uint8_t* target_global;
int8_t* pssm_global;
uint32_t target_len;
uint32_t pssm_len;

typedef struct {
    int16_t score;
    int16_t diagonal;
} TaskletResult;

TaskletResult tasklet_results[NR_TASKLETS];


static void* load_mram(__mram_ptr void* raw_mram_addr, void* wram_buffer, uint32_t aligned_size) {
    uintptr_t addr_val = (uintptr_t)raw_mram_addr;
    uintptr_t aligned_mram_addr = addr_val & ~7U;
    uint32_t offset = addr_val & 7U;
    uint32_t read_offset = 0;
    uint8_t* buffer_bytes = (uint8_t*)wram_buffer;
    for (; read_offset + MRAM_MAX < aligned_size; read_offset += MRAM_MAX) {
        mram_read((__mram_ptr void*)(aligned_mram_addr + read_offset), buffer_bytes + read_offset, MRAM_MAX);
    } mram_read((__mram_ptr void*)(aligned_mram_addr + read_offset), buffer_bytes + read_offset, aligned_size - read_offset);
    return (void*)(buffer_bytes + offset);
}

inline void store_hit(uint32_t i, uint32_t j) {
    int16_t best_score = 0;
    int16_t best_diag = 0;
    for (int k = 0; k < NR_TASKLETS; k++) {
        if (tasklet_results[k].score > best_score) {
            best_score = tasklet_results[k].score;
            best_diag = tasklet_results[k].diagonal;
        }
    }
    Hit hit;
    hit.target_id = target_meta.target_id;
    hit.query_id = query_meta.query_id;
    hit.score = best_score;
    hit.diagonal = best_diag;
    hit.pad1 = 0;
    hit.pad2 = 0;

    uintptr_t res_addr = mram_base + batch_descriptor.results_offset + ((i * batch_descriptor.num_targets + j) * sizeof(Hit));
    mram_write(&hit, (__mram_ptr void*)res_addr, MRAM_ALIGN_SIZE(sizeof(Hit)));

    printf("DPU[%u] Q=%u T=%u S=%d D=%d\n", me(), i, target_meta.target_id, best_score, best_diag);
}


static void align_short(uint32_t tasklet_id, uint8_t* target, int8_t* pssm, uint32_t t_len, uint32_t q_len) {
    int16_t max_score = 0;
    int16_t best_diag = 0;
    uint32_t num_diags = q_len + t_len - 1;

    // Iterate over diagonals
    for (uint32_t diag_idx = tasklet_id-1; diag_idx < num_diags; diag_idx += NR_TASKLETS-1) {
        int32_t delta = (int32_t)diag_idx - (int32_t)q_len + 1;
        int32_t q_start = 0;
        int32_t q_end = q_len;
        if (-delta > q_start) q_start = -delta;
        if ((int32_t)t_len - delta < q_end) q_end = (int32_t)t_len - delta;

        // Iterate over diagonal
        int16_t diag_score = 0;
        int16_t score = 0;
        for (int32_t q = q_start; q < q_end; ++q) {
            int32_t t = q + delta;
            uint8_t aa = target[t];
            if (aa >= AA_NUMBER) aa = 20;
            int16_t val = (int16_t) pssm[XAA(q) + aa];
            score += val;
            if (score < 0) score = 0;
            if (score > diag_score) diag_score = score;
        }

        // Update Tasklet Best
        if (diag_score > max_score) {
            max_score = diag_score;
            best_diag = (int16_t)((int32_t)(q_len - 1) - (int32_t)diag_idx); 
        }
    }
    tasklet_results[tasklet_id].score = max_score;
    tasklet_results[tasklet_id].diagonal = best_diag;
}

static void align_long_t(uint32_t tasklet_id, uint8_t* target, int8_t* pssm, uint32_t t_len, uint32_t q_len, int32_t t_start, int16_t* buffer) {
    int16_t max_score = 0;
    int16_t best_diag = 0;

    int32_t diag_start = 1 - (int32_t)q_len;
    int32_t diag_end = (int32_t)t_len;

    for (int32_t delta = diag_start + (int32_t)tasklet_id-1; delta < diag_end; delta += NR_TASKLETS-1) {
        int16_t diag_score = 0;
        int16_t score = 0;

        int32_t read_idx = ((int32_t)q_len + delta - 1);
        int32_t write_idx = ((int32_t)q_len-(int32_t)t_len+delta-1);
        uint8_t write = 0;

        int32_t q_start = 0;
        int32_t q_end = (int32_t)q_len;

        if (- delta >= q_start) {
            score = buffer[read_idx];
            q_start = -delta;
        }

        if ((int32_t)t_len - delta < (int32_t)q_len) {
            write = 1;
            q_end = t_len - delta;
        }

        for (int32_t q = q_start; q < q_end; ++q) {
            int32_t t = q + delta;
            uint8_t aa = target[t];
            if (aa >= AA_NUMBER) aa = 20;
            int16_t val = (int16_t)pssm[XAA(q) + aa];
            score += val;
            if (score < 0) score = 0;
            if (score > diag_score) diag_score = score;
        }

        if (diag_score > max_score) {
            max_score = diag_score;
            best_diag = (int16_t)(-t_start-delta);
        }
        if (write) {
            buffer[write_idx] = score;
        }
    }
 
    if (max_score > tasklet_results[tasklet_id].score) {
        tasklet_results[tasklet_id].score = max_score;
        tasklet_results[tasklet_id].diagonal = best_diag;
    }
}

static void align_long_q(uint32_t tasklet_id, uint8_t* target, int8_t* pssm, uint32_t t_len, uint32_t q_len, int32_t q_start, int16_t* buffer) {
    int16_t max_score = 0;
    int16_t best_diag = 0;

    int32_t diag_start = 1 - (int32_t)q_len;
    int32_t diag_end = (int32_t)t_len;

    for (int32_t delta = diag_end - 1 - (int32_t)tasklet_id; delta >= diag_start; delta -= NR_TASKLETS) {
        int16_t diag_score = 0;
        int16_t score = 0;

        int32_t read_idx = (delta - 1);
        int32_t write_idx = ((int32_t)q_len-delta-1);
        uint8_t write = 0;

        int32_t t_start = delta > 0 ? delta : 0;
        int32_t t_end = t_len;

        if (delta > 0) {
            t_start = delta;
            score = buffer[read_idx];
        }

        if ((int32_t)t_len - delta >= (int32_t)q_len) {
            write = 1;
            t_end = delta + (int32_t)q_len;
        }

        for (int32_t t = t_start; t < t_end; ++t) {
            int32_t q = t - delta;
            uint8_t aa = target[t];
            if (aa >= AA_NUMBER) aa = 20;
            int16_t val = (int16_t)pssm[XAA(q) + aa];
            score += val;
            if (score < 0) score = 0;
            if (score > diag_score) diag_score = score;
        }

        if (diag_score > max_score) {
            max_score = diag_score;
            best_diag = (int16_t)(q_start-delta);
        }
        if (write) {
            buffer[write_idx] = score;
        }
    }
 
    if (max_score > tasklet_results[tasklet_id].score) {
        tasklet_results[tasklet_id].score = max_score;
        tasklet_results[tasklet_id].diagonal = best_diag;
    }
}

static void align_long_qt(uint32_t tasklet_id, uintptr_t mram_base, int8_t* pssm_buffer, uint8_t* target_buffer, int16_t* band_scores, int16_t* band_max_scores) {
    uint32_t t_len = target_meta.target_len;
    uint32_t q_len = query_meta.query_len;
    uint32_t max_t_chunk_len = LONG_TARGET_LEN;
    uint32_t max_q_chunk_len = LONG_QUERY_LEN;

    int32_t min_diag = 1 - (int32_t)q_len;
    int32_t max_diag = (int32_t)t_len - 1;
    int32_t max_band_width = (int32_t)max_t_chunk_len - (int32_t)max_q_chunk_len + 1;

    int16_t my_max_score = 0;
    int16_t my_best_diag = 0;

    // begin band
    for (int32_t band_start_diag = min_diag; band_start_diag <= max_diag; band_start_diag += max_band_width) {
        
        int32_t band_width = max_band_width;
        if (max_diag - band_start_diag + 1 < band_width) {
            band_width = max_diag - band_start_diag + 1;
        }
        int32_t band_end_diag = band_start_diag + band_width - 1;

        for (int32_t k = tasklet_id; k < band_width; k += NR_TASKLETS) {
            band_scores[k] = 0;
            band_max_scores[k] = 0;
        }
        barrier_wait(&barrier);

        // begin chunk
        for (uint32_t q_start = 0; q_start < q_len; q_start += max_q_chunk_len) {

            uint32_t q_chunk_len = max_q_chunk_len;
            if (q_len - q_start < q_chunk_len) q_chunk_len = q_len - q_start;

            // load query
            if (tasklet_id == 0) {
                uint32_t aligned_size = MRAM_ALIGN_SIZE(q_chunk_len * AA_NUMBER);
                uintptr_t src_addr = mram_base + batch_descriptor.pssm_data_offset + query_meta.pssm_offset_in_batch + (q_start * AA_NUMBER);
                pssm_global = load_mram(src_addr, pssm_buffer, aligned_size);
            }
            barrier_wait(&barrier);

            int32_t t_idx_min = (int32_t)q_start + band_start_diag;
            int32_t t_idx_max = ((int32_t)q_start + (int32_t)q_chunk_len - 1) + band_end_diag;
            int32_t t_start = t_idx_min > 0 ? t_idx_min : 0;
            int32_t t_end = t_idx_max < (int32_t)t_len - 1 ? t_idx_max : (int32_t)t_len - 1;
            if (t_start > t_end) {
                barrier_wait(&barrier); 
                continue;
            }

            // load target
            int32_t t_chunk_len = t_end - t_start + 1;
            if (tasklet_id == 0) {
                uint32_t aligned_size = MRAM_ALIGN_SIZE(t_chunk_len);
                uintptr_t src_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data + t_start;
                target_global = load_mram(src_addr, target_buffer, aligned_size);
            }
            barrier_wait(&barrier);

            // compute scores
            int8_t* pssm_chunk = pssm_global;
            uint8_t* target_chunk = target_global;
            for (uint32_t i = 0; i < q_chunk_len; ++i) {
                int32_t q_global = q_start + i;
                uint32_t q_offset = i * AA_NUMBER;
                for (int32_t k = tasklet_id; k < band_width; k += NR_TASKLETS) {
                    int32_t diag = band_start_diag + k;
                    int32_t t_global = q_global + diag;
                    if (t_global >= 0 && t_global < (int32_t)t_len) {
                        int32_t t_local = t_global - t_start;
                        if (t_local >= 0 && t_local < t_chunk_len) {
                            uint8_t t_val = target_chunk[t_local];
                            if (t_val >= AA_NUMBER) t_val = 20;
                            int16_t match_score = (int16_t)pssm_chunk[q_offset + t_val];
                            int16_t score = band_scores[k] + match_score;
                            if (score < 0) score = 0;
                            band_scores[k] = score;
                            if (score > band_max_scores[k]) {
                                band_max_scores[k] = score;
                            }
                        }
                    }
                }
            }
        } // end chunk

        for (int32_t k = tasklet_id; k < band_width; k += NR_TASKLETS) {
            if (band_max_scores[k] > my_max_score) {
                my_max_score = band_max_scores[k];
                my_best_diag = (int16_t)(- band_start_diag - k);
            }
        }
        barrier_wait(&barrier);

    } // end band

    if (my_max_score > tasklet_results[tasklet_id].score) {
        tasklet_results[tasklet_id].score = my_max_score;
        tasklet_results[tasklet_id].diagonal = my_best_diag;
    }
}

int main() {
    uint32_t tasklet_id = me();

    // Load batch descriptor
    if (tasklet_id == 0) {
        mram_base = (uintptr_t)__sys_used_mram_end;
        mram_read((__mram_ptr void*)mram_base, &batch_descriptor, MRAM_ALIGN_SIZE(sizeof(BatchDescriptor)));
        printf("DPU[%u] Batch: Targets=%u QLen=%u\n", me(), batch_descriptor.num_targets, batch_descriptor.query_len);
    }
    barrier_wait(&barrier);

    int8_t* pssm_buffer = (int8_t*)wram; // [0-42]
    uint8_t* target_buffer = (uint8_t*)(wram + (AA_NUMBER * LONG_QUERY_LEN)); // [42-46]
    int16_t* score_buffer = (int16_t*)(wram + (AA_NUMBER * LONG_QUERY_LEN) + LONG_TARGET_LEN); // [46-54]
    int16_t* next_buffer = (int16_t*)(wram + WRAM_CACHE - LONG_TARGET_LEN); // [50-54]

    // Iterate over queries
    for (uint32_t i = 0; i < batch_descriptor.num_queries; ++i) {

        if (tasklet_id == 0) {
            // Load query metadata
            uintptr_t query_meta_addr = mram_base + batch_descriptor.queries_metadata_offset + (i * sizeof(QueryMetadata));
            mram_read((__mram_ptr void*)query_meta_addr, &query_meta, MRAM_ALIGN_SIZE(sizeof(QueryMetadata)));
        }
        barrier_wait(&barrier);      
  
        // SHORT QUERY
        if (query_meta.query_len <= LONG_QUERY_LEN) {

            // Load query
            uint32_t aligned_pssm_size = MRAM_ALIGN_SIZE(query_meta.query_len * AA_NUMBER);
            uintptr_t pssm_addr = mram_base + batch_descriptor.pssm_data_offset + query_meta.pssm_offset_in_batch;
            uintptr_t aligned_pssm_addr = pssm_addr & ~7U;
            uint32_t offset = pssm_addr & 7U;
            uint32_t pssm_read_offset = 0;
            for(;pssm_read_offset + MRAM_MAX < aligned_pssm_size; pssm_read_offset += MRAM_MAX) {
                mram_read((__mram_ptr void*)(aligned_pssm_addr + pssm_read_offset), pssm_buffer + pssm_read_offset, MRAM_MAX);
            } mram_read((__mram_ptr void*)(aligned_pssm_addr + pssm_read_offset), pssm_buffer + pssm_read_offset, aligned_pssm_size - pssm_read_offset);
            pssm_global = pssm_buffer + offset;
            barrier_wait(&barrier);

            // Iterate over targets
            uint8_t* current_buffer = target_buffer;
            uint8_t* prefetch_buffer = next_buffer;
            bool init = true;
            for (uint32_t j = 0; j < batch_descriptor.num_targets; j += 1) {

                // Load Target & Metadata
                if(tasklet_id == 0) {
                    if (!init) {
                        target_meta = next_meta;
                    } else {
                        init = false;
                        uintptr_t target_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + (j * sizeof(TargetMetadata));
                        mram_read((__mram_ptr void*)target_meta_addr, &target_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
                        uint32_t load_len = target_meta.target_len > LONG_TARGET_LEN ? LONG_TARGET_LEN : target_meta.target_len;
                        uint32_t aligned_target_size = MRAM_ALIGN_SIZE(load_len);
                        uintptr_t target_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data;
                        target_global = load_mram(target_addr, current_buffer, aligned_target_size);
                    }
                }
                barrier_wait(&barrier);

                // Initialize results
                tasklet_results[tasklet_id].score = 0;
                tasklet_results[tasklet_id].diagonal = 0;

                // SHORT TARGET
                if (target_meta.target_len <= LONG_TARGET_LEN) {

                    // Compute score and diagonal
                    if (tasklet_id != 0) {
                        align_short(tasklet_id, target_global, pssm_global, target_meta.target_len, query_meta.query_len);
                    } else {
                        if (j + 1 < batch_descriptor.num_targets) {
                            uintptr_t next_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + ((j + 1) * sizeof(TargetMetadata));
                            mram_read((__mram_ptr void*)next_meta_addr, &next_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
                            uint32_t next_len = next_meta.target_len > LONG_TARGET_LEN ? LONG_TARGET_LEN : next_meta.target_len;
                            uint32_t aligned_next_size = MRAM_ALIGN_SIZE(next_len);
                            uintptr_t next_data_addr = mram_base + batch_descriptor.targets_data_offset + next_meta.offset_in_data;
                            load_mram(next_data_addr, prefetch_buffer, aligned_next_size);
                        }
                    }
                    barrier_wait(&barrier);
                }
                // LONG TARGET
                else {
                    bool is_last_chunk = false;
                    for(int32_t idx = tasklet_id; idx < LONG_QUERY_LEN; idx += NR_TASKLETS) score_buffer[idx] = 0;
                    for(int32_t target_start = 0; target_start < target_meta.target_len; target_start += LONG_TARGET_LEN) {

                        // Compute score and diagonal
                        uint32_t next_start = target_start + LONG_TARGET_LEN;
                        is_last_chunk = (next_start >= target_meta.target_len);
                        if (tasklet_id != 0) {
                            align_long_t(tasklet_id, target_global, pssm_global, target_meta.target_len, query_meta.query_len, target_start, score_buffer);
                        } else if(!is_last_chunk) {
                            uint32_t remaining = target_meta.target_len - next_start;
                            uint32_t load_len = remaining > LONG_TARGET_LEN ? LONG_TARGET_LEN : remaining;
                            uintptr_t t_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data + next_start;
                            load_mram(t_addr, prefetch_buffer, MRAM_ALIGN_SIZE(load_len));
                        } else if (j + 1 < batch_descriptor.num_targets) {
                            uintptr_t next_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + ((j + 1) * sizeof(TargetMetadata));
                            mram_read((__mram_ptr void*)next_meta_addr, &next_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
                            uint32_t next_len = next_meta.target_len > LONG_TARGET_LEN ? LONG_TARGET_LEN : next_meta.target_len;
                            uint32_t aligned_next_size = MRAM_ALIGN_SIZE(next_len);
                            uintptr_t next_data_addr = mram_base + batch_descriptor.targets_data_offset + next_meta.offset_in_data;
                            load_mram(next_data_addr, prefetch_buffer, aligned_next_size);
                        }
                        barrier_wait(&barrier);
                        if (!is_last_chunk) {
                            if (tasklet_id == 0) {
                                uint8_t* temp = current_buffer;
                                current_buffer = prefetch_buffer;
                                prefetch_buffer = temp;
                                uintptr_t next_chunk_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data + next_start;
                                uint32_t offset = next_chunk_addr & 7U;
                                target_global = (void*)(current_buffer + offset);
                            }
                            barrier_wait(&barrier);
                        }
                    }
                }

                // Store score and diagonal
                if (tasklet_id == 0) {
                    store_hit(i, j);

                    uint8_t* temp = current_buffer;
                    current_buffer = prefetch_buffer;
                    prefetch_buffer = temp;
                    uintptr_t next_data_addr = mram_base + batch_descriptor.targets_data_offset + next_meta.offset_in_data;
                    uint32_t offset = next_data_addr & 7U;
                    target_global = (void*)(current_buffer + offset);
                }

                barrier_wait(&barrier);
            }
        }
        // LONG QUERY
        else {
            // Iterate over targets
            for (uint32_t j = 0; j < batch_descriptor.num_targets; j += 1) {

                // Load Target Metadata
                if(tasklet_id == 0) {
                    uintptr_t target_meta_addr = mram_base + batch_descriptor.targets_metadata_offset + (j * sizeof(TargetMetadata));
                    mram_read((__mram_ptr void*)target_meta_addr, &target_meta, MRAM_ALIGN_SIZE(sizeof(TargetMetadata)));
                }
                barrier_wait(&barrier);
                
                // Initialize results
                tasklet_results[tasklet_id].score = 0;
                tasklet_results[tasklet_id].diagonal = 0;

                // SHORT TARGET
                if (target_meta.target_len <= LONG_TARGET_LEN) {

                    // Load Target Sequence
                    if (tasklet_id == 0) {
                        uint32_t aligned_target_size = MRAM_ALIGN_SIZE(target_meta.target_len);
                        uintptr_t target_addr = mram_base + batch_descriptor.targets_data_offset + target_meta.offset_in_data;
                        target_global = load_mram(target_addr, target_buffer, aligned_target_size);
                    }
                    barrier_wait(&barrier);

                    for(int32_t idx = tasklet_id; idx < LONG_TARGET_LEN; idx += NR_TASKLETS) score_buffer[idx] = 0;
                    for(int32_t query_start = 0; query_start < query_meta.query_len; query_start += LONG_QUERY_LEN) {
                        // Load query
                        if (tasklet_id == 0) {
                            uint32_t aligned_pssm_size = query_start + LONG_QUERY_LEN > query_meta.query_len ?
                                MRAM_ALIGN_SIZE((query_meta.query_len - query_start) * AA_NUMBER) : MRAM_ALIGN_SIZE(LONG_QUERY_LEN * AA_NUMBER);
                            if (aligned_pssm_size > LONG_QUERY_LEN * AA_NUMBER) aligned_pssm_size = LONG_QUERY_LEN * AA_NUMBER;
                            uintptr_t pssm_addr = mram_base + batch_descriptor.pssm_data_offset + query_meta.pssm_offset_in_batch + query_start * AA_NUMBER;
                            pssm_global = load_mram(pssm_addr, pssm_buffer, aligned_pssm_size);
                            pssm_len = aligned_pssm_size / AA_NUMBER;
                        }
                        barrier_wait(&barrier);

                        align_long_q(tasklet_id, target_global, pssm_global, target_meta.target_len, pssm_len, query_start, score_buffer);
                        barrier_wait(&barrier);
                    }                    
                }
                // LONG TARGET
                else {
                    int8_t* pssm_buf = (int8_t*)wram; 
                    uint8_t* target_buf = (uint8_t*)(wram + (AA_NUMBER * LONG_QUERY_LEN)); 
                    int16_t* band_scores = (int16_t*)(wram + (AA_NUMBER * LONG_QUERY_LEN) + LONG_TARGET_LEN);
                    int16_t* band_max_scores = (int16_t*)(wram + (AA_NUMBER * LONG_QUERY_LEN) + LONG_TARGET_LEN + 2*(LONG_TARGET_LEN - LONG_QUERY_LEN));
                    align_long_qt(tasklet_id, mram_base, pssm_buf, target_buf, band_scores, band_max_scores);
                    barrier_wait(&barrier);
                }

                // Store score and diagonal
                if (tasklet_id == 0) { store_hit(i, j); }

                barrier_wait(&barrier);
            }
        }
    }
    return 0;
}
