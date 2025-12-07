#include "DpuGroupManager.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "Debug.h"

namespace mmseqs::dpu {

DpuGroupManager::DpuGroupManager(uint32_t num_groups, uint32_t dpus_per_group)
    : num_groups_(num_groups),
      dpus_per_group_(dpus_per_group),
      dpu_sets_(num_groups),
      statuses_(num_groups, GroupStatus::IDLE),
      contexts_(num_groups) {
    
    if (num_groups == 0) {
        Debug(Debug::ERROR) << "[DPU GROUP] Must have at least 1 group\n";
        exit(EXIT_FAILURE);
    }
    
    // If dpus_per_group not specified, try to allocate evenly from available DPUs
    if (dpus_per_group_ == 0) {
        // First, find how many DPUs are available total
        struct dpu_set_t probe_set;
        dpu_error_t status = dpu_alloc(DPU_ALLOCATE_ALL, NULL, &probe_set);
        if (status != DPU_OK) {
            Debug(Debug::ERROR) << "[DPU GROUP] Cannot probe available DPUs: " << dpu_error_to_string(status) << "\n";
            exit(EXIT_FAILURE);
        }
        
        uint32_t total_available;
        dpu_get_nr_dpus(probe_set, &total_available);
        dpu_free(probe_set);
        
        dpus_per_group_ = std::max(1u, total_available / num_groups);
        Debug(Debug::INFO) << "[DPU GROUP] Auto-detected " << total_available << " DPUs, "
                          << "using " << dpus_per_group_ << " per group\n";
    }
    
    // Allocate each group separately
    for (uint32_t g = 0; g < num_groups; ++g) {
        dpu_error_t status = dpu_alloc(dpus_per_group_, NULL, &dpu_sets_[g]);
        if (status != DPU_OK) {
            Debug(Debug::ERROR) << "[DPU GROUP] Failed to allocate group " << g << ": "
                                 << dpu_error_to_string(status) << "\n";
            // Free already allocated groups
            for (uint32_t prev = 0; prev < g; ++prev) {
                dpu_free(dpu_sets_[prev]);
            }
            exit(EXIT_FAILURE);
        }
        
        uint32_t actual_dpus;
        dpu_get_nr_dpus(dpu_sets_[g], &actual_dpus);
        
        // Update dpus_per_group_ to actual count (first group sets the standard)
        if (g == 0) {
            dpus_per_group_ = actual_dpus;
        }
        
        Debug(Debug::INFO) << "[DPU GROUP] Allocated group " << g << " with " << actual_dpus << " DPUs\n";
    }
    
    Debug(Debug::INFO) << "[DPU GROUP] Initialized " << num_groups_ << " groups × " << dpus_per_group_ << " DPUs = " << getTotalDpus() << "\n";
}

DpuGroupManager::~DpuGroupManager() {
    // Wait for any executing groups
    syncAllGroups();
    
    // Free all groups
    for (uint32_t g = 0; g < num_groups_; ++g) {
        dpu_free(dpu_sets_[g]);
    }
    
    Debug(Debug::INFO) << "[DPU GROUP] Released " << num_groups_ << " groups\n";
}

void DpuGroupManager::loadKernel(const char* kernel_binary_path) {
    // Wait for all groups to be idle before loading new kernel
    syncAllGroups();
    
    for (uint32_t g = 0; g < num_groups_; ++g) {
        dpu_error_t status = dpu_load(dpu_sets_[g], kernel_binary_path, NULL);
        checkStatus(status, "Kernel load", g);
    }
    
    Debug(Debug::INFO) << "[DPU GROUP] Loaded kernel '" << kernel_binary_path << "' to all " << num_groups_ << " groups\n";
}

DpuGroupManager::GroupStatus DpuGroupManager::getGroupStatus(uint32_t group_id) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        return GroupStatus::FAULT;
    }
    
    // For EXECUTING groups, poll to check if done
    if (statuses_[group_id] == GroupStatus::EXECUTING) {
        bool done = false;
        bool fault = false;
        dpu_error_t status = dpu_status(dpu_sets_[group_id], &done, &fault);
        
        if (status != DPU_OK) {
            Debug(Debug::ERROR) << "[DPU GROUP] Failed to poll group " << group_id << ": "
                                 << dpu_error_to_string(status) << "\n";
            statuses_[group_id] = GroupStatus::FAULT;
        } else if (fault) {
            Debug(Debug::ERROR) << "[DPU GROUP] Group " << group_id << " faulted\n";
            statuses_[group_id] = GroupStatus::FAULT;
            // Try to get logs for debugging
            readGroupLogs(group_id);
        } else if (done) {
            statuses_[group_id] = GroupStatus::COMPLETED;
        }
    }
    
    return statuses_[group_id];
}

uint32_t DpuGroupManager::findIdleGroup() {
    for (uint32_t g = 0; g < num_groups_; ++g) {
        if (getGroupStatus(g) == GroupStatus::IDLE) {
            return g;
        }
    }
    return UINT32_MAX;
}

uint32_t DpuGroupManager::findCompletedGroup() {
    for (uint32_t g = 0; g < num_groups_; ++g) {
        if (getGroupStatus(g) == GroupStatus::COMPLETED) {
            return g;
        }
    }
    return UINT32_MAX;
}

void DpuGroupManager::pollAllGroups() {
    for (uint32_t g = 0; g < num_groups_; ++g) {
        getGroupStatus(g);  // Updates status for EXECUTING groups
    }
}

void DpuGroupManager::scatterToGroup(uint32_t group_id, uint32_t dpu_idx_in_group,
                                    const void* data, uint32_t size, uint32_t mram_offset) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (statuses_[group_id] != GroupStatus::IDLE) {
        Debug(Debug::ERROR) << "[DPU GROUP] Cannot scatter to non-idle group " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (mram_offset % MRAM_ALIGN != 0 || size % MRAM_ALIGN != 0) {
        Debug(Debug::ERROR) << "[DPU GROUP] MRAM offset/size not 8-byte aligned\n";
        exit(EXIT_FAILURE);
    }
    
    uint32_t idx = 0;
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_sets_[group_id], dpu) {
        if (idx == dpu_idx_in_group) {
            dpu_error_t status = dpu_copy_to(dpu, "__sys_used_mram_end", mram_offset,
                                             (void*)data, size);
            checkStatus(status, "Scatter", group_id);
            return;
        }
        idx++;
    }
    
    Debug(Debug::ERROR) << "[DPU GROUP] DPU index " << dpu_idx_in_group << " out of range for group " << group_id << "\n";
    exit(EXIT_FAILURE);
}

void DpuGroupManager::broadcastToGroup(uint32_t group_id, const void* data,
                                      uint32_t size, uint32_t mram_offset) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (statuses_[group_id] != GroupStatus::IDLE) {
        Debug(Debug::ERROR) << "[DPU GROUP] Cannot broadcast to non-idle group " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (mram_offset % MRAM_ALIGN != 0 || size % MRAM_ALIGN != 0) {
        Debug(Debug::ERROR) << "[DPU GROUP] MRAM offset/size not 8-byte aligned\n";
        exit(EXIT_FAILURE);
    }
    
    // Broadcast to all DPUs in the group
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_sets_[group_id], dpu) {
        dpu_error_t status = dpu_copy_to(dpu, "__sys_used_mram_end", mram_offset,
                                          (void*)data, size);
        checkStatus(status, "Broadcast", group_id);
    }
}

void DpuGroupManager::launchGroupAsync(uint32_t group_id, const GroupContext& context) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (statuses_[group_id] != GroupStatus::IDLE) {
        Debug(Debug::ERROR) << "[DPU GROUP] Cannot launch non-idle group " << group_id << " (status=" << (int)statuses_[group_id] << ")\n";
        exit(EXIT_FAILURE);
    }
    
    contexts_[group_id] = context;
    
    dpu_error_t status = dpu_launch(dpu_sets_[group_id], DPU_ASYNCHRONOUS);
    checkStatus(status, "Async launch", group_id);
    
    statuses_[group_id] = GroupStatus::EXECUTING;
}

void DpuGroupManager::gatherFromGroup(uint32_t group_id, uint32_t dpu_idx_in_group,
                                     void* buffer, uint32_t size, uint32_t mram_offset) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    // Allow gathering from COMPLETED or IDLE (for inspection)
    if (statuses_[group_id] == GroupStatus::EXECUTING) {
        Debug(Debug::ERROR) << "[DPU GROUP] Cannot gather from executing group " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (mram_offset % MRAM_ALIGN != 0 || size % MRAM_ALIGN != 0) {
        Debug(Debug::ERROR) << "[DPU GROUP] MRAM offset/size not 8-byte aligned\n";
        exit(EXIT_FAILURE);
    }
    
    uint32_t idx = 0;
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_sets_[group_id], dpu) {
        if (idx == dpu_idx_in_group) {
            dpu_error_t status = dpu_copy_from(dpu, "__sys_used_mram_end", mram_offset,
                                                buffer, size);
            checkStatus(status, "Gather", group_id);
            return;
        }
        idx++;
    }
    
    Debug(Debug::ERROR) << "[DPU GROUP] DPU index " << dpu_idx_in_group << " out of range for group " << group_id << "\n";
    exit(EXIT_FAILURE);
}

void DpuGroupManager::releaseGroup(uint32_t group_id) {
    if (group_id >= num_groups_) {
        Debug(Debug::ERROR) << "[DPU GROUP] Invalid group ID " << group_id << "\n";
        exit(EXIT_FAILURE);
    }
    
    if (statuses_[group_id] == GroupStatus::EXECUTING) {
        // Wait for it to complete first
        dpu_error_t status = dpu_sync(dpu_sets_[group_id]);
        checkStatus(status, "Sync before release", group_id);
    }
    
    statuses_[group_id] = GroupStatus::IDLE;
    contexts_[group_id] = GroupContext{};  // Clear context
}

void DpuGroupManager::syncAllGroups() {
    for (uint32_t g = 0; g < num_groups_; ++g) {
        if (statuses_[g] == GroupStatus::EXECUTING) {
            dpu_error_t status = dpu_sync(dpu_sets_[g]);
            if (status != DPU_OK) {
                Debug(Debug::WARNING) << "[DPU GROUP] Sync failed for group " << g << ": " << dpu_error_to_string(status) << "\n";
                statuses_[g] = GroupStatus::FAULT;
            } else {
                statuses_[g] = GroupStatus::COMPLETED;
            }
        }
    }
}

void DpuGroupManager::readGroupLogs(uint32_t group_id) {
    if (group_id >= num_groups_) return;
    
    Debug(Debug::INFO) << "[DPU GROUP] Logs for group " << group_id << ":\n";
    struct dpu_set_t dpu;
    uint32_t idx = 0;
    DPU_FOREACH(dpu_sets_[group_id], dpu) {
        Debug(Debug::INFO) << "  DPU " << idx << ":\n";
        dpu_error_t status = dpu_log_read(dpu, stderr);
        if (status != DPU_OK) {
            Debug(Debug::WARNING) << "    (failed to read log: " << dpu_error_to_string(status) << ")\n";
        }
        idx++;
    }
}

void DpuGroupManager::checkStatus(dpu_error_t status, const char* context, uint32_t group_id) {
    if (status != DPU_OK) {
        Debug(Debug::ERROR) << "[DPU GROUP] FATAL: " << context << " failed for group " << group_id << ": "
                           << dpu_error_to_string(status) << "\n";
        exit(EXIT_FAILURE);
    }
}

}  // namespace mmseqs::dpu
