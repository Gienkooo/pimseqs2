#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <dpu.h>
#ifdef __cplusplus
}
#endif

#include <cstdint>
#include <vector>
#include <cstring>
#include <functional>
#include <atomic>

namespace mmseqs::dpu {

/**
 * DpuGroupManager - Manages multiple independent DPU groups for pipelined execution.
 *
 * Each group wraps one or more DPUs (dpu_set_t) to support independent scatter/launch
 * and gather operations. Use this to pipeline work over multiple DPU groups.
 */
class DpuGroupManager {
public:
    // Status of each DPU group
    enum class GroupStatus {
        IDLE,        // Ready for new work
        EXECUTING,   // DPUs are running
        COMPLETED,   // Execution done, results ready to gather
        FAULT        // Error occurred
    };
    
    // Context for work item assigned to a group
    struct GroupContext {
        uint32_t query_id;           // User-provided ID for tracking
        uint32_t results_offset;     // MRAM offset for results
        uint32_t results_size;       // Size of results buffer
        uint32_t num_targets;        // Number of targets assigned
        // Additional user data can be added here
    };

    /**
     * @param num_groups Number of independent DPU groups to create
     * @param dpus_per_group Number of DPUs in each group (0 = auto)
     */
    explicit DpuGroupManager(uint32_t num_groups, uint32_t dpus_per_group = 0);
    ~DpuGroupManager();
    
    // Disable copy
    DpuGroupManager(const DpuGroupManager&) = delete;
    DpuGroupManager& operator=(const DpuGroupManager&) = delete;

    uint32_t getNumGroups() const { return num_groups_; }
    uint32_t getDpusPerGroup() const { return dpus_per_group_; }
    uint32_t getTotalDpus() const { return num_groups_ * dpus_per_group_; }

    // Load a kernel binary to ALL groups. Call before launching any work.
    void loadKernel(const char* kernel_binary_path);

    // Get the current status of a group.
    GroupStatus getGroupStatus(uint32_t group_id);

    // Find an idle group; returns UINT32_MAX when none is available.
    uint32_t findIdleGroup();

    // Find a completed group; returns UINT32_MAX when none is available.
    uint32_t findCompletedGroup();

    // Poll executing groups and update their status.
    void pollAllGroups();

    // Scatter data to a specific DPU within a group. Group must be IDLE.
    void scatterToGroup(uint32_t group_id, uint32_t dpu_idx_in_group,
                       const void* data, uint32_t size, uint32_t mram_offset);
    
    // Broadcast data to all DPUs in a group. Group must be IDLE.
    void broadcastToGroup(uint32_t group_id, const void* data,
                         uint32_t size, uint32_t mram_offset);

    // Launch all DPUs in a group asynchronously; group becomes EXECUTING.
    void launchGroupAsync(uint32_t group_id, const GroupContext& context);

    // Gather results from a specific DPU in a group. Group must be COMPLETED.
    void gatherFromGroup(uint32_t group_id, uint32_t dpu_idx_in_group,
                        void* buffer, uint32_t size, uint32_t mram_offset);

    // Mark a completed group as idle. Call after gathering results.
    void releaseGroup(uint32_t group_id);

    // Get the context for a group (set during launchGroupAsync).
    const GroupContext& getGroupContext(uint32_t group_id) const { 
        return contexts_[group_id]; 
    }

    // Wait for ALL groups to complete.
    void syncAllGroups();

    // Read DPU logs from a group for debugging.
    void readGroupLogs(uint32_t group_id);

    // MRAM alignment helpers (same as DpuCommunicationManager)
    static constexpr uint32_t MRAM_ALIGN = 8;
    static inline uint32_t alignToMram(uint32_t size) {
        return ((size + MRAM_ALIGN - 1) / MRAM_ALIGN) * MRAM_ALIGN;
    }

private:
    uint32_t num_groups_;
    uint32_t dpus_per_group_;
    
    std::vector<struct dpu_set_t> dpu_sets_;
    std::vector<GroupStatus> statuses_;
    std::vector<GroupContext> contexts_;

    void checkStatus(dpu_error_t status, const char* context, uint32_t group_id);
};

}  // namespace mmseqs::dpu
