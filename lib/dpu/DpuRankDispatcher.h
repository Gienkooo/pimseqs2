#pragma once

#include "DpuGroupManager.h"
#include <vector>
#include <cstdint>

namespace mmseqs::dpu {

// Lightweight helper to manage rank-level asynchronous scheduling with DpuGroupManager.
class RankDispatcher {
public:
    RankDispatcher(DpuGroupManager& mgr, const std::vector<std::vector<uint32_t>>& group_to_dpu_ids)
        : group_mgr_(mgr), group_to_dpu_ids_(group_to_dpu_ids), group_in_flight_(group_to_dpu_ids.size(), false) {}

    // Drain all completed groups; for each DPU in that group, call handler(dpu_id).
    template <typename HandlerFn>
    size_t drainCompleted(const HandlerFn& handle_dpu) {
        size_t drained = 0;
        while (true) {
            uint32_t gid = group_mgr_.findCompletedGroup();
            if (gid == UINT32_MAX) break;
            for (uint32_t d : group_to_dpu_ids_[gid]) {
                handle_dpu(d);
            }
            group_mgr_.releaseGroup(gid);
            group_in_flight_[gid] = false;
            drained++;
        }
        return drained;
    }

    // Attempt to launch the specified group after preparing its DPUs.
    // prepare_dpu should return true if the DPU was armed with work.
    template <typename PrepareFn>
    bool launchGroup(uint32_t gid, const PrepareFn& prepare_dpu) {
        if (gid >= group_to_dpu_ids_.size()) return false;
        if (group_in_flight_[gid]) return false;

        bool has_work = false;
        for (uint32_t d : group_to_dpu_ids_[gid]) {
            if (prepare_dpu(d)) {
                has_work = true;
            }
        }

        if (!has_work) return false;

        DpuGroupManager::GroupContext ctx{};
        group_mgr_.launchGroupAsync(gid, ctx);
        group_in_flight_[gid] = true;
        return true;
    }

    void poll() { group_mgr_.pollAllGroups(); }
    
    // Get number of groups currently in flight
    size_t getInflightCount() const {
        size_t count = 0;
        for (bool b : group_in_flight_) { if (b) count++; }
        return count;
    }
    
    // Check if a specific group is in flight
    bool isGroupInFlight(uint32_t gid) const {
        return gid < group_in_flight_.size() && group_in_flight_[gid];
    }
    
    // Get total number of groups
    uint32_t getNumGroups() const { return static_cast<uint32_t>(group_to_dpu_ids_.size()); }
    
    // Find next idle group (returns UINT32_MAX if none)
    uint32_t findIdleGroup() const {
        for (uint32_t gid = 0; gid < group_in_flight_.size(); ++gid) {
            if (!group_in_flight_[gid]) return gid;
        }
        return UINT32_MAX;
    }
    
    // Execute with work queue pattern: continuously dispatch work items to idle groups
    template <typename WorkT, typename WorkProviderFn, typename OnCompleteFn>
    void executeWorkQueue(
        const WorkProviderFn& work_provider,
        const OnCompleteFn& on_complete,
        uint32_t max_iterations = UINT32_MAX)
    {
        std::vector<WorkT> group_work(group_to_dpu_ids_.size());
        std::vector<bool> group_has_work(group_to_dpu_ids_.size(), false);
        uint32_t iterations = 0;
        
        while (iterations < max_iterations) {
            while (true) {
                uint32_t gid = group_mgr_.findCompletedGroup();
                if (gid == UINT32_MAX) break;
                
                if (group_has_work[gid]) {
                    on_complete(gid, group_to_dpu_ids_[gid], group_work[gid]);
                    group_has_work[gid] = false;
                }
                group_mgr_.releaseGroup(gid);
                group_in_flight_[gid] = false;
            }
            
            bool any_launched = false;
            for (uint32_t gid = 0; gid < group_to_dpu_ids_.size(); ++gid) {
                if (group_in_flight_[gid]) continue;
                
                auto [has_work, work_item] = work_provider(gid, group_to_dpu_ids_[gid]);
                if (!has_work) continue;
                
                group_work[gid] = work_item;
                group_has_work[gid] = true;
                
                DpuGroupManager::GroupContext ctx{};
                group_mgr_.launchGroupAsync(gid, ctx);
                group_in_flight_[gid] = true;
                any_launched = true;
            }
            
            size_t inflight = getInflightCount();
            if (inflight == 0 && !any_launched) break;
            
            if (!any_launched) {
                group_mgr_.pollAllGroups();
            }
            
            iterations++;
        }
    }

private:
    DpuGroupManager& group_mgr_;
    const std::vector<std::vector<uint32_t>>& group_to_dpu_ids_;
    std::vector<bool> group_in_flight_;
};

} // namespace mmseqs::dpu
