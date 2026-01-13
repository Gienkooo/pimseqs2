#pragma once
#include "DpuCommunicationManager.h"
#include <string>
#include <unistd.h>

namespace mmseqs::dpu {

class DpuKernelManager {
public:
    enum class KernelType { KMER, UNGAPPED, GAPPED, COMBINED, BOOT, NONE };

    explicit DpuKernelManager(DpuCommunicationManager& comm) 
        : comm_(comm), lastLoadedKernel_(KernelType::NONE) {}

    void loadKernel(KernelType type) {
        // Optimization: On hardware, avoid reloading the same kernel to save time.
        // The kernel must reset its own state (BSS) if not reloaded.
        if (!comm_.isSimulator() && type == lastLoadedKernel_) {
            return;
        }

        std::string path = resolvePath(type);
        comm_.loadKernel(path.c_str());
        lastLoadedKernel_ = type;
    }

    static std::string resolvePath(KernelType type) {
        std::string name;
        switch (type) {
            case KernelType::KMER:     name = "kmer_prefilter"; break;
            case KernelType::UNGAPPED: name = "ungapped_prefilter"; break;
            case KernelType::GAPPED:   name = "gapped_prefilter"; break;
            case KernelType::COMBINED: name = "ungapped_gapped_prefilter"; break;
            case KernelType::BOOT:     name = "boot"; break;
            default: return "";
        }

        if (type == KernelType::GAPPED && getenv("DPU_GAPPED_KERNEL")) {
            return std::string(getenv("DPU_GAPPED_KERNEL"));
        }

        std::string buildPath = std::string("build/lib/dpu/kernels/") + name;
        return (access(buildPath.c_str(), F_OK) != -1) ? buildPath : std::string("lib/mmseqs/dpu/") + name;
    }

private:
    DpuCommunicationManager& comm_;
    KernelType lastLoadedKernel_;
};
}
