#pragma once
#include "DpuCommunicationManager.h"
#include <string>
#include <unistd.h>

namespace mmseqs::dpu {

class DpuKernelManager {
public:
    enum class KernelType { KMER, UNGAPPED, GAPPED, COMBINED };

    explicit DpuKernelManager(DpuCommunicationManager& comm) : comm_(comm) {}

    void loadKernel(KernelType type) {
        std::string path = resolvePath(type);
        comm_.loadKernel(path.c_str());
    }

private:
    DpuCommunicationManager& comm_;

    std::string resolvePath(KernelType type) {
        std::string name;
        switch (type) {
            case KernelType::KMER:     name = "kmer_prefilter"; break;
            case KernelType::UNGAPPED: name = "ungapped_prefilter"; break;
            case KernelType::GAPPED:   name = "gapped_prefilter"; break;
            case KernelType::COMBINED: name = "ungapped_gapped_prefilter"; break;
        }

        if (type == KernelType::GAPPED && getenv("DPU_GAPPED_KERNEL")) {
            return std::string(getenv("DPU_GAPPED_KERNEL"));
        }

        std::string buildPath = std::string("build/lib/dpu/kernels/") + name;
        return (access(buildPath.c_str(), F_OK) != -1) ? buildPath : std::string("lib/mmseqs/dpu/") + name;
    }
};
}
