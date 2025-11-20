#include <cstdlib>
#include <cstdio>
#include <cstring>

// include the project's DPU header which wraps <dpu.h> correctly
#include "../../lib/dpu/DpuCommunicationManager.h"

int main(int argc, char** argv) {
    const char* force = std::getenv("MMSEQS_DPU_FORCE_SIMULATOR");
    if (force && strlen(force) > 0) {
        setenv("UPMEM_SIMULATOR", "1", 1);
        setenv("DPU_SIMULATOR", "1", 1);
        printf("[TestDpuSmoke] MMSEQS_DPU_FORCE_SIMULATOR set → export UPMEM_SIMULATOR=1\n");
    }

    const char* s = dpu_error_to_string((dpu_error_t)0);
    if (s) {
        printf("[TestDpuSmoke] dpu_error_to_string(0) => %s\n", s);
        return 0;
    } else {
        fprintf(stderr, "[TestDpuSmoke] dpu_error_to_string returned NULL\n");
        return 2;
    }
}
