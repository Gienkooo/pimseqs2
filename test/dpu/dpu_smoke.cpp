#include <cstdlib>
#include <cstdio>
#include <cstring>

// include the project's DPU header which wraps <dpu.h> correctly
#include "../../lib/dpu/DpuCommunicationManager.h"

int main(int argc, char** argv) {
    // Allow forcing simulator mode via env var
    const char* force = std::getenv("MMSEQS_DPU_FORCE_SIMULATOR");
    if (force && strlen(force) > 0) {
        // set a couple of commonly used env vars for DPU runtimes so simulator
        // backends are preferred. If these names are not used by the SDK, they
        // harmlessly remain set.
        setenv("UPMEM_SIMULATOR", "1", 1);
        setenv("DPU_SIMULATOR", "1", 1);
        printf("[dpu_smoke] MMSEQS_DPU_FORCE_SIMULATOR set → export UPMEM_SIMULATOR=1\n");
    }

    // Call a trivial SDK function to ensure symbols are available.
    // dpu_error_to_string is provided by libdpu and should return a valid C string.
    const char* s = dpu_error_to_string((dpu_error_t)0);
    if (s) {
        printf("[dpu_smoke] dpu_error_to_string(0) => %s\n", s);
        return 0;
    } else {
        fprintf(stderr, "[dpu_smoke] dpu_error_to_string returned NULL\n");
        return 2;
    }
}
