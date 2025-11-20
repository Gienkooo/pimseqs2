# DPU Kernel Compilation - AUTO DISCOVERY
# Builds ANY .c files in kernels/ subdirectory automatically

if(NOT HAVE_DPU)
    return()
endif()

# Auto-find DPU compiler
find_program(DPU_COMPILER dpu-upmem-dpurte-clang)

if(NOT DPU_COMPILER)
    message(WARNING "[DPU-Kernels] dpu-upmem-dpurte-clang not found - skipping kernels")
    return()
endif()

message(STATUS "[DPU-Kernels] Found compiler: ${DPU_COMPILER}")

# ============================================================================
# Auto-discover and compile ALL .c files in kernels/ directory
# ============================================================================

set(KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
file(GLOB_RECURSE KERNEL_SOURCES "${KERNELS_DIR}/*.c")

if(NOT KERNEL_SOURCES)
    message(WARNING "[DPU-Kernels] No .c files found in ${KERNELS_DIR}")
    return()
endif()

message(STATUS "[DPU-Kernels] Found kernels: ${KERNEL_SOURCES}")

# --------------------------------------------------------------
# Build a host-side static library for DPU support
# --------------------------------------------------------------
file(GLOB_RECURSE DPU_HOST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE DPU_HOST_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

if(DPU_HOST_SOURCES)
    add_library(mmseqs_dpu STATIC ${DPU_HOST_SOURCES} ${DPU_HOST_HEADERS})
    target_include_directories(mmseqs_dpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
    # Include paths
    target_include_directories(mmseqs_dpu PUBLIC
        ${CMAKE_BINARY_DIR}/generated
        ${PROJECT_BINARY_DIR}/generated
        ${PROJECT_SOURCE_DIR}/src/alignment
        ${PROJECT_SOURCE_DIR}/src/clustering
        ${PROJECT_SOURCE_DIR}/src/commons
        ${PROJECT_SOURCE_DIR}/src/multihit
        ${PROJECT_SOURCE_DIR}/src/prefiltering
        ${PROJECT_SOURCE_DIR}/src/linclust
        ${PROJECT_SOURCE_DIR}/src/taxonomy
        ${PROJECT_SOURCE_DIR}/src/util
        ${PROJECT_SOURCE_DIR}/src
    )
    
    # Link SDK
    if(TARGET DPU::sdk)
        target_link_libraries(mmseqs_dpu PUBLIC DPU::sdk)
    else()
        if(DEFINED DPU_LDFLAGS_CACHED)
            separate_arguments(DPU_LDFLAGS_LIST UNIX_COMMAND "${DPU_LDFLAGS_CACHED}")
            target_link_libraries(mmseqs_dpu PUBLIC ${DPU_LDFLAGS_LIST})
        endif()
    endif()
    message(STATUS "[DPU] Created host library target: mmseqs_dpu")
else()
    message(WARNING "[DPU] No host-side .cpp files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Create output directory
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dpu/kernels")

# Process each kernel file
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME "${KERNEL_SRC}" NAME_WE)
    set(KERNEL_BIN "${CMAKE_BINARY_DIR}/lib/dpu/kernels/${KERNEL_NAME}")
    
    string(REGEX MATCHALL "-I[^ ]+" DPU_INCLUDE_FLAGS "${DPU_CFLAGS_CACHED}")
    
    add_custom_command(
        OUTPUT ${KERNEL_BIN}
        COMMAND ${DPU_COMPILER} -O2 -g 
                -DNR_TASKLETS=1
                ${DPU_INCLUDE_FLAGS}
                -o ${KERNEL_BIN} ${KERNEL_SRC}
        DEPENDS ${KERNEL_SRC}
        COMMENT "[DPU] Compiling kernel: ${KERNEL_NAME}"
        VERBATIM
    )
    
    add_custom_target("dpu_kernel_${KERNEL_NAME}" ALL DEPENDS ${KERNEL_BIN})
    
    # --- FIX: Add dependency here inside the loop ---
    if(TARGET mmseqs_dpu)
        add_dependencies(mmseqs_dpu "dpu_kernel_${KERNEL_NAME}")
    endif()
    
    message(STATUS "[DPU-Kernels] ${KERNEL_NAME} -> ${KERNEL_BIN}")
    
    string(TOUPPER ${KERNEL_NAME} KERNEL_NAME_UPPER)
    set(DPU_KERNEL_${KERNEL_NAME_UPPER} "${KERNEL_BIN}" CACHE INTERNAL "")
    list(APPEND DPU_KERNEL_BINARIES "${KERNEL_BIN}")
endforeach()

add_custom_target(dpu_kernels ALL DEPENDS ${DPU_KERNEL_BINARIES})