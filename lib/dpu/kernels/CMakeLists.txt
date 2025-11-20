if(NOT UPMEM_DPU_COMPILER)
    message(FATAL_ERROR "DPU compiler not found: dpu-upmem-dpurte-clang")
endif()

set(KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/ungapped_prefilter.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gapped_prefilter.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kmer_prefilter.c
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dpu/kernels")

set(DPU_KERNEL_BINARIES "")

foreach(KSRC IN LISTS KERNEL_SRCS)
    get_filename_component(KNAME ${KSRC} NAME_WE)
    set(KBIN ${CMAKE_BINARY_DIR}/lib/dpu/kernels/${KNAME})

    string(REGEX MATCHALL "-I[^ ]+" DPU_INCLUDE_FLAGS "${UPMEM_CFLAGS_CACHED}")

    add_custom_command(
        OUTPUT ${KBIN}
        COMMAND ${UPMEM_DPU_COMPILER} -O2 -DNR_TASKLETS=4 ${DPU_INCLUDE_FLAGS} -I${UPMEM_INCLUDE_DIRS}
                -o ${KBIN} ${KSRC}
        DEPENDS ${KSRC}
        COMMENT "[DPU] Compiling kernel: ${KNAME}"
        VERBATIM
    )

    list(APPEND DPU_KERNEL_BINARIES ${KBIN})
    add_custom_target(dpu_kernel_${KNAME} ALL DEPENDS ${KBIN})
endforeach()

add_custom_target(dpu_kernels ALL DEPENDS ${DPU_KERNEL_BINARIES})

install(FILES ${DPU_KERNEL_BINARIES}
        DESTINATION lib/mmseqs/dpu
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)