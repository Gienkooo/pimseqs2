## cmake/FindUPMEM.cmake
# Lightweight UPMEM DPU detection and imported-target creation.
# Usage: include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindUPMEM.cmake")
# then call find_upmem_dpu() (it reads the ENABLE_DPU cache option).

function(find_upmem_dpu)
    if(NOT DEFINED ENABLE_DPU)
        set(ENABLE_DPU OFF CACHE BOOL "Enable UPMEM DPU support" FORCE)
    endif()

    set(DPU_IMPORTED_TARGETS "")

    if(NOT ENABLE_DPU)
        set(HAVE_DPU FALSE CACHE INTERNAL "DPU support enabled")
        return()
    endif()

    find_program(DPU_PKG_CONFIG dpu-pkg-config)
    if(NOT DPU_PKG_CONFIG)
        message(WARNING "[DPU] dpu-pkg-config not found in PATH")
        message(WARNING "[DPU] Try: export PATH=/opt/upmem-sdk/bin:$PATH")
        set(HAVE_DPU FALSE CACHE INTERNAL "DPU support enabled")
        set(ENABLE_DPU OFF CACHE BOOL "Enable UPMEM DPU support" FORCE)
        return()
    endif()

    message(STATUS "[DPU] Found dpu-pkg-config: ${DPU_PKG_CONFIG}")

    execute_process(COMMAND ${DPU_PKG_CONFIG} --cflags dpu
                    OUTPUT_VARIABLE DPU_CFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${DPU_PKG_CONFIG} --libs dpu
                    OUTPUT_VARIABLE DPU_LDFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCHALL "-I[^ ]+" DPU_INCLUDE_FLAGS "${DPU_CFLAGS}")
    set(DPU_INCLUDE_DIRS "${UPMEM_INCLUDE_DIR}")
    foreach(INCLUDE_FLAG ${DPU_INCLUDE_FLAGS})
        string(REGEX REPLACE "-I" "" INCLUDE_DIR "${INCLUDE_FLAG}")
        list(APPEND DPU_INCLUDE_DIRS ${INCLUDE_DIR})
    endforeach()

    set(HAVE_DPU TRUE CACHE INTERNAL "DPU support enabled")
    set(DPU_CFLAGS_CACHED "${DPU_CFLAGS}" CACHE INTERNAL "")
    set(DPU_LDFLAGS_CACHED "${DPU_LDFLAGS}" CACHE INTERNAL "")

    string(REGEX MATCHALL "-L[^ ]+" DPU_LIB_FLAGS "${DPU_LDFLAGS}")
    set(DPU_LIB_DIRS "")
    foreach(LIB_FLAG ${DPU_LIB_FLAGS})
        string(REGEX REPLACE "-L" "" LIB_DIR "${LIB_FLAG}")
        list(APPEND DPU_LIB_DIRS ${LIB_DIR})
        link_directories(${LIB_DIR})
        message(STATUS "[DPU] Library path: ${LIB_DIR}")
    endforeach()

    string(REGEX MATCHALL "-l[^ ]+" DPU_LIB_NAME_FLAGS "${DPU_LDFLAGS}")
    foreach(LIB_NAME_FLAG ${DPU_LIB_NAME_FLAGS})
        string(REGEX REPLACE "-l" "" LIB_NAME "${LIB_NAME_FLAG}")
        set(FOUND_LIB_PATH "")
        foreach(LIB_DIR ${DPU_LIB_DIRS})
            if(EXISTS "${LIB_DIR}/lib${LIB_NAME}.so")
                set(FOUND_LIB_PATH "${LIB_DIR}/lib${LIB_NAME}.so")
                break()
            elseif(EXISTS "${LIB_DIR}/lib${LIB_NAME}.a")
                set(FOUND_LIB_PATH "${LIB_DIR}/lib${LIB_NAME}.a")
                break()
            endif()
        endforeach()
        if(FOUND_LIB_PATH)
            add_library(DPU::${LIB_NAME} UNKNOWN IMPORTED)
            set_target_properties(DPU::${LIB_NAME} PROPERTIES IMPORTED_LOCATION "${FOUND_LIB_PATH}")
            list(APPEND DPU_IMPORTED_TARGETS DPU::${LIB_NAME})
            message(STATUS "[DPU] Created imported target DPU::${LIB_NAME} -> ${FOUND_LIB_PATH}")
        else()
            message(WARNING "[DPU] Could not locate lib${LIB_NAME} in DPU lib dirs; will fall back to -l${LIB_NAME}")
        endif()
    endforeach()

    add_library(mmseqs_dpu_sdk INTERFACE)
    add_library(DPU::sdk ALIAS mmseqs_dpu_sdk)
    if(DPU_INCLUDE_DIRS)
        target_include_directories(mmseqs_dpu_sdk INTERFACE ${DPU_INCLUDE_DIRS})
    endif()

    if(DPU_IMPORTED_TARGETS)
        target_link_libraries(mmseqs_dpu_sdk INTERFACE ${DPU_IMPORTED_TARGETS})
    else()
        separate_arguments(DPU_LDFLAGS_LIST UNIX_COMMAND "${DPU_LDFLAGS}")
        if(DPU_LDFLAGS_LIST)
            target_link_libraries(mmseqs_dpu_sdk INTERFACE ${DPU_LDFLAGS_LIST})
        endif()
    endif()

    target_compile_definitions(mmseqs_dpu_sdk INTERFACE HAVE_DPU=1)

    set(DPU_IMPORTED_TARGETS "${DPU_IMPORTED_TARGETS}" CACHE INTERNAL "List of imported DPU targets")

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/dpu)
endfunction()

if (NOT UPMEM_HOME)
    if (DEFINED ENV{UPMEM_HOME})
        set(UPMEM_HOME "$ENV{UPMEM_HOME}")
    endif()
endif()

find_path(UPMEM_INCLUDE_DIR
    NAMES dpu.h
    HINTS ${UPMEM_HOME}/include/dpu
    PATHS /usr/include/dpu /usr/local/include/dpu /opt/upmem-sdk/include/dpu
)

find_library(UPMEM_LIBRARY
    NAMES dpu
    HINTS ${UPMEM_HOME}/lib
    PATHS /usr/lib /usr/local/lib /opt/upmem-sdk/lib
)

find_program(UPMEM_DPU_COMPILER
    NAMES dpu-upmem-dpurte-clang
    HINTS ${UPMEM_HOME}/bin
    PATHS /usr/bin /usr/local/bin /opt/upmem-sdk/bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UPMEM
    REQUIRED_VARS UPMEM_LIBRARY UPMEM_INCLUDE_DIR UPMEM_DPU_COMPILER
)

if (UPMEM_FOUND)
    set(UPMEM_LIBRARIES ${UPMEM_LIBRARY})
    set(UPMEM_INCLUDE_DIRS ${UPMEM_INCLUDE_DIR})
    mark_as_advanced(UPMEM_INCLUDE_DIR UPMEM_LIBRARY UPMEM_DPU_COMPILER)
endif()