if(NOT HAVE_DPU)
    return()
endif()

if(NOT HAVE_TESTS)
    return()
endif()

add_executable(dpu_smoke dpu_smoke.cpp)
# Use the SDK interface for includes and linking
if(TARGET DPU::sdk)
    target_link_libraries(dpu_smoke PRIVATE DPU::sdk)
endif()
# If we built the host library, link it too so wrappers are available
if(TARGET mmseqs_dpu)
    target_link_libraries(dpu_smoke PRIVATE mmseqs_dpu)
endif()

install(TARGETS dpu_smoke DESTINATION bin)
