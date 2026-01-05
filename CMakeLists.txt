cmake_minimum_required(VERSION 3.10)
project(sable C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buildit directory
set(BUILDIT_DIR "${CMAKE_SOURCE_DIR}/buildit")
set(BUILDIT_INCLUDE_DIR "${BUILDIT_DIR}/include")
set(BUILDIT_BUILD_DIR "${BUILDIT_DIR}/build")
set(BUILDIT_LIB_DIR "${BUILDIT_BUILD_DIR}")

# Include directories
include_directories(
    ${BUILDIT_INCLUDE_DIR}
    ${BUILDIT_BUILD_DIR}/gen_headers
    ${CMAKE_SOURCE_DIR}/spv8-public/src
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler flags (matching buildit's setvars.mk)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -Wno-deprecated -Wdelete-non-virtual-dtor -Werror -Wno-vla -pedantic-errors")
# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mprefer-vector-width=512 -mavx -ffast-math")

# Link directories
link_directories(${BUILDIT_LIB_DIR})

# Build buildit library if needed
add_custom_target(buildit-build
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${BUILDIT_DIR}
    WORKING_DIRECTORY ${BUILDIT_DIR}
    COMMENT "Building buildit library..."
    BYPRODUCTS ${BUILDIT_LIB_DIR}/libbuildit.a
)

# Build spv8-public if needed
set(SPV8_DIR "${CMAKE_SOURCE_DIR}/spv8-public")
add_custom_target(spv8-public-build
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${SPV8_DIR}
    WORKING_DIRECTORY ${SPV8_DIR}
    COMMENT "Building spv8-public..."
    BYPRODUCTS ${SPV8_DIR}/bin/spmv_spv8.o ${SPV8_DIR}/bin/spmv_spv8_main.o ${SPV8_DIR}/bin/spmv_spv8
)

# Executable sable-spv8 - include spv8 source
add_executable(sable-spv8 sable-spv8.cpp spv8-public/src/spmv_spv8.c)
add_dependencies(sable-spv8 buildit-build spv8-public-build)

# Set C compiler flags for the C source file
# Define SABLE_BUILD to exclude main() from spmv_spv8.c
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mavx -Wall -Wextra")
set_source_files_properties(spv8-public/src/spmv_spv8.c PROPERTIES
    COMPILE_DEFINITIONS "SABLE_BUILD"
    LANGUAGE C
)

# Link against buildit library and math library
target_link_libraries(sable-spv8 
    ${BUILDIT_LIB_DIR}/libbuildit.a
    dl
    m
)

# Executable sable-mkl - uses MKL instead of spv8
add_executable(sable-mkl sable-mkl.cpp)
add_dependencies(sable-mkl buildit-build)

# Find MKL
find_package(PkgConfig)
if(NOT DEFINED ENV{MKLROOT})
    # Try to find MKL using pkg-config or common paths
    find_library(MKL_CORE_LIB mkl_core PATHS
        /opt/intel/mkl/lib/intel64
        /usr/lib/x86_64-linux-gnu
        ENV LD_LIBRARY_PATH
    )
    find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 PATHS
        /opt/intel/mkl/lib/intel64
        /usr/lib/x86_64-linux-gnu
        ENV LD_LIBRARY_PATH
    )
    find_library(MKL_SEQUENTIAL_LIB mkl_sequential PATHS
        /opt/intel/mkl/lib/intel64
        /usr/lib/x86_64-linux-gnu
        ENV LD_LIBRARY_PATH
    )
    
    if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB)
        target_link_libraries(sable-mkl 
            ${BUILDIT_LIB_DIR}/libbuildit.a
            ${MKL_INTEL_LP64_LIB}
            ${MKL_SEQUENTIAL_LIB}
            ${MKL_CORE_LIB}
            dl
            m
        )
    else()
        # Fallback: try linking with -lmkl_rt (runtime library)
        target_link_libraries(sable-mkl 
            ${BUILDIT_LIB_DIR}/libbuildit.a
            mkl_rt
            dl
            m
        )
    endif()
else()
    # Use MKLROOT environment variable
    set(MKL_LIB_DIR "$ENV{MKLROOT}/lib/intel64")
    target_link_libraries(sable-mkl 
        ${BUILDIT_LIB_DIR}/libbuildit.a
        ${MKL_LIB_DIR}/libmkl_intel_lp64.so
        ${MKL_LIB_DIR}/libmkl_sequential.so
        ${MKL_LIB_DIR}/libmkl_core.so
        dl
        m
    )
endif()

