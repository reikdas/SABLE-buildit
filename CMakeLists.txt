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

# Executable - include spv8 source
add_executable(sable sable.cpp spv8-public/src/spmv_spv8.c)
add_dependencies(sable buildit-build spv8-public-build)

# Set C compiler flags for the C source file
# Define SABLE_BUILD to exclude main() from spmv_spv8.c
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mavx -Wall -Wextra")
set_source_files_properties(spv8-public/src/spmv_spv8.c PROPERTIES
    COMPILE_DEFINITIONS "SABLE_BUILD"
    LANGUAGE C
)

# Link against buildit library and math library
target_link_libraries(sable 
    ${BUILDIT_LIB_DIR}/libbuildit.a
    dl
    m
)

