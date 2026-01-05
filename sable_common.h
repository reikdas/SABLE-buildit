// Shared code between sable-spv8.cpp and sable-mkl.cpp
// This file is in the project root (not in include/) to indicate it's not a regular include file

#ifndef SABLE_COMMON_H
#define SABLE_COMMON_H

#include "blocks/c_code_generator.h"
#include "blocks/rce.h"
#include "builder/dyn_var.h"
#include "builder/builder_dynamic.h"
#include "builder/builder_context.h"
#include "vbr_reader.h"
#include "timing_utils.h"

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

using namespace builder;
using namespace block;

typedef void (*spmv_func_t)(double*, double*, double*);

// Generate specialized kernel for one dense block with stage-2 constants inlined
inline void generate_dense_block_kernel(dyn_var<double*> y, dyn_var<double*> x, 
                                         dyn_var<double*> val,
                                         const DenseBlockInfo& block) {
    // Stage 2 constants - these values are inlined at JIT compile time
    int i_start = block.i_start;
    int i_end = block.i_end;
    int j_start = block.j_start;
    int j_end = block.j_end;
    int val_offset = block.val_offset;
    
    int i_range = i_end - i_start;
    int j_range = j_end - j_start;
    
    if (i_range == 1) {
        // Row vector optimization
        for (dyn_var<int> j_idx = 0; j_idx < j_range; j_idx = j_idx + 1) {
            dyn_var<int> j = j_start + j_idx;
            dyn_var<int> idx = val_offset + j_idx;
            y[i_start] = y[i_start] + val[idx] * x[j];
        }
    } else if (j_range == 1) {
        // Column vector optimization
        dyn_var<double> xj = x[j_start];
        for (dyn_var<int> i_idx = 0; i_idx < i_range; i_idx = i_idx + 1) {
            dyn_var<int> i = i_start + i_idx;
            dyn_var<int> idx = val_offset + i_idx;
            y[i] = y[i] + val[idx] * xj;
        }
    } else {
        // General dense block
        for (dyn_var<int> j_idx = 0; j_idx < j_range; j_idx = j_idx + 1) {
            dyn_var<int> j = j_start + j_idx;
            for (dyn_var<int> i_idx = 0; i_idx < i_range; i_idx = i_idx + 1) {
                dyn_var<int> i = i_start + i_idx;
                dyn_var<int> idx = val_offset + j_idx * i_range + i_idx;
                y[i] = y[i] + val[idx] * x[j];
            }
        }
    }
}

// Stage 2: Generate and compile specialized function for a single block
inline spmv_func_t compile_single_block_spmv(const DenseBlockInfo& block, int block_idx) {
    builder_context context;
    context.dynamic_compiler_flags = "-march=native -mavx -mprefer-vector-width=512 -ffast-math";
    
    auto ast = context.extract_function_ast(
        [&](dyn_var<double*> y, dyn_var<double*> x, dyn_var<double*> val) {
            // Generate code for this single dense block with stage-2 constants inlined
            generate_dense_block_kernel(y, x, val, block);
        },
        "specialized_spmv_block_" + std::to_string(block_idx)
    );
    
    // JIT compile to executable function
    return (spmv_func_t)builder::compile_asts(context, {ast}, "specialized_spmv_block_" + std::to_string(block_idx));
}

#endif // SABLE_COMMON_H

