#include "blocks/c_code_generator.h"
#include "blocks/rce.h"
#include "builder/dyn_var.h"
#include "builder/builder_dynamic.h"
#include "builder/builder_context.h"
#include "spv8-public/src/utility.h"
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
void generate_dense_block_kernel(dyn_var<double*> y, dyn_var<double*> x, 
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
spmv_func_t compile_single_block_spmv(const DenseBlockInfo& block, int block_idx) {
    builder_context context;
    
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

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <vbr_file> <vector_file> [bench]\n", argv[0]);
        return 1;
    }
    
    const char* vbr_file = argv[1];
    const char* vector_file = argv[2];
    int bench = 100;  // Default value
    if (argc >= 4) {
        bench = atoi(argv[3]);
        if (bench <= 0) {
            fprintf(stderr, "Error: bench must be a positive integer\n");
            return 1;
        }
    }
    
    // Stage 1: Read structure from file
    printf("Reading VBR structure...\n");
    VBRData<double> data = read_vbr_file<double>(vbr_file);
    
    std::vector<DenseBlockInfo> dense_blocks = extract_dense_blocks(data);
    
    int n_rows = data.rpntr.back();
    int n_cols = data.cpntr.back();
    int nnz = data.csr_val.size();
    
    // Check if we have sparse elements (CSR data)
    bool has_sparse = !data.csr_val.empty() && !data.indptr.empty() && !data.indices.empty();
    
    // Validate array sizes only if we have sparse elements
    if (has_sparse) {
        if (data.indptr.size() < (size_t)(n_rows + 1)) {
            fprintf(stderr, "Error: indptr size (%zu) is less than required (n_rows + 1 = %d)\n", 
                    data.indptr.size(), n_rows + 1);
            return 1;
        }
        if (data.indices.size() < (size_t)nnz) {
            fprintf(stderr, "Error: indices size (%zu) is less than nnz (%d)\n", 
                    data.indices.size(), nnz);
            return 1;
        }
        if (data.csr_val.size() < (size_t)nnz) {
            fprintf(stderr, "Error: csr_val size (%zu) is less than nnz (%d)\n", 
                    data.csr_val.size(), nnz);
            return 1;
        }
    }
    
    printf("Matrix: %d x %d, NNZ: %d, Dense blocks: %zu\n", n_rows, n_cols, nnz, dense_blocks.size());
    
    // Allocate arrays - sizes known from stage 1
    double* y = (double*)calloc(n_rows, sizeof(double));
    double* x = (double*)calloc(n_cols, sizeof(double));
    double* val = (double*)malloc(data.val.size() * sizeof(double));
    
    if (!y || !x || !val) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        return 1;
    }
    
    // Copy data
    memcpy(val, data.val.data(), data.val.size() * sizeof(double));
    
    // Read x vector
    FILE* vec_file = fopen(vector_file, "r");
    if (!vec_file) {
        fprintf(stderr, "Error: Cannot open vector file %s\n", vector_file);
        free(y);
        free(x);
        free(val);
        return 1;
    }
    for (int i = 0; i < n_cols && fscanf(vec_file, "%lf,", &x[i]) == 1; i++);
    fclose(vec_file);
    
    // Set up CSR matrix only if we have sparse elements
    struct csr_matrix csr_mat = {0};
    struct tr_matrix tr = {0};
    
    if (has_sparse) {
        csr_mat = input_matrix(
            nnz, n_rows, n_cols,
            data.csr_val.data(),
            data.indices.data(),
            data.indptr.data()
        );
        
        // Validate matrix before processing
        if (!csr_mat.nnz || !csr_mat.col || !csr_mat.rowb || !csr_mat.rowe) {
            fprintf(stderr, "Error: Failed to allocate CSR matrix\n");
            destroy_matrix(&csr_mat);
            free(y);
            free(x);
            free(val);
            return 1;
        }
        
        // Validate matrix dimensions
        if (csr_mat.rows <= 0 || csr_mat.cols <= 0 || csr_mat.m <= 0) {
            fprintf(stderr, "Error: Invalid matrix dimensions: rows=%d, cols=%d, m=%d\n", 
                    csr_mat.rows, csr_mat.cols, csr_mat.m);
            destroy_matrix(&csr_mat);
            free(y);
            free(x);
            free(val);
            return 1;
        }
        
        // Process matrix for spv8 kernel (process() now determines panel_count internally)
        tr = process(&csr_mat);
    }
    
    // Stage 2: Compile specialized function for each dense block (only if there are dense blocks)
    std::vector<spmv_func_t> block_functions;
    if (!dense_blocks.empty()) {
        printf("Compiling specialized kernels...\n");
        for (size_t i = 0; i < dense_blocks.size(); i++) {
            block_functions.push_back(compile_single_block_spmv(dense_blocks[i], i));
        }
    }
    
    // Benchmark
    long* sparse_times = (long*)malloc(bench * sizeof(long));
    size_t dense_times_size = dense_blocks.empty() ? 1 : dense_blocks.size();  // Avoid 0-size malloc
    long (*dense_block_times)[bench] = (long(*)[bench])malloc(dense_times_size * bench * sizeof(long));
    if (!sparse_times || !dense_block_times) {
        fprintf(stderr, "Error: Failed to allocate memory for timing arrays\n");
        // Cleanup
        if (sparse_times) free(sparse_times);
        if (dense_block_times) free(dense_block_times);
        if (has_sparse) {
            destroy_matrix(&csr_mat);
            for (int t = 0; t < tr.task_count; t++) {
                free(tr.tasks[t]);
            }
            free(tr.tasks);
            free(tr.task_sizes);
            free(tr.spvv8_len);
            destroy_matrix(&tr.mat);
        }
        free(y);
        free(x);
        free(val);
        return 1;
    }
    
    // Initialize dense_block_times
    for (int i = 0; i < bench; i++) {
        sparse_times[i] = 0;
        for (size_t j = 0; j < dense_blocks.size(); j++) {
            dense_block_times[j][i] = 0;
        }
    }
    
    struct timespec t1, t2;
    
    printf("Running benchmark (%d iterations)...\n", bench);
    fflush(stdout);
    
    for (int i = 0; i < bench; i++) {
        memset(y, 0, n_rows * sizeof(double));
        
        // Sparse computation - only if we have sparse elements
        if (has_sparse) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            spmv_tr_spvv8_kernel(&tr, x, y);
            clock_gettime(CLOCK_MONOTONIC, &t2);
            sparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + 
                              (t2.tv_nsec - t1.tv_nsec);
        }
        
        // Dense computation - time each block individually (only if there are dense blocks)
        if (!dense_blocks.empty()) {
            for (size_t j = 0; j < dense_blocks.size(); j++) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                block_functions[j](y, x, val);
                clock_gettime(CLOCK_MONOTONIC, &t2);
                dense_block_times[j][i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + 
                                          (t2.tv_nsec - t1.tv_nsec);
            }
        }
    }
    
    // Print timing results
    printf("Sparse: ");
    for (int i = 0; i < bench; i++) {
        printf("%ld,", sparse_times[i]);
    }
    printf("\n");
    printf("Dense: ");
    for (int i = 0; i < bench; i++) {
        long total_dense = 0;
        for (size_t j = 0; j < dense_blocks.size(); j++) {
            total_dense += dense_block_times[j][i];
        }
        printf("%ld,", total_dense);
    }
    printf("\n");
    for (size_t j = 0; j < dense_blocks.size(); j++) {
        printf("Dense Block %zu: ", j + 1);
        for (int i = 0; i < bench; i++) {
            printf("%ld,", dense_block_times[j][i]);
        }
        printf("\n");
    }

    // Print average total time (matching Python: avg_sparse_time + avg_dense_time)
    double average = calculate_average_total_time(sparse_times, (const long*)dense_block_times, bench, dense_blocks.size());
    printf("Average total time: %lf\n", average);
    
    // Print result
    printf("\nResult vector:\n");
    for (int i = 0; i < n_rows; i++) {
        printf("%lf\n", y[i]);
    }
    
    // Cleanup
    if (has_sparse) {
        destroy_matrix(&csr_mat);
        // Free tr_matrix
        destroy_matrix(&tr.mat);
        for (int t = 0; t < tr.task_count; t++) {
            free(tr.tasks[t]);
        }
        free(tr.tasks);
        free(tr.task_sizes);
        free(tr.spvv8_len);
    }
    free(sparse_times);
    free(dense_block_times);
    free(y);
    free(x);
    free(val);
    
    return 0;
}
