#include <mkl_spblas.h>
#include <mkl.h>
#include "sable_common.h"

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
    
    // Set up MKL sparse matrix only if we have sparse elements
    sparse_matrix_t A = nullptr;
    struct matrix_descr descr;
    
    if (has_sparse) {
        // Prepare CSR arrays for MKL (0-based indexing)
        MKL_INT* indptr = (MKL_INT*)data.indptr.data();
        MKL_INT* indices = (MKL_INT*)data.indices.data();
        double* csr_val = data.csr_val.data();
        
        // MKL expects row_start, row_end, col_ind, values
        // indptr[i] points to start of row i, indptr[i+1] points to end
        sparse_status_t status = mkl_sparse_d_create_csr(
            &A,
            SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)n_rows,
            (MKL_INT)n_cols,
            indptr,
            indptr + 1,
            indices,
            csr_val
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "Error: Failed to create MKL sparse matrix (status=%d)\n", status);
            free(y);
            free(x);
            free(val);
            return 1;
        }
        
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_set_num_threads(1);
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
        if (has_sparse && A != nullptr) {
            mkl_sparse_destroy(A);
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
            sparse_status_t status = mkl_sparse_d_mv(
                SPARSE_OPERATION_NON_TRANSPOSE,
                1.0,
                A,
                descr,
                x,
                0.0,
                y
            );
            clock_gettime(CLOCK_MONOTONIC, &t2);
            if (status != SPARSE_STATUS_SUCCESS) {
                fprintf(stderr, "Error: MKL sparse matrix-vector multiplication failed (status=%d)\n", status);
            }
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
    if (has_sparse && A != nullptr) {
        mkl_sparse_destroy(A);
    }
    free(sparse_times);
    free(dense_block_times);
    free(y);
    free(x);
    free(val);
    
    return 0;
}

