#include <vector>
#include <algorithm>

// Remove outliers using deciles (10th and 90th percentiles)
std::vector<double> remove_outliers_deciles(const std::vector<double>& data) {
    if (data.size() < 10) {
        return data;  // Not enough data points for deciles
    }
    
    // Create a sorted copy for percentile calculation
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Calculate 10th and 90th percentiles
    size_t n = sorted_data.size();
    size_t idx_10 = (size_t)(n * 0.10);
    size_t idx_90 = (size_t)(n * 0.90);
    
    double D1 = sorted_data[idx_10];  // 10th percentile
    double D9 = sorted_data[idx_90];  // 90th percentile
    
    // Filter data to keep only values between D1 and D9 (inclusive)
    std::vector<double> filtered;
    for (double x : data) {
        if (D1 <= x && x <= D9) {
            filtered.push_back(x);
        }
    }
    
    return filtered;
}

// Calculate average total time from sparse and dense block times
// Returns avg_sparse_time + avg_dense_time after removing outliers
// dense_block_times is a 2D array: dense_block_times[block_idx][iteration]
double calculate_average_total_time(const long* sparse_times,
                                    const long* dense_block_times,
                                    int bench,
                                    size_t num_dense_blocks) {
    // Collect sparse times
    std::vector<double> sparse_times_vec;
    for (int i = 0; i < bench; i++) {
        sparse_times_vec.push_back((double)sparse_times[i]);
    }
    
    // Remove outliers from sparse times
    std::vector<double> sparse_times_filtered = remove_outliers_deciles(sparse_times_vec);
    
    // Calculate average sparse time
    double avg_sparse_time = 0;
    if (!sparse_times_filtered.empty()) {
        for (double t : sparse_times_filtered) {
            avg_sparse_time += t;
        }
        avg_sparse_time /= sparse_times_filtered.size();
    }
    
    // Collect dense times (sum of all dense blocks per iteration)
    std::vector<double> dense_times_vec;
    for (int i = 0; i < bench; i++) {
        long total_dense = 0;
        for (size_t j = 0; j < num_dense_blocks; j++) {
            // Access dense_block_times[j][i] using pointer arithmetic
            total_dense += dense_block_times[j * bench + i];
        }
        dense_times_vec.push_back((double)total_dense);
    }
    
    // Remove outliers from dense times
    std::vector<double> dense_times_filtered = remove_outliers_deciles(dense_times_vec);
    
    // Calculate average dense time
    double avg_dense_time = 0;
    if (!dense_times_filtered.empty()) {
        for (double t : dense_times_filtered) {
            avg_dense_time += t;
        }
        avg_dense_time /= dense_times_filtered.size();
    }
    
    return avg_sparse_time + avg_dense_time;
}