#!/usr/bin/env python3
"""
Benchmark script for SABLE sparse matrix-vector multiplication.

This script:
1. Reads YAML files from find-submatrices/results/ to get dense block information
2. Downloads matrices from SuiteSparse
3. Converts to VBRC format
4. Runs sable-spv8 and sable-mkl binaries
5. Collects timing information and writes to JSON files
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))
from convert_to_vbrc import convert_sparse_to_vbrc_with_blocks, parse_yaml_blocks, write_vbrc_file
from utils import build_sable_binaries, get_project_root

# Add find-submatrices to path for importing
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
from find_matrices import download_matrix, get_matrix_paths, cleanup_matrix_files, get_matrix_info


FILEPATH = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
BUILD_DIR = FILEPATH / "build"
SABLE_SPV8_BINARY = BUILD_DIR / "sable-spv8"
SABLE_MKL_BINARY = BUILD_DIR / "sable-mkl"
SUITESPARSE_DIR = FILEPATH / "Suitesparse"
GENERATED_VBRC_DIR = FILEPATH / "Generated_VBRC"
GENERATED_VBRC_SPARSE_DIR = FILEPATH / "Generated_VBRC_sparse"

# Default benchmark iterations
DEFAULT_BENCH_ITERATIONS = 100


def remove_outliers_deciles(data: List[float]) -> List[float]:
    """Remove outliers using deciles (10th and 90th percentiles)."""
    if len(data) < 10:
        return data
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx_10 = int(n * 0.10)
    idx_90 = int(n * 0.90)
    
    d1 = sorted_data[idx_10]  # 10th percentile
    d9 = sorted_data[idx_90]  # 90th percentile
    
    return [x for x in data if d1 <= x <= d9]


def download_matrix_with_cleanup(matrix_name: str) -> Optional[Tuple[str, Any]]:
    """
    Download a matrix from SuiteSparse to Suitesparse directory and return the path to the .mtx file and matrix_info.
    Returns None if the download fails.
    
    Returns:
        Tuple of (matrix_path, matrix_info) or None if download fails
    """
    # Ensure Suitesparse directory exists
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    
    # Set environment variable to control download location
    original_ssgetpy_dir = os.environ.get('SSGETPY_DIR')
    try:
        os.environ['SSGETPY_DIR'] = str(SUITESPARSE_DIR)
        matrix_path, matrix_info = download_matrix(matrix_name)
        if matrix_path is None:
            return None
        return matrix_path, matrix_info
    finally:
        # Restore original environment variable
        if original_ssgetpy_dir is not None:
            os.environ['SSGETPY_DIR'] = original_ssgetpy_dir
        elif 'SSGETPY_DIR' in os.environ:
            del os.environ['SSGETPY_DIR']


def create_vector_file(n_cols: int, output_path: str, fill_value: float = 1.0):
    """Create a vector file with the given number of columns."""
    with open(output_path, 'w') as f:
        values = [str(fill_value)] * n_cols
        f.write(','.join(values))


def ensure_sable_binaries():
    """Ensure the sable-spv8 and sable-mkl binaries exist, building if necessary."""
    if SABLE_SPV8_BINARY.exists() and SABLE_MKL_BINARY.exists():
        print("Sable binaries already exist.")
        return True
    
    try:
        spv8_binary, mkl_binary = build_sable_binaries(build_dir=BUILD_DIR, skip_mkl=False)
        return spv8_binary.exists() and (mkl_binary is None or mkl_binary.exists())
    except RuntimeError as e:
        print(f"Error building sable binaries: {e}")
        return False


def run_sable_benchmark(
    binary_path: pathlib.Path,
    vbrc_path: str,
    vector_path: str,
    bench_iterations: int = DEFAULT_BENCH_ITERATIONS
) -> Tuple[float, float, Dict[int, float], List[float], List[float]]:
    """
    Run a sable binary and parse its output.
    
    Returns:
        Tuple of (avg_sparse_time, avg_dense_time, avg_individual_block_times,
                  raw_sparse_times, raw_dense_times)
    """
    sparse_times = []
    dense_times = []
    individual_dense_block_times: Dict[int, List[float]] = {}
    
    try:
        output = subprocess.check_output(
            [str(binary_path), vbrc_path, vector_path, str(bench_iterations)],
            stderr=subprocess.STDOUT
        ).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error running {binary_path}: {e}")
        print(f"Output: {e.output.decode('utf-8') if e.output else 'None'}")
        return 0, 0, {}, [], []
    
    lines = output.split('\n')
    
    for line in lines:
        # Parse sparse times
        if line.startswith('Sparse: '):
            sparse_content = line[8:].strip().rstrip(',')
            if sparse_content:
                sparse_times = [float(x.strip()) for x in sparse_content.split(',') if x.strip()]
        
        # Parse total dense times
        elif line.startswith('Dense: '):
            dense_content = line[7:].strip().rstrip(',')
            if dense_content:
                dense_times = [float(x.strip()) for x in dense_content.split(',') if x.strip()]
        
        # Parse individual dense block times
        else:
            dense_block_match = re.match(r'Dense Block (\d+): (.+)', line)
            if dense_block_match:
                block_id = int(dense_block_match.group(1))
                block_times_str = dense_block_match.group(2).strip().rstrip(',')
                if block_times_str:
                    block_times = [float(x.strip()) for x in block_times_str.split(',') if x.strip()]
                    individual_dense_block_times[block_id] = block_times
    
    # Remove outliers and calculate averages
    sparse_times_clean = remove_outliers_deciles(sparse_times)
    dense_times_clean = remove_outliers_deciles(dense_times)
    
    avg_sparse_time = statistics.mean(sparse_times_clean) if sparse_times_clean else 0
    avg_dense_time = statistics.mean(dense_times_clean) if dense_times_clean else 0
    
    # Calculate averages for individual dense blocks
    avg_individual_block_times = {}
    for block_id, times in individual_dense_block_times.items():
        times_clean = remove_outliers_deciles(times)
        avg_individual_block_times[block_id] = statistics.mean(times_clean) if times_clean else 0
    
    return avg_sparse_time, avg_dense_time, avg_individual_block_times, sparse_times, dense_times


def analyze_dense_blocks(
    mat: csr_matrix,
    dense_blocks: List[Tuple[int, int, int, int]]
) -> List[Dict[str, Any]]:
    """Analyze dense blocks and return their characteristics."""
    results = []
    
    for row_start, row_end, col_start, col_end in dense_blocks:
        block = mat[row_start:row_end, col_start:col_end]
        rows = row_end - row_start
        cols = col_end - col_start
        block_size = rows * cols
        block_nnz = block.nnz
        density = (block_nnz / block_size * 100) if block_size > 0 else 0
        
        results.append({
            "rows": rows,
            "cols": cols,
            "density_percent": density,
            "nnz": block_nnz,
        })
    
    return results


def process_matrix(
    matrix_name: str,
    yaml_path: pathlib.Path,
    bench_iterations: int = DEFAULT_BENCH_ITERATIONS
) -> Optional[Dict[str, Any]]:
    """
    Process a single matrix and return benchmark results.
    
    Args:
        matrix_name: Name of the matrix
        yaml_path: Path to the YAML file with dense block info
        bench_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with benchmark results, or None if processing failed
    """
    print(f"\nProcessing {matrix_name}...")
    
    # Download matrix
    download_result = download_matrix_with_cleanup(matrix_name)
    if download_result is None:
        return None
    mtx_path, matrix_info = download_result
    
    try:
        # Load matrix
        print(f"  Loading matrix from {mtx_path}...")
        mat = mmread(mtx_path)
        if not isinstance(mat, csr_matrix):
            mat = csr_matrix(mat)
        
        matrix_rows, matrix_cols = mat.shape
        matrix_nnz = mat.nnz
        
        print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")
        
        # Parse dense blocks from YAML
        print(f"  Parsing dense blocks from {yaml_path}...")
        dense_blocks = parse_yaml_blocks(str(yaml_path))
        print(f"  Found {len(dense_blocks)} dense blocks")
        
        # Analyze dense blocks
        dense_block_info = analyze_dense_blocks(mat, dense_blocks)
        
        # Convert to VBRC format (with dense blocks)
        print(f"  Converting to VBRC format (split)...")
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
            convert_sparse_to_vbrc_with_blocks(mat, dense_blocks)
        
        # Write VBRC file (split version with dense blocks) to Generated_VBRC/
        GENERATED_VBRC_DIR.mkdir(parents=True, exist_ok=True)
        vbrc_matrix_dir = GENERATED_VBRC_DIR / matrix_name
        vbrc_matrix_dir.mkdir(parents=True, exist_ok=True)
        vbrc_path = str(vbrc_matrix_dir / f"{matrix_name}.vbrc")
        write_vbrc_file(vbrc_path, val, indx, bindx, rpntr, cpntr,
                        bpntrb, bpntre, ublocks, indptr, indices, csr_val)
        
        # Convert to fully sparse VBRC format (no dense blocks - all CSR)
        print(f"  Converting to VBRC format (fully sparse)...")
        val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, \
            bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, \
            indices_sparse, csr_val_sparse = convert_sparse_to_vbrc_with_blocks(mat, [])
        
        # Write fully sparse VBRC file to Generated_VBRC_sparse/
        GENERATED_VBRC_SPARSE_DIR.mkdir(parents=True, exist_ok=True)
        sparse_vbrc_matrix_dir = GENERATED_VBRC_SPARSE_DIR / matrix_name
        sparse_vbrc_matrix_dir.mkdir(parents=True, exist_ok=True)
        sparse_vbrc_path = str(sparse_vbrc_matrix_dir / f"{matrix_name}.vbrc")
        write_vbrc_file(sparse_vbrc_path, val_sparse, indx_sparse, bindx_sparse,
                        rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse,
                        ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse)
        
        # Create vector file (in same directory as split VBRC for convenience)
        vector_path = str(vbrc_matrix_dir / f"vector_{matrix_cols}.vector")
        create_vector_file(matrix_cols, vector_path)
        
        # Calculate matrix statistics
        density_calculation = matrix_nnz / (matrix_rows * matrix_cols) if matrix_rows * matrix_cols > 0 else 0
        
        # Calculate nnz statistics
        dense_all = sum(block.get("rows", 0) * block.get("cols", 0) for block in dense_block_info)
        dense_nnz = sum(block.get("nnz", 0) for block in dense_block_info)
        sparse_nnz = matrix_nnz - dense_nnz
        extra_zeros = dense_all - dense_nnz
        dense_nnz_perc = (dense_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
        sparse_nnz_perc = (sparse_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
        
        result = {
            "matrix_name": matrix_name,
            "matrix_dimensions": {
                "rows": matrix_rows,
                "cols": matrix_cols,
                "nnz": matrix_nnz,
                "density": density_calculation
            },
            "nnz": {
                "sparse_nnz": sparse_nnz,
                "dense_all": dense_all,
                "dense_nnz": dense_nnz,
                "extra_zeros": extra_zeros,
                "dense_nnz_perc": dense_nnz_perc,
                "sparse_nnz_perc": sparse_nnz_perc
            },
            "dense_blocks": dense_block_info,
            "vbrc_path": vbrc_path,
            "sparse_vbrc_path": sparse_vbrc_path,
            "vector_path": vector_path
        }
        
        return result
        
    except Exception as e:
        print(f"  Error processing {matrix_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup downloaded matrix files
        if 'matrix_info' in locals() and matrix_info:
            tar_path, _, matrix_subdir, _ = get_matrix_paths(matrix_info)
            cleanup_matrix_files(tar_path, matrix_subdir)


def benchmark_single_binary(
    binary_path: pathlib.Path,
    matrix_result: Dict[str, Any],
    bench_iterations: int = DEFAULT_BENCH_ITERATIONS
) -> Dict[str, Any]:
    """
    Benchmark a single binary on a processed matrix.
    
    Returns updated result dictionary with timing information.
    """
    vbrc_path = matrix_result["vbrc_path"]
    sparse_vbrc_path = matrix_result["sparse_vbrc_path"]
    vector_path = matrix_result["vector_path"]
    matrix_name = matrix_result["matrix_name"]
    matrix_nnz = matrix_result["matrix_dimensions"]["nnz"]
    dense_block_info = matrix_result["dense_blocks"]
    dense_nnz_perc = matrix_result["nnz"]["dense_nnz_perc"]
    
    # Benchmark split version (with dense blocks)
    print(f"  Running {binary_path.name} (split)...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, _, _ = \
        run_sable_benchmark(binary_path, vbrc_path, vector_path, bench_iterations)
    
    # Benchmark fully sparse version
    print(f"  Running {binary_path.name} (fully sparse)...")
    fully_sparse_time, _, _, _, _ = \
        run_sable_benchmark(binary_path, sparse_vbrc_path, vector_path, bench_iterations)
    
    # Calculate percentages
    total_time = avg_sparse_time + avg_dense_time
    sparse_percentage = (avg_sparse_time / total_time * 100) if total_time > 0 else 0
    dense_percentage = (avg_dense_time / total_time * 100) if total_time > 0 else 0
    
    # Calculate speedup metrics
    speedup = (fully_sparse_time / total_time) if total_time > 0 else 0
    expected_sparse_time_ns = ((100 - dense_nnz_perc) / 100) * fully_sparse_time
    dense_if_sparse_time_ns = fully_sparse_time - expected_sparse_time_ns
    
    # Build result
    result = {
        "matrix_name": matrix_name,
        "matrix_dimensions": matrix_result["matrix_dimensions"],
        "timing": {
            "sparse_time_ns": avg_sparse_time,
            "dense_time_ns": avg_dense_time,
            "total_time_ns": total_time,
            "sparse_percentage": sparse_percentage,
            "dense_percentage": dense_percentage,
            "fully_sparse_time": fully_sparse_time,
            "speedup": speedup,
            "expected_sparse_time_ns": expected_sparse_time_ns,
            "dense_if_sparse_time_ns": dense_if_sparse_time_ns,
        },
        "nnz": matrix_result["nnz"],
        "individual_dense_block_timings": {}
    }
    
    # Add individual dense block timings
    for block_id, block_time in avg_individual_block_times.items():
        # block_id is 1-indexed, dense_block_info is 0-indexed
        block_info = dense_block_info[block_id - 1] if block_id - 1 < len(dense_block_info) else {}
        
        result["individual_dense_block_timings"][f"block_{block_id}"] = {
            "time_ns": block_time,
            "percentage_of_total_time": (block_time / total_time * 100) if total_time > 0 else 0,
            "percentage_of_dense_time": (block_time / avg_dense_time * 100) if avg_dense_time > 0 else 0,
            "percentage_of_total_nnz": (block_info.get("nnz", 0) / matrix_nnz * 100) if matrix_nnz > 0 else 0,
            "rows": block_info.get("rows", 0),
            "cols": block_info.get("cols", 0),
            "density_percent": block_info.get("density_percent", 0),
            "nnz": block_info.get("nnz", 0),
        }
    
    return result


def get_available_matrices() -> List[str]:
    """Get list of matrices with YAML files in find-submatrices/results/."""
    yaml_files = list(RESULTS_DIR.glob("*.yaml"))
    return [f.stem for f in yaml_files]


def main():
    """Main benchmarking function."""
    
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix-vector multiplication",
        epilog="Examples:\n"
               "  %(prog)s eris1176 bloweybl heart1\n"
               "  %(prog)s --matrices eris1176 bloweybl\n"
               "  %(prog)s  # processes all available matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "matrices",
        nargs="*",
        help="Matrix names to benchmark (positional arguments). If not provided, all available matrices are processed."
    )
    parser.add_argument("--matrices", dest="matrices_flag", nargs="*", metavar="MATRIX",
                        help="Matrix names to benchmark (alternative to positional arguments)")
    parser.add_argument("--bench", type=int, default=DEFAULT_BENCH_ITERATIONS, 
                        help=f"Number of benchmark iterations (default: {DEFAULT_BENCH_ITERATIONS})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--skip-spv8", action="store_true", help="Skip spv8 benchmarks")
    parser.add_argument("--skip-mkl", action="store_true", help="Skip MKL benchmarks")
    
    args = parser.parse_args()
    
    # Build binaries
    if not ensure_sable_binaries():
        print("Error: Failed to build sable binaries")
        return 1
    
    # Get matrices to process - merge positional arguments and --matrices flag, or use all
    matrices = args.matrices or args.matrices_flag
    if not matrices:
        matrices = get_available_matrices()
    
    print(f"Will process {len(matrices)} matrices")
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ensure Suitesparse directory exists
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    
    spv8_results = []
    mkl_results = []
    
    for matrix_name in matrices:
            yaml_path = RESULTS_DIR / f"{matrix_name}.yaml"
            
            if not yaml_path.exists():
                print(f"Warning: YAML file not found for {matrix_name}, skipping")
                continue
            
            # Process matrix (download, convert, etc.)
            matrix_result = process_matrix(matrix_name, yaml_path, args.bench)
            
            if matrix_result is None:
                continue
            
            # Benchmark with sable-spv8
            if not args.skip_spv8 and SABLE_SPV8_BINARY.exists():
                spv8_result = benchmark_single_binary(
                    SABLE_SPV8_BINARY, matrix_result, args.bench
                )
                spv8_results.append(spv8_result)
                
                # Write intermediate results
                spv8_output = output_dir / "spmv_spv8.json"
                with open(spv8_output, 'w') as f:
                    json.dump(spv8_results, f, indent=2)
                print(f"  SPV8 results written to {spv8_output}")
            
            # Benchmark with sable-mkl
            if not args.skip_mkl and SABLE_MKL_BINARY.exists():
                mkl_result = benchmark_single_binary(
                    SABLE_MKL_BINARY, matrix_result, args.bench
                )
                mkl_results.append(mkl_result)
                
                # Write intermediate results
                mkl_output = output_dir / "spmv_mkl.json"
                with open(mkl_output, 'w') as f:
                    json.dump(mkl_results, f, indent=2)
                print(f"  MKL results written to {mkl_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    if spv8_results:
        print(f"\nSPV8 Results ({len(spv8_results)} matrices):")
        for result in spv8_results:
            num_dense_blocks = len(result['individual_dense_block_timings'])
            print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                  f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                  f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                  f"total: {result['timing']['total_time_ns']:.0f}ns, "
                  f"speedup: {result['timing']['speedup']:.3f}x")
    
    if mkl_results:
        print(f"\nMKL Results ({len(mkl_results)} matrices):")
        for result in mkl_results:
            num_dense_blocks = len(result['individual_dense_block_timings'])
            print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                  f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                  f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                  f"total: {result['timing']['total_time_ns']:.0f}ns, "
                  f"speedup: {result['timing']['speedup']:.3f}x")
    
    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())

