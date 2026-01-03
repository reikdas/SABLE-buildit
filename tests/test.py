#!/usr/bin/env python3
"""
Test script to verify that sable.cpp produces the same results as scipy
for matrix-vector multiplication.

This script:
1. Uses scipy to read heart1.mtx and compute product with generated_vector_3557.vector
2. Uses sable.cpp (compiled binary) to read heart1.vbrc and generated_vector_3557.vector
3. Compares the results to ensure they match
"""

import os
import subprocess
import numpy as np
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pathlib
import shutil

# Get the test directory and project root
TEST_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT = TEST_DIR.parent

SABLE_BINARY = TEST_DIR / "sable"


def build_sable():
    """Build the sable binary in the tests directory.
    
    Note: CMakeLists.txt handles building buildit and spv8-public dependencies.
    """
    print("Building sable binary...")
    
    # Create build directory for cmake
    build_dir = PROJECT_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure cmake if needed
    cmake_cache = build_dir / "CMakeCache.txt"
    if not cmake_cache.exists():
        print("Configuring CMake...")
        result = subprocess.run(
            ["cmake", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error configuring CMake: {result.stderr}")
    
    # Build sable (CMake will handle buildit and spv8-public dependencies)
    print("Compiling sable...")
    result = subprocess.run(
        ["make", "-j", str(os.cpu_count() or 1), "sable"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error building sable: {result.stderr}")
    
    # Copy binary to tests directory
    source_binary = build_dir / "sable"
    if source_binary.exists():
        shutil.copy2(source_binary, SABLE_BINARY)
        os.chmod(SABLE_BINARY, 0o755)
        print(f"Binary copied to {SABLE_BINARY}")
    else:
        raise RuntimeError(f"Error: Binary not found at {source_binary}")


@pytest.fixture(scope="session")
def sable_binary():
    """Fixture to ensure sable binary is built."""
    build_sable()
    assert SABLE_BINARY.exists(), f"Sable binary not found at {SABLE_BINARY}"
    return SABLE_BINARY


def compute_scipy_result(mtx_file, vector_file):
    """Compute matrix-vector product using scipy.
    
    Args:
        mtx_file: Path to the .mtx matrix file
        vector_file: Path to the .vector file containing comma-separated values
    
    Returns:
        numpy array containing the matrix-vector product result
    """
    print("Reading matrix with scipy...")
    matrix = mmread(str(mtx_file))
    
    # Convert to CSR if needed
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()
    
    print(f"Matrix shape: {matrix.shape}")
    
    # Read vector file
    print("Reading vector file...")
    with open(vector_file, 'r') as f:
        content = f.read().strip()
        # Parse comma-separated values
        vector_values = [float(x.strip()) for x in content.split(',') if x.strip()]
    
    vector = np.array(vector_values, dtype=np.float64)
    print(f"Vector length: {len(vector)}")
    
    # Check dimensions
    assert matrix.shape[1] == len(vector), \
        f"Matrix columns ({matrix.shape[1]}) != vector length ({len(vector)})"
    
    # Compute product
    print("Computing matrix-vector product with scipy...")
    result = matrix.dot(vector)
    
    return result


def compute_sable_result(sable_binary_path, vbrc_file, vector_file):
    """Compute matrix-vector product using sable binary.
    
    Args:
        sable_binary_path: Path to the sable executable
        vbrc_file: Path to the .vbrc file
        vector_file: Path to the .vector file
    
    Returns:
        numpy array containing the matrix-vector product result
    """
    print("Running sable binary...")
    result = subprocess.run(
        [str(sable_binary_path), str(vbrc_file), str(vector_file)],
        capture_output=True,
        text=True,
        cwd=TEST_DIR
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Error running sable: {result.stderr}\nstdout: {result.stdout}"
        )
    
    # Parse output to extract result vector
    # The result vector is printed after "Result vector:\n"
    lines = result.stdout.split('\n')
    result_start = False
    result_values = []
    
    for line in lines:
        if "Result vector:" in line:
            result_start = True
            continue
        if result_start:
            line = line.strip()
            if line:
                try:
                    result_values.append(float(line))
                except ValueError:
                    # Skip non-numeric lines
                    pass
    
    if not result_values:
        raise RuntimeError(
            f"Could not parse result vector from sable output.\n"
            f"Output:\n{result.stdout}"
        )
    
    return np.array(result_values, dtype=np.float64)


def compare_results(scipy_result, sable_result, rtol=1e-5, atol=1e-8):
    """Compare scipy and sable results with detailed reporting.
    
    Args:
        scipy_result: numpy array from scipy computation
        sable_result: numpy array from sable computation
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Raises:
        AssertionError if results don't match within tolerances
    """
    # Check shapes match
    assert scipy_result.shape == sable_result.shape, \
        f"Shape mismatch! Scipy: {scipy_result.shape}, Sable: {sable_result.shape}"
    
    diff = np.abs(scipy_result - sable_result)
    max_diff = np.max(diff)
    mismatches = np.where(~np.isclose(scipy_result, sable_result, rtol=rtol, atol=atol))[0]
    
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Number of mismatched elements: {len(mismatches)} / {len(scipy_result)} ({100*len(mismatches)/len(scipy_result):.2f}%)")
    
    if len(mismatches) > 0:
        print(f"\nMismatched indices and values:")
        for idx in mismatches[:20]:  # Show first 20 mismatches
            print(f"  [{idx}] Scipy: {scipy_result[idx]:.15e}, Sable: {sable_result[idx]:.15e}, Diff: {diff[idx]:.15e}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more mismatches")
    
    # Use numpy's allclose for the assertion
    np.testing.assert_allclose(
        scipy_result,
        sable_result,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"Results do not match!\n"
            f"Max difference: {max_diff:.2e}\n"
            f"Mismatched elements: {len(mismatches)} / {len(scipy_result)}\n"
            f"Scipy shape: {scipy_result.shape}, Sable shape: {sable_result.shape}"
        )
    )


def test_sable_vs_scipy_example(sable_binary):
    """Test that sable.cpp produces the same results as scipy for example."""
    mtx_file = TEST_DIR / "example-canon.mtx"
    vbrc_file = TEST_DIR / "example.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)


def test_sable_vs_scipy_example2(sable_binary):
    """Test that sable.cpp produces the same results as scipy for example2."""
    mtx_file = TEST_DIR / "example-canon2.mtx"
    vbrc_file = TEST_DIR / "example2.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)
