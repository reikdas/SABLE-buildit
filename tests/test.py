#!/usr/bin/env python3
"""
Test script to verify that sable.cpp produces the same results as scipy
for matrix-vector multiplication.
"""

import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pathlib
import shutil
from pathlib import Path

# Get the test directory and project root
TEST_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT = TEST_DIR.parent

# Add project root to path to import from scripts
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from convert_to_vbrc import convert_yaml_to_vbrc
from utils import build_sable_binaries

SABLE_SPV8_BINARY = TEST_DIR / "sable-spv8"
SABLE_MKL_BINARY = TEST_DIR / "sable-mkl"


def build_sable():
    """Build the sable-spv8 and sable-mkl binaries in the tests directory.
    
    Note: CMakeLists.txt handles building buildit and spv8-public dependencies.
    MKL binary is skipped in GitHub Actions where MKL is not available.
    """
    # Check if we're in GitHub Actions (skip MKL build if so)
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    
    build_dir = PROJECT_ROOT / "build"
    build_sable_binaries(
        build_dir=build_dir,
        skip_mkl=is_github_actions,
        copy_to=TEST_DIR
    )


@pytest.fixture(scope="session")
def sable_spv8_binary():
    """Fixture to ensure sable-spv8 binary is built."""
    build_sable()
    assert SABLE_SPV8_BINARY.exists(), f"Sable-spv8 binary not found at {SABLE_SPV8_BINARY}"
    return SABLE_SPV8_BINARY


@pytest.fixture(scope="session")
def sable_mkl_binary():
    """Fixture to ensure sable-mkl binary is built.
    
    Note: This fixture will skip if running in GitHub Actions where MKL is not available.
    """
    # Skip if in GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("MKL binary not available in GitHub Actions")
    
    build_sable()
    if not SABLE_MKL_BINARY.exists():
        pytest.skip("Sable-mkl binary not found (MKL may not be available)")
    return SABLE_MKL_BINARY


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


def test_sable_spv8_vs_scipy_example(sable_spv8_binary):
    """Test that sable-spv8 produces the same results as scipy for example."""
    mtx_file = TEST_DIR / "example-canon.mtx"
    vbrc_file = TEST_DIR / "example.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_spv8_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)


def test_sable_spv8_vs_scipy_example2(sable_spv8_binary):
    """Test that sable-spv8 produces the same results as scipy for example2."""
    mtx_file = TEST_DIR / "example-canon2.mtx"
    vbrc_file = TEST_DIR / "example2.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_spv8_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="MKL tests skipped in GitHub Actions (MKL not available)"
)
def test_sable_mkl_vs_scipy_example(sable_mkl_binary):
    """Test that sable-mkl produces the same results as scipy for example."""
    mtx_file = TEST_DIR / "example-canon.mtx"
    vbrc_file = TEST_DIR / "example.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_mkl_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="MKL tests skipped in GitHub Actions (MKL not available)"
)
def test_sable_mkl_vs_scipy_example2(sable_mkl_binary):
    """Test that sable-mkl produces the same results as scipy for example2."""
    mtx_file = TEST_DIR / "example-canon2.mtx"
    vbrc_file = TEST_DIR / "example2.vbrc"
    vector_file = TEST_DIR / "generated_vector_11.vector"
    
    # Compute results
    scipy_result = compute_scipy_result(mtx_file, vector_file)
    sable_result = compute_sable_result(sable_mkl_binary, vbrc_file, vector_file)
    
    # Compare results
    compare_results(scipy_result, sable_result)


def parse_vbrc_file(vbrc_path):
    """Parse a VBRC file and return a dictionary of arrays."""
    data = {}
    with open(vbrc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse lines like "val=[1.0,2.0,3.0]"
            if '=' in line:
                name, value_str = line.split('=', 1)
                # Remove brackets and split by comma
                value_str = value_str.strip('[]')
                if value_str:
                    # Parse all values as floats (VBRC format uses floats)
                    try:
                        values = [float(x.strip()) for x in value_str.split(',')]
                    except ValueError:
                        values = []
                else:
                    values = []
                data[name] = values
    return data


def compare_vbrc_files(file1_path, file2_path):
    """Compare two VBRC files and assert they are identical."""
    data1 = parse_vbrc_file(file1_path)
    data2 = parse_vbrc_file(file2_path)
    
    # Check that both files have the same keys
    assert set(data1.keys()) == set(data2.keys()), \
        f"Keys mismatch: {set(data1.keys())} vs {set(data2.keys())}"
    
    # Compare each array
    for key in data1.keys():
        arr1 = np.array(data1[key], dtype=float)
        arr2 = np.array(data2[key], dtype=float)
        assert len(arr1) == len(arr2), \
            f"Length mismatch for {key}: {len(arr1)} vs {len(arr2)}"
        
        # Use numpy's allclose for comparison (handles both int and float values)
        np.testing.assert_allclose(
            arr1, arr2,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Array {key} does not match"
        )


def test_convert_yaml_to_vbrc_example2():
    """Test that convert_yaml_to_vbrc generates the same VBRC file as example2.vbrc."""
    yaml_file = TEST_DIR / "example2.yaml"
    mtx_file = TEST_DIR / "example-canon2.mtx"
    expected_vbrc = TEST_DIR / "example2.vbrc"
    generated_vbrc = TEST_DIR / "example2-generated.vbrc"
    
    # Verify expected file exists
    assert expected_vbrc.exists(), f"Expected VBRC file not found at {expected_vbrc}"
    
    # Generate VBRC file - it will create example2.vbrc in TEST_DIR
    # Use a temporary output directory to avoid overwriting the expected file
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        convert_yaml_to_vbrc(
            str(yaml_file),
            str(mtx_file),
            tmpdir
        )
        
        # The generated file should be named example2.vbrc in the output directory
        generated_in_tmp = Path(tmpdir) / "example2.vbrc"
        assert generated_in_tmp.exists(), f"Generated VBRC file not found at {generated_in_tmp}"
        
        # Copy to the final location
        if generated_vbrc.exists():
            generated_vbrc.unlink()
        import shutil
        shutil.copy2(generated_in_tmp, generated_vbrc)
    
    # Verify the generated file exists
    assert generated_vbrc.exists(), f"Generated VBRC file not found at {generated_vbrc}"
    
    # Compare the generated file with the expected file
    compare_vbrc_files(str(expected_vbrc), str(generated_vbrc))
