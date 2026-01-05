"""Utility functions for building and managing SABLE binaries."""

import os
import pathlib
import shutil
import subprocess
from typing import Optional, Tuple


def get_project_root() -> pathlib.Path:
    """Get the project root directory (parent of scripts/)."""
    return pathlib.Path(__file__).resolve().parent.parent


def build_sable_binaries(
    build_dir: Optional[pathlib.Path] = None,
    skip_mkl: bool = False,
    copy_to: Optional[pathlib.Path] = None
) -> Tuple[pathlib.Path, Optional[pathlib.Path]]:
    """
    Build the sable-spv8 and sable-mkl binaries.
    
    Args:
        build_dir: Directory where build should occur (default: project_root/build)
        skip_mkl: If True, skip building sable-mkl (useful for CI environments)
        copy_to: If provided, copy binaries to this directory after building
    
    Returns:
        Tuple of (spv8_binary_path, mkl_binary_path) where paths may be None if not built
    
    Raises:
        RuntimeError: If building fails
    """
    project_root = get_project_root()
    
    if build_dir is None:
        build_dir = project_root / "build"
    
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
    
    # Check if we're in GitHub Actions (skip MKL build if so)
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    
    # Build sable-spv8 (always) and sable-mkl (only if not skipped and not in GitHub Actions)
    targets = ["sable-spv8"]
    if not skip_mkl and not is_github_actions:
        targets.append("sable-mkl")
    
    print(f"Compiling {' and '.join(targets)}...")
    result = subprocess.run(
        ["make", "-j", str(os.cpu_count() or 1)] + targets,
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error building sable binaries: {result.stderr}")
    
    # Get binary paths
    spv8_binary = build_dir / "sable-spv8"
    mkl_binary = build_dir / "sable-mkl" if not skip_mkl and not is_github_actions else None
    
    # Copy binaries if requested
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        
        if spv8_binary.exists():
            dest_spv8 = copy_to / "sable-spv8"
            shutil.copy2(spv8_binary, dest_spv8)
            os.chmod(dest_spv8, 0o755)
            print(f"Binary copied to {dest_spv8}")
            spv8_binary = dest_spv8
        else:
            raise RuntimeError(f"Error: Binary not found at {spv8_binary}")
        
        if mkl_binary is not None and mkl_binary.exists():
            dest_mkl = copy_to / "sable-mkl"
            shutil.copy2(mkl_binary, dest_mkl)
            os.chmod(dest_mkl, 0o755)
            print(f"Binary copied to {dest_mkl}")
            mkl_binary = dest_mkl
        elif mkl_binary is not None:
            raise RuntimeError(f"Error: Binary not found at {mkl_binary}")
    
    return spv8_binary, mkl_binary

