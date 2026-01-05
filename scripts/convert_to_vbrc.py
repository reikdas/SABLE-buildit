import re
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple
import scipy.sparse
from scipy.io import mmread
from scipy.sparse import spmatrix


def _compute_partitioning_from_dense_blocks(
    mat: spmatrix,
    dense_blocks: List[Tuple[int, int, int, int]]
) -> Tuple[List[int], List[int]]:
    """
    Compute rpntr and cpntr partitioning from dense block coordinates.
    
    Args:
        mat: Sparse matrix
        dense_blocks: List of dense block coordinates as (row_start, row_end, col_start, col_end)
    
    Returns:
        Tuple of (rpntr, cpntr) partitioning arrays
    """
    # Collect all unique row and column boundaries from dense blocks
    row_boundaries = set([0, mat.shape[0]])  # Always include start and end
    col_boundaries = set([0, mat.shape[1]])  # Always include start and end
    
    for dense_r_start, dense_r_end, dense_c_start, dense_c_end in dense_blocks:
        row_boundaries.add(dense_r_start)
        row_boundaries.add(dense_r_end)
        col_boundaries.add(dense_c_start)
        col_boundaries.add(dense_c_end)
    
    # Sort and convert to lists
    rpntr = sorted(row_boundaries)
    cpntr = sorted(col_boundaries)
    
    return rpntr, cpntr

def convert_sparse_to_vbrc_with_blocks(
    mat: spmatrix, 
    dense_blocks: List[Tuple[int, int, int, int]]
) -> Tuple[List[float], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[float]]:
    """
    Convert sparse matrix to VBRC format using specified dense block coordinates.
    The partitioning (rpntr, cpntr) is computed from the dense block boundaries.
    
    Args:
        mat: Sparse matrix to convert
        dense_blocks: List of dense block coordinates as (row_start, row_end, col_start, col_end)
                     where row_end and col_end are exclusive (e.g., [1409, 1944) means rows 1409 to 1943)
    
    Returns:
        Tuple of VBRC data structures: (val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    """
    # Compute partitioning from dense block boundaries
    rpntr, cpntr = _compute_partitioning_from_dense_blocks(mat, dense_blocks)
    
    def block_processor(r_start, r_end, c_start, c_end, r_i, c_i):
        # Extract the block
        block = mat[r_start:r_end, c_start:c_end]
        nnz = block.nnz
        
        # Only process non-empty blocks
        if nnz == 0:
            return None
            
        block_sx, block_sy = block.shape
        
        # Check if this partitioning block is contained within any specified dense block
        # Since partitioning is computed from dense block boundaries, we check containment
        is_dense_block = False
        for dense_r_start, dense_r_end, dense_c_start, dense_c_end in dense_blocks:
            # Partitioning block is contained if it's within the dense block boundaries
            if (dense_r_start <= r_start and r_end <= dense_r_end and 
                dense_c_start <= c_start and c_end <= dense_c_end):
                is_dense_block = True
                break
        
        # Extract non-zero elements and their indices using sparse operations (more efficient)
        coo_block = block.tocoo()
        dense_elems = coo_block.data.tolist()
        idxs_i = (coo_block.row + r_start).tolist()
        idxs_j = (coo_block.col + c_start).tolist()
        
        block_vals = block.todense().flatten(order='F').A1
        
        return block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy, is_dense_block
    
    # Generate VBRC data structures
    val2: List[float] = []
    indx2: List[int] = [0]
    bindx: List[int] = []
    bpntrb: List[int] = []
    bpntre: List[int] = []
    ublocks: List[int] = []
    coo_i: List[int] = []
    coo_j: List[int] = []
    coo_val: List[float] = []
    
    block_count = 0
    
    for r_i in range(len(rpntr) - 1):
        row_start_block = block_count
        r_start, r_end = rpntr[r_i], rpntr[r_i + 1]
        
        for c_i in range(len(cpntr) - 1):
            c_start, c_end = cpntr[c_i], cpntr[c_i + 1]
            
            # Process the block
            result = block_processor(r_start, r_end, c_start, c_end, r_i, c_i)
            if result is None:
                continue  # Skip empty blocks
            
            block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy, is_dense_block = result
            
            if is_dense_block:
                # Keep as dense block
                val2.extend(block_vals)
                indx2.append(len(val2))
                bindx.append(c_i)
            else:
                # Unroll to CSR
                coo_val.extend(dense_elems)
                ublocks.append(block_count)
                coo_i.extend(idxs_i)
                coo_j.extend(idxs_j)
                bindx.append(c_i)
            
            block_count += 1
        
        # Update row pointers
        if row_start_block < block_count:
            bpntrb.append(row_start_block)
            bpntre.append(block_count)
        else:
            # Empty row - mark with -1
            bpntrb.append(-1)
            bpntre.append(-1)
    
    # Create CSR representation for unrolled blocks
    if len(coo_i) > 0:
        if (rpntr[-1]-1) not in coo_i or (cpntr[-1]-1) not in coo_j:
            coo_i.append(rpntr[-1]-1)
            coo_j.append(cpntr[-1]-1)
            coo_val.append(0.0)
        csr = scipy.sparse.coo_array((coo_val, (coo_i, coo_j))).tocsr()
        indptr = csr.indptr.tolist()
        assert(len(indptr) == (rpntr[-1]+1))
        indices = csr.indices.tolist()
        csr_val = csr.data.tolist()
    else:
        csr_val = []
        indptr = []
        indices = []
    
    return val2, indx2, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val


def parse_yaml_blocks(yaml_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse YAML file to extract dense block coordinates.
    
    Args:
        yaml_path: Path to YAML file containing block information
    
    Returns:
        List of dense block coordinates as (row_start, row_end, col_start, col_end)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    blocks = []
    for block in data.get('blocks', []):
        # Parse rows: [start, end] format (end is exclusive in code, but written as inclusive in YAML)
        rows_data = block['rows']
        cols_data = block['cols']
        
        # Handle string format "[30, 2107]" (quoted in YAML)
        if isinstance(rows_data, str):
            # Match both [start, end] and [start, end) for backward compatibility
            rows_match = re.match(r'\[(\d+),\s*(\d+)[\])]', rows_data)
            if rows_match:
                row_start = int(rows_match.group(1))
                row_end = int(rows_match.group(2))
            else:
                continue
        elif isinstance(rows_data, list) and len(rows_data) == 2:
            row_start = int(rows_data[0])
            row_end = int(rows_data[1])
        else:
            continue
        
        if isinstance(cols_data, str):
            # Match both [start, end] and [start, end) for backward compatibility
            cols_match = re.match(r'\[(\d+),\s*(\d+)[\])]', cols_data)
            if cols_match:
                col_start = int(cols_match.group(1))
                col_end = int(cols_match.group(2))
            else:
                continue
        elif isinstance(cols_data, list) and len(cols_data) == 2:
            col_start = int(cols_data[0])
            col_end = int(cols_data[1])
        else:
            continue
        
        blocks.append((row_start, row_end, col_start, col_end))
    
    return blocks


def write_vbrc_file(output_path: str, val: List[float], indx: List[int], 
                    bindx: List[int], rpntr: List[int], cpntr: List[int],
                    bpntrb: List[int], bpntre: List[int], ublocks: List[int],
                    indptr: List[int], indices: List[int], csr_val: List[float]):
    """
    Write VBRC data structures to a file in the format shown in example2.vbrc.
    
    Args:
        output_path: Path to output file
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val:
            VBRC data structures
    """
    with open(output_path, 'w') as f:
        # Format arrays as comma-separated values in brackets
        def format_array(arr, name):
            if not arr:
                f.write(f"{name}=[]\n")
            else:
                f.write(f"{name}=[{','.join(str(x) for x in arr)}]\n")
        
        format_array(val, "val")
        format_array(csr_val, "csr_val")
        format_array(indptr, "indptr")
        format_array(indices, "indices")
        format_array(indx, "indx")
        format_array(bindx, "bindx")
        format_array(rpntr, "rpntr")
        format_array(cpntr, "cpntr")
        format_array(bpntrb, "bpntrb")
        format_array(bpntre, "bpntre")
        format_array(ublocks, "ublocks")


def convert_yaml_to_vbrc(yaml_path: str, mtx_path: str, output_dir: str = "Generated_VBRC"):
    """
    Convert a YAML file containing dense block information to VBRC format.
    
    Args:
        yaml_path: Path to YAML file from find-submatrices/results/
        mtx_path: Path to the .mtx matrix file
        output_dir: Directory to write output VBRC file (default: "Generated_VBRC")
    """
    yaml_path_obj = Path(yaml_path)
    matrix_name = yaml_path_obj.stem
    
    print(f"Processing {yaml_path}...")
    
    # Parse YAML to get dense blocks
    print("Parsing YAML file...")
    dense_blocks = parse_yaml_blocks(yaml_path)
    print(f"Found {len(dense_blocks)} dense blocks")
    
    # Load matrix file
    print(f"Loading matrix from {mtx_path}...")
    mat = mmread(mtx_path)
    
    # Convert to CSR if needed
    if not isinstance(mat, scipy.sparse.csr_matrix):
        mat = mat.tocsr()
    
    print(f"Matrix shape: {mat.shape}, NNZ: {mat.nnz}")
    
    # Convert to VBRC format
    print("Converting to VBRC format...")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(mat, dense_blocks)
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    
    # Write VBRC file
    output_path = output_dir_path / f"{matrix_name}.vbrc"
    print(f"Writing VBRC file to {output_path}...")
    write_vbrc_file(str(output_path), val, indx, bindx, rpntr, cpntr, 
                    bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    print(f"Successfully created {output_path}")