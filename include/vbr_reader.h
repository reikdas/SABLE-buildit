#ifndef VBR_READER_H
#define VBR_READER_H

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cstdio>

template<typename T>
struct VBRData {
    std::vector<T> val;
    std::vector<T> csr_val;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<int> rpntr;
    std::vector<int> cpntr;
    std::vector<int> bindx;
    std::vector<int> bpntrb;
    std::vector<int> bpntre;
    std::vector<int> indx;
    std::vector<int> ublocks;
};

struct DenseBlockInfo {
    int i_start, i_end;
    int j_start, j_end;
    int val_offset;
};

// Parse a vector from string like "[1,2,3]"
template<typename T>
std::vector<T> parse_array(const std::string& str) {
    std::vector<T> result;
    size_t start = str.find('[') + 1;
    size_t end = str.find(']');
    std::string content = str.substr(start, end - start);
    
    if (content.empty()) return result;
    
    std::stringstream ss(content);
    T value;
    char comma;
    while (ss >> value) {
        result.push_back(value);
        ss >> comma; // consume comma
    }
    return result;
}

// Read VBR file
template<typename T>
VBRData<T> read_vbr_file(const char* filename) {
    VBRData<T> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open VBR file %s\n", filename);
        return data;
    }
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.find("val=") == 0) {
            data.val = parse_array<T>(line);
        } else if (line.find("csr_val=") == 0) {
            data.csr_val = parse_array<T>(line);
        } else if (line.find("indptr=") == 0) {
            data.indptr = parse_array<int>(line);
        } else if (line.find("indices=") == 0) {
            data.indices = parse_array<int>(line);
        } else if (line.find("indx=") == 0) {
            data.indx = parse_array<int>(line);
        } else if (line.find("bindx=") == 0) {
            data.bindx = parse_array<int>(line);
        } else if (line.find("rpntr=") == 0) {
            data.rpntr = parse_array<int>(line);
        } else if (line.find("cpntr=") == 0) {
            data.cpntr = parse_array<int>(line);
        } else if (line.find("bpntrb=") == 0) {
            data.bpntrb = parse_array<int>(line);
        } else if (line.find("bpntre=") == 0) {
            data.bpntre = parse_array<int>(line);
        } else if (line.find("ublocks=") == 0) {
            data.ublocks = parse_array<int>(line);
        }
    }
    
    return data;
}

// Extract dense block information
template<typename T>
std::vector<DenseBlockInfo> extract_dense_blocks(const VBRData<T>& data) {
    std::vector<DenseBlockInfo> blocks;
    int count = 0;
    int nnz_block = 0;
    
    for (size_t a = 0; a < data.rpntr.size() - 1; a++) {
        if (data.bpntrb[a] == -1) continue;
        
        for (int b_idx = data.bpntrb[a]; b_idx < data.bpntre[a]; b_idx++) {
            int b = data.bindx[b_idx];
            
            // Check if this block is NOT in ublocks (i.e., it's dense)
            bool is_sparse = std::find(data.ublocks.begin(), 
                                      data.ublocks.end(), 
                                      nnz_block) != data.ublocks.end();
            
            if (!is_sparse) {
                DenseBlockInfo block;
                block.i_start = data.rpntr[a];
                block.i_end = data.rpntr[a + 1];
                block.j_start = data.cpntr[b];
                block.j_end = data.cpntr[b + 1];
                block.val_offset = data.indx[count];
                blocks.push_back(block);
                count++;
            }
            nnz_block++;
        }
    }
    
    return blocks;
}

#endif // VBR_READER_H

