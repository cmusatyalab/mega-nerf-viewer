// Released under MIT License
// license available in LICENSE file, or at
// http://www.opensource.org/licenses/mit-license.php

#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <stdint.h>
#include <stdexcept>
#include <regex>

char cnpy::BigEndianTest() {
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const std::type_info& t) {
    if (t == typeid(float)) return 'f';
    if (t == typeid(double)) return 'f';
    if (t == typeid(long double)) return 'f';

    if (t == typeid(int)) return 'i';
    if (t == typeid(char)) return 'i';
    if (t == typeid(short)) return 'i';
    if (t == typeid(long)) return 'i';
    if (t == typeid(long long)) return 'i';

    if (t == typeid(char)) return 'u';
    if (t == typeid(unsigned short)) return 'u';
    if (t == typeid(unsigned long)) return 'u';
    if (t == typeid(unsigned long long)) return 'u';
    if (t == typeid(unsigned int)) return 'u';

    if (t == typeid(bool)) return 'b';

    if (t == typeid(std::complex<float>)) return 'c';
    if (t == typeid(std::complex<double>)) return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

template <>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs,
                                    const std::string& rhs) {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const char* rhs) {
    // write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

uint16_t cnpy::parse_npy_header(const char* buffer, size_t& word_size,
                                std::vector<size_t>& shape,
                                bool& fortran_order) {
    // std::string magic_string(buffer,6);
    uint8_t major_version = *reinterpret_cast<const uint8_t*>(buffer + 6);
    uint8_t minor_version = *reinterpret_cast<const uint8_t*>(buffer + 7);
    uint16_t header_len = *reinterpret_cast<const uint16_t*>(buffer + 8);
    std::string header(reinterpret_cast<const char*>(buffer + 9), header_len);

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr") + 9;
    bool littleEndian =
        (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());

    char fmt = header[loc1 + 1];
    // MODIFIED: fix unicode string
    if (fmt == 'U') word_size *= 4;

    return header_len + 10;
}

void cnpy::parse_npy_header(FILE* fp, size_t& word_size,
                            std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
            "parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian =
        (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    char fmt = header[loc1 + 1];
    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());

    // MODIFIED: fix unicode string
    if (fmt == 'U') word_size *= 4;
}

void cnpy::parse_zip_footer(FILE* fp, uint16_t& nrecs,
                            size_t& global_header_size,
                            size_t& global_header_offset) {
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22) throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*)&footer[4];
    disk_start = *(uint16_t*)&footer[6];
    nrecs_on_disk = *(uint16_t*)&footer[8];
    nrecs = *(uint16_t*)&footer[10];
    global_header_size = *(uint32_t*)&footer[12];
    global_header_offset = *(uint32_t*)&footer[16];
    comment_len = *(uint16_t*)&footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

cnpy::NpyArray load_the_npy_file(FILE* fp) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

cnpy::NpyArray load_mem_npy_file(const char** ptr, const char* ptr_end) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    *ptr += cnpy::parse_npy_header(*ptr, word_size, shape, fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    if (*ptr + arr.num_bytes() >= ptr_end)
        throw std::runtime_error("load_mem_npy_file: unexpected EOF");
    memcpy(arr.data<char>(), *ptr, arr.num_bytes());
    *ptr += arr.num_bytes();
    return arr;
}

cnpy::NpyArray load_the_npz_array(FILE* fp, uint64_t compr_bytes,
                                  uint64_t uncompr_bytes) {
    std::vector<char> buffer_compr(compr_bytes);
    cnpy::NpyArray array;
    array.data_holder.resize(uncompr_bytes);
    size_t nread = fread(&buffer_compr[0], 1, compr_bytes, fp);
    if (nread != compr_bytes) {
        throw std::runtime_error("load_the_npz_array: failed fread");
    }

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = reinterpret_cast<unsigned char*>(&buffer_compr[0]);
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = reinterpret_cast<unsigned char*>(&array.data_holder[0]);

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(&array.data_holder[0], word_size, shape,
                           fortran_order);

    array.reinit(shape, word_size, fortran_order);
    return array;
}

cnpy::NpyArray load_mem_npz_array(const char** ptr, const char* ptr_end,
                                  uint64_t compr_bytes,
                                  uint64_t uncompr_bytes) {
    if (*ptr + compr_bytes > ptr_end) {
        throw std::runtime_error("load_mem_npz_array: unexpected EOF");
    }
    cnpy::NpyArray array;
    array.data_holder.resize(uncompr_bytes);

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = const_cast<unsigned char*>(
        reinterpret_cast<const unsigned char*>(*ptr));
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = reinterpret_cast<unsigned char*>(&array.data_holder[0]);

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(&array.data_holder[0], word_size, shape,
                           fortran_order);
    array.reinit(shape, word_size, fortran_order);
    *ptr += compr_bytes;
    return array;
}

cnpy::npz_t cnpy::npz_load(const std::string& fname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp) {
        throw std::runtime_error("npz_load: Error! Unable to open file " +
                                 fname + "!");
    }

    cnpy::npz_t arrays;

    while (1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);

        if (headerres != 30) throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string varname(name_len, ' ');
        size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        // erase the lagging .npy
        varname.resize(varname.size() - 4);

        uint16_t compr_method =
            *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint64_t compr_bytes =
            *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint64_t uncompr_bytes =
            *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        if (extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res =
                fread(&buff[0], sizeof(char), extra_field_len, fp);
            if (compr_bytes == 0xffffffff && uncompr_bytes == 0xffffffff) {
                // Handle zip64 for large file (note: not in original cnpy)
                uint16_t header_id = *reinterpret_cast<uint16_t*>(&buff[0]);
                // Check if really zip64
                if (header_id == 1) {
                    // Update file sizes to 64-bit values
                    uncompr_bytes = *reinterpret_cast<uint64_t*>(&buff[0] + 4);
                    compr_bytes = *reinterpret_cast<uint64_t*>(&buff[0] + 12);
                }
            }
            if (efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        if (compr_method == 0) {
            arrays[varname] = load_the_npy_file(fp);
        } else {
            arrays[varname] =
                load_the_npz_array(fp, compr_bytes, uncompr_bytes);
        }
    }

    fclose(fp);
    return arrays;
}

cnpy::npz_t cnpy::npz_load_mem(const char* data, uint64_t size) {
    cnpy::npz_t arrays;
    const char* ptr = data;
    const char* ptr_end = ptr + size;
    while (1) {
        const char* local_header = ptr;
        if (ptr + 30 > ptr_end) {
            throw std::runtime_error("npz_load_mem: unexpected EOF");
        }
        ptr += 30;

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        if (ptr + name_len > ptr_end) {
            throw std::runtime_error("npz_load_mem: unexpected EOF");
        }
        std::string varname(ptr, ptr + name_len - 4);
        ptr += name_len;

        uint16_t compr_method =
            *reinterpret_cast<const uint16_t*>(&local_header[0] + 8);
        uint64_t compr_bytes =
            *reinterpret_cast<const uint32_t*>(&local_header[0] + 18);
        uint64_t uncompr_bytes =
            *reinterpret_cast<const uint32_t*>(&local_header[0] + 22);

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        if (extra_field_len > 0) {
            // std::vector<char> buff(extra_field_len);
            const char* buff = ptr;
            if (compr_bytes == 0xffffffff && uncompr_bytes == 0xffffffff) {
                // Handle zip64 for large file (note: not in original cnpy)
                uint16_t header_id = *reinterpret_cast<const uint16_t*>(buff);
                // Check if really zip64
                if (header_id == 1) {
                    // Update file sizes to 64-bit values
                    uncompr_bytes =
                        *reinterpret_cast<const uint64_t*>(buff + 4);
                    compr_bytes = *reinterpret_cast<const uint64_t*>(buff + 12);
                }
            }
            ptr += extra_field_len;
        }

        if (compr_method == 0) {
            arrays[varname] = load_mem_npy_file(&ptr, ptr_end);
        } else {
            arrays[varname] =
                load_mem_npz_array(&ptr, ptr_end, compr_bytes, uncompr_bytes);
        }
    }
    return arrays;
}

cnpy::NpyArray cnpy::npz_load(const std::string& fname,
                              const std::string& varname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp) throw std::runtime_error("npz_load: Unable to open file " + fname);

    while (1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0], sizeof(char), 30, fp);
        if (header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string vname(name_len, ' ');
        size_t vname_res = fread(&vname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end() - 4, vname.end());  // erase the lagging .npy

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        fseek(fp, extra_field_len, SEEK_CUR);  // skip past the extra field

        uint16_t compr_method =
            *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes =
            *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes =
            *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if (vname == varname) {
            NpyArray array =
                (compr_method == 0)
                    ? load_the_npy_file(fp)
                    : load_the_npz_array(fp, compr_bytes, uncompr_bytes);
            fclose(fp);
            return array;
        } else {
            // skip past the data
            uint32_t size = *(uint32_t*)&local_header[18 /*22*/];
            fseek(fp, size, SEEK_CUR);
        }
    }

    fclose(fp);

    // if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name " + varname +
                             " not found in " + fname);
}

cnpy::NpyArray cnpy::npy_load(const std::string& fname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}
