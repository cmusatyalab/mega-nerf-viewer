#include "../include/imwrite.h"

#include <png.h>
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>

bool write_png_file(const std::string &filename,
                    uint8_t *ptr,
                    int width,
                    int height,
                    int buf_width) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "PNG destination could not be opened" << std::endl;
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        std::cerr << "PNG write failed" << std::endl;
        return false;
    }

    png_set_compression_level(png, 0);
    png_set_compression_strategy(png, Z_HUFFMAN_ONLY);
    png_set_filter_heuristics(png, PNG_FILTER_NONE, 0, 0, 0);

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "PNG write failed" << std::endl;
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "PNG write failed" << std::endl;
        return false;
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    if (!ptr) {
        std::cerr << "PNG write failed" << std::endl;
        return false;
    }

    std::vector<uint8_t *> row_ptrs(height);
    for (int i = 0; i < height; ++i) {
        row_ptrs[i] = ptr + i * buf_width * 4;
    }

    png_write_image(png, row_ptrs.data());
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
    return true;
}
