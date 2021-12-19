#include <string>

bool write_png_file(const std::string &filename,
                    uint8_t *ptr,
                    int width,
                    int height,
                    int buf_width);