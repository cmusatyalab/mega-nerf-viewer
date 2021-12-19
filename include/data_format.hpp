#pragma once

#include <string>

namespace viewer {

struct DataFormat {
    enum {
        RGBA,  // Simply stores rgba
        SH,
        _COUNT,
    } format;

    // SH dimension per channel
    int basis_dim = -1;

    // Parse a string like 'SH16'
    void parse(const std::string& str);

    // Convert to string
    std::string to_string() const;
};

}  // namespace viewer
