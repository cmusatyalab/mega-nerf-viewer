#pragma once

#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../data_format.hpp"
#include "cnpy.h"
#include "glm/vec3.hpp"
#include "torch/torch.h"

namespace viewer {

// N3Tree loader
struct N3Tree {
    N3Tree();
    explicit N3Tree(const std::string& path);

    // Open npz
    void open(const std::string& path);

    void move_to_device(long max_capacity, bool need_parent, bool need_sample_counts);

    // Generate wireframe (returns line vertex positions; 9 * (a-b c-d) ..)
    // assignable to Mesh.vert
    // up to given depth (default none)
    std::vector<float> gen_wireframe(int max_depth = 100000) const;

    // Spatial branching factor. Only 2 really supported.
    int N = 0;
    // Size of data stored on each leaf

    int data_dim;
    // Data format (SH, RGBA etc)

    DataFormat data_format;

    // Scaling for coordinates
    torch::Tensor scale;

    // Translation
    torch::Tensor offset;

    // Index pack/unpack
    int64_t pack_index(int nd, int i, int j, int k);
    std::tuple<int, int, int, int> unpack_index(int64_t packed);

    // Main data holder
    torch::Tensor data;

    // Child link data holder
    torch::Tensor child;

    // Child link data holder
    torch::Tensor parent;

    torch::Tensor sample_counts;

    // Number of chunks in the tree
    int capacity;

private:
    // Load data from npz (destructive since it moves some data)
    void load_npz(cnpy::npz_t& npz);

    int N2_, N3_;
};

}  // namespace viewer
