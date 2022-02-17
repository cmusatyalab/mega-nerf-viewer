#include "../../include/n3tree/n3tree.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

namespace viewer {

    N3Tree::N3Tree() {}

    N3Tree::N3Tree(const std::string &path) { open(path); }

    void N3Tree::open(const std::string &path) {
        assert(path.size() > 3 && path.substr(path.size() - 4) == ".npz");

        if (!std::ifstream(path)) {
            printf("Can't load because file does not exist: %s\n", path.c_str());
            return;
        }

        cnpy::npz_t npz = cnpy::npz_load(path);
        load_npz(npz);
    }

    void N3Tree::load_npz(cnpy::npz_t &npz) {
        data_dim = (int) *npz["data_dim"].data<int64_t>();
        auto &df_node = npz["data_format"];
        std::string data_format_str = std::string(df_node.data_holder.begin(),
                                                  df_node.data_holder.end());
        // Unicode to ASCII
        for (size_t i = 4; i < data_format_str.size(); i += 4) {
            data_format_str[i / 4] = data_format_str[i];
        }
        data_format_str.resize(data_format_str.size() / 4);
        data_format.parse(data_format_str);

        std::cout << "Data format " << data_format.to_string() << std::endl;

        scale = torch::empty(
                {3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        if (npz.count("invradius3")) {
            const float *scale_data = npz["invradius3"].data<float>();
            for (int i = 0; i < 3; ++i) scale[i] = scale_data[i];
        } else {
            scale[0] = scale[1] = scale[2] =
                    (float) *npz["invradius"].data<double>();
        }

        std::cout << "Scale: " << scale[0].item().toFloat() << " " << scale[1].item().toFloat() << " "
                  << scale[2].item().toFloat() << std::endl;

        offset = torch::empty(
                {3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        const float *offset_data = npz["offset"].data<float>();
        for (int i = 0; i < 3; ++i) offset[i] = offset_data[i];

        std::cout << "Offset: " << offset[0].item().toFloat() << " " << offset[1].item().toFloat() << " "
                  << offset[2].item().toFloat() << std::endl;

        auto radius = 0.5 / scale;
        auto center = radius * (1 - 2 * offset);
        auto range_min = center - radius;
        auto range_max = center + radius;

        std::cout << "Center: [" << center[0].item().toFloat() << ", " << center[1].item().toFloat() << ", "
                  << center[2].item().toFloat() << "]" << std::endl;

        std::cout << "Radius: [" << radius[0].item().toFloat() << ", " << radius[1].item().toFloat() << ", "
                  << radius[2].item().toFloat() << "]" << std::endl;

        std::cout << "Range: [" << range_min[0].item().toFloat() << ", " << range_min[1].item().toFloat() << ", "
                  << range_min[2].item().toFloat() << "], [" << range_max[0].item().toFloat() << ", "
                  << range_max[1].item().toFloat() << ", " << range_max[2].item().toFloat() << "]" << std::endl;

        cnpy::NpyArray &child_node = npz["child"];

        N = child_node.shape[1];
        if (N != 2) {
            std::cout << "WARNING: N != 2 probably doesn't work." << std::endl;
        }

        N2_ = N * N;
        N3_ = N * N * N;

        child = torch::from_blob(child_node.data<int32_t>(),
                                 {static_cast<long>(child_node.shape[0]), N3_},
                                 torch::TensorOptions()
                                         .dtype(torch::kInt)
                                         .device(torch::kCPU))
                .clone();

        cnpy::NpyArray &parent_node = npz["parent_depth"];
        assert(parent_node.word_size == 4);
        parent = torch::empty(
                {static_cast<long>(parent_node.shape[0])},
                torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
        int32_t *parent_depth_ptr = parent_node.data<int32_t>();
        for (int i = 0; i < parent_node.shape[0]; ++i) {
            parent[i] = parent_depth_ptr[i * 2];
        }

        if (npz.count("quant_colors")) {
            std::cout << "Decoding quantized colors" << std::endl;
            auto &quant_colors_node = npz["quant_colors"];

            if (quant_colors_node.word_size != 2) {
                throw std::runtime_error("codebook must be stored in half precision");
            }

            auto &quant_map_node = npz["quant_map"];
            int n_basis = quant_map_node.shape[0];

            if (quant_colors_node.shape[0] != n_basis) {
                throw std::runtime_error("codebook and map basis numbers does not match");
            }

            int n_basis_retain = npz.count("data_retained") ? npz["data_retained"].shape[0] : 0;
            n_basis += n_basis_retain;

            long capacity = quant_map_node.shape[1];
            data = torch::empty({capacity, N3_, data_dim},
                                torch::TensorOptions()
                                        .dtype(torch::kFloat16)
                                        .device(torch::kCPU));

            // Decode quantized
            const uint16_t *quant_map_ptr = quant_map_node.data<uint16_t>();
            const at::Half *quant_colors_ptr = quant_colors_node.data<at::Half>();

            for (int i = 0; i < capacity; i++) {
                for (int j = 0; j < N3_; j++) {
                    for (int basis = n_basis_retain; basis < n_basis; basis++) {
                        size_t subPtr = basis * capacity * N3_ + i * N3_ + j;
                        uint16_t id = quant_map_ptr[subPtr];
                        const at::Half *colors_ptr =
                                quant_colors_ptr + basis * 65536 * 3 + id * 3;
                        for (int channel = 0; channel < 3; channel++) {
                            data[i][j][channel * n_basis] = colors_ptr[channel];
                        }
                    }
                }
            }

            if (n_basis_retain) {
                auto &retain_node = npz["data_retained"];
                const at::Half *retain_ptr = retain_node.data<at::Half>();

                for (int i = 0; i < capacity; i++) {
                    for (int j = 0; j < N3_; j++) {
                        for (int basis = 0; basis < n_basis_retain; basis++) {
                            size_t subPtr = basis * capacity * N3_ + i * N3_ + j;
                            const at::Half *colors_ptr = retain_ptr + subPtr;
                            for (int channel = 0; channel < 3; channel++) {
                                data[i][j][channel * n_basis] = colors_ptr[channel];
                            }
                        }
                    }
                }
            }

            auto &sigma_node = npz["sigma"];
            const at::Half *sigma_ptr = sigma_node.data<at::Half>();
            for (int i = 0; i < capacity; i++) {
                for (int j = 0; j < N3_; j++) {
                    size_t subPtr = i * N3_ + j;
                    data[i][j][data_dim - 1] = sigma_ptr[subPtr];
                }
            }
        } else {
            auto &data_node = npz["data"];
            long capacity = data_node.shape[0];
            if (data_node.word_size != 2) {
                throw std::runtime_error("data must be stored in half precision");
            }

            data = torch::from_blob(data_node.data<at::Half>(),
                                    {capacity, N3_, data_dim},
                                    torch::TensorOptions()
                                            .dtype(torch::kFloat16)
                                            .device(torch::kCPU))
                    .clone();
        }

        sample_counts = 8 * torch::ones(
                {data.size(0), N3_},
                torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));

        if (data.size(0) != parent.size(0)) {
            throw std::runtime_error("data and parent sizes not aligned");
        }

        if (data.size(0) != child.size(0)) {
            throw std::runtime_error("data and child sizes not aligned");
        }

        std::cout << "Data size: " << data.size(0) << std::endl;
        capacity = data.size(0);
    }

    void N3Tree::move_to_device(long max_capacity, bool need_parent, bool need_sample_counts) {
        torch::Tensor new_data =
                torch::empty({max_capacity, N3_, data_dim},
                             torch::TensorOptions()
                                     .device(torch::kCUDA)
                                     .dtype(torch::kFloat16));

        new_data.slice(0, 0, capacity) = data;
        data = new_data;

        torch::Tensor new_child = torch::empty({max_capacity, N3_},
                                               torch::TensorOptions()
                                                       .device(torch::kCUDA)
                                                       .dtype(torch::kInt32));

        new_child.slice(0, 0, capacity) = child;
        child = new_child;

        if (need_parent) {
            torch::Tensor new_parent = torch::empty({max_capacity},
                                                    torch::TensorOptions()
                                                            .device(torch::kCUDA)
                                                            .dtype(torch::kInt32));

            new_parent.slice(0, 0, capacity) = parent;
            parent = new_parent;
        }

        if (need_sample_counts) {
            torch::Tensor new_sample_counts = torch::empty(
                    {max_capacity, N3_}, torch::TensorOptions()
                            .device(torch::kCUDA)
                            .dtype(torch::kInt16));
            sample_counts = new_sample_counts;
        }


        scale = scale.to(torch::kCUDA);
        offset = offset.to(torch::kCUDA);
    }

    namespace {
        void _push_wireframe_bb(const float bb[6], std::vector<float> &verts_out) {
#define PUSH_VERT(i, j, k)              \
    verts_out.push_back(bb[i * 3]);     \
    verts_out.push_back(bb[j * 3 + 1]); \
    verts_out.push_back(bb[k * 3 + 2]); \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(0);             \
    verts_out.push_back(1)
            // clang-format off
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    PUSH_VERT(0, i, j);
                    PUSH_VERT(1, i, j);
                    PUSH_VERT(i, 0, j);
                    PUSH_VERT(i, 1, j);
                    PUSH_VERT(i, j, 0);
                    PUSH_VERT(i, j, 1);
                }
            }
            // clang-format on
#undef PUSH_VERT
        }

        void _gen_wireframe_impl(const N3Tree &tree,
                                 int32_t nodeid,
                                 size_t xi,
                                 size_t yi,
                                 size_t zi,
                                 int depth,
                                 size_t gridsz,
                                 int max_depth,
                                 std::vector<float> &verts_out) {
            int cnt = 0;
            // Use integer coords to avoid precision issues
            for (size_t i = xi * tree.N; i < (xi + 1) * tree.N; ++i) {
                for (size_t j = yi * tree.N; j < (yi + 1) * tree.N; ++j) {
                    for (size_t k = zi * tree.N; k < (zi + 1) * tree.N; ++k) {
                        int32_t child = tree.child[nodeid][cnt].item().toInt();
                        if (child == 0 || depth >= max_depth) {
                            // Add this cube
                            const float bb[6] = {
                                    ((float) i / gridsz -
                                     tree.offset[0].item().toFloat()) /
                                    tree.scale[0].item().toFloat(),
                                    ((float) j / gridsz -
                                     tree.offset[1].item().toFloat()) /
                                    tree.scale[1].item().toFloat(),
                                    ((float) k / gridsz -
                                     tree.offset[2].item().toFloat()) /
                                    tree.scale[2].item().toFloat(),
                                    ((float) (i + 1) / gridsz -
                                     tree.offset[0].item().toFloat()) /
                                    tree.scale[0].item().toFloat(),
                                    ((float) (j + 1) / gridsz -
                                     tree.offset[1].item().toFloat()) /
                                    tree.scale[1].item().toFloat(),
                                    ((float) (k + 1) / gridsz -
                                     tree.offset[2].item().toFloat()) /
                                    tree.scale[2].item().toFloat()};
                            _push_wireframe_bb(bb, verts_out);
                        } else {
                            _gen_wireframe_impl(tree, nodeid + child, i, j, k,
                                                depth + 1, gridsz * tree.N, max_depth,
                                                verts_out);
                        }
                        ++cnt;
                    }
                }
            }
        }  // namespace
    }  // namespace

    std::vector<float> N3Tree::gen_wireframe(int max_depth) const {
        std::vector<float> verts;
        _gen_wireframe_impl(*this, 0, 0, 0, 0,
                /*depth*/ 0, N, max_depth, verts);
        return verts;
    }

    int64_t N3Tree::pack_index(int nd, int i, int j, int k) {
        assert(i < N && j < N && k < N && i >= 0 && j >= 0 && k >= 0);
        int64_t result = nd * N3_ + i * N2_ + j * N + k;
        return result;
    }

    std::tuple<int, int, int, int> N3Tree::unpack_index(int64_t packed) {
        int k = packed % N;
        packed /= N;
        int j = packed % N;
        packed /= N;
        int i = packed % N;
        packed /= N;
        return std::tuple < int, int, int, int > {packed, i, j, k};
    }

}  // namespace viewer
