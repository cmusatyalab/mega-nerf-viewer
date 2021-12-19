#pragma once

#include <cuda_runtime.h>

#include "../camera.hpp"
#include "../n3tree/n3tree.hpp"
#include "../render_options.hpp"
#include "torch/torch.h"

namespace viewer {

    __host__ void render_nerf_results(
            N3Tree &tree,
            const Camera &cam,
            const RenderOptions &opt,
            cudaArray_t &image_arr,
            cudaStream_t &stream,
            const torch::Tensor &sample_values,
            const torch::Tensor &z_vals,
            const torch::Tensor &offsets,
            bool offscreen);

    __host__ void render_voxels(
            N3Tree &tree,
            const Camera &cam,
            const RenderOptions &opt,
            cudaArray_t &image_arr,
            cudaArray_t &depth_arr,
            cudaStream_t &stream,
            const torch::Tensor &to_split,
            const torch::Tensor &to_sample,
            const torch::Tensor &visited,
            bool track_visit,
            bool offscreen);

    __host__ void get_samples_from_voxels(
            N3Tree &tree,
            const Camera &cam,
            const RenderOptions &opt,
            cudaArray_t &depth_arr,
            cudaStream_t &stream,
            const torch::Tensor &to_split,
            const torch::Tensor &to_sample,
            const torch::Tensor &visited,
            bool track_visit,
            bool offscreen,
            const torch::Tensor &num_samples,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range);

    __host__ void add_children_and_generate_samples(
            N3Tree &tree,
            const RenderOptions &opt,
            const torch::Tensor &parent_nodes,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &visited,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range);

    __host__ void generate_samples(
            N3Tree &tree,
            const RenderOptions &opt,
            const torch::Tensor &nodes,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range);

    __host__ void adjust_parents_and_children(
            N3Tree &tree,
            int first_shift_index,
            const torch::Tensor &to_delete,
            const torch::Tensor &index_shifts);

}  // namespace viewer
