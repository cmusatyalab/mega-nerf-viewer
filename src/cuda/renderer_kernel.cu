#include "../../include/cuda/common.cuh"
#include "../../include/cuda/renderer_kernel.hpp"
#include "../../include/cuda/rt_core.cuh"
#include "../../include/data_spec.hpp"

namespace viewer {

    using internal::CameraSpec;
    using internal::TreeSpec;

// Automatically choose number of CUDA threads based on HW CUDA kernel count
    int cuda_n_threads = -1;

    __host__ void auto_cuda_threads() {
        if (~cuda_n_threads) return;
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, 0);
        const int n_cores = get_sp_cores(dev_prop);
        // Optimize number of CUDA threads per block
        if (n_cores < 2048) {
            cuda_n_threads = 256;
        }
        if (n_cores < 8192) {
            cuda_n_threads = 512;
        } else {
            cuda_n_threads = 1024;
        }
    }

    template<typename scalar_t>
    __host__ __device__ __inline__ static void screen2worlddir(
            int ix, int iy, const CameraSpec cam, scalar_t *out, scalar_t *cen) {
        scalar_t xyz[3] = {(ix + 0.5f - cam.cx) / cam.fx,
                           -(iy + 0.5f - cam.cy) / cam.fy, -1.0f};
        _mv3(cam.transform, xyz, out);
        _normalize(out);
        _copy3(cam.transform + 9, cen);
    }

    template<typename scalar_t>
    __host__ __device__ __inline__ void rodrigues(const scalar_t *__restrict__ aa,
                                                  scalar_t *__restrict__ dir) {
        scalar_t angle = _norm(aa);
        if (angle < 1e-6) return;
        scalar_t k[3];

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            k[i] = aa[i] / angle;
        }
        scalar_t cos_angle = cos(angle), sin_angle = sin(angle);
        scalar_t cross[3];
        _cross3(k, dir, cross);
        scalar_t dot = _dot3(k, dir);

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            dir[i] = dir[i] * cos_angle + cross[i] * sin_angle +
                     k[i] * dot * (1.0 - cos_angle);
        }
    }

    __global__ static void adjust_parents_and_children_kernel(
            TreeSpec tree,
            const int first_shift_index,
            const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> to_delete,
            const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> index_shifts) {
        CUDA_GET_THREAD_ID(tid, tree.capacity - first_shift_index);
        int32_t chunk_idx = tid + first_shift_index;

        int32_t parent_node_chunk_idx = tree.parent[chunk_idx] / tree.N3;
        int32_t parent_node_child_idx = tree.parent[chunk_idx] % tree.N3;

        if (to_delete[chunk_idx]) {
            tree.child[parent_node_chunk_idx][parent_node_child_idx] = 0;
        } else {
            int32_t parent_shift = index_shifts[parent_node_chunk_idx];
            int32_t child_shift = index_shifts[chunk_idx];

            tree.child[parent_node_chunk_idx][parent_node_child_idx] +=
                    (parent_shift - child_shift);

            tree.parent[chunk_idx] -=
                    (index_shifts[parent_node_chunk_idx] * tree.N3);
        }
    }

    __device__ __inline__ void generate_samples_inner(
            const TreeSpec tree,
            const RenderOptions &opt,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> samples,
            torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> cluster_indices,
            const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> grid_dim,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> min_position,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> range,
            const int idx,
            const int32_t abs_chunk_idx,
            const int32_t child_idx) {
        int32_t curr[4];
        curr[0] = abs_chunk_idx * tree.N3 + child_idx;

        uint8_t depth = 0;
        float corners[] = {0, 0, 0};

        while (true) {
#pragma unroll 3
            for (int i = 3; i > 0; --i) {
                curr[i] = curr[0] % tree.N;
                curr[0] /= tree.N;
            }

#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                corners[i] += curr[i + 1];
                corners[i] /= tree.N;
            }

            if (curr[0] == 0) break;
            curr[0] = tree.parent[curr[0]];
            depth += 1;
        }

        float length_local = pow(tree.N, -depth - 1);

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            corners[i] -= tree.offset[i];
            corners[i] /= tree.scale[i];

            for (int j = 0; j < opt.samples_per_corner; j++) {
                samples[idx][j][i] *= (length_local / tree.scale[i]);
                samples[idx][j][i] += corners[i];
            }
        }

        if (opt.need_viewdir) {
            for (int j = 0; j < opt.samples_per_corner; j++) {
                // Top down seems to be a sensible default direction
                // Could alternatively sample with random directions
                samples[idx][j][3] = 1;
                samples[idx][j][4] = 0;
                samples[idx][j][5] = 0;

                if (opt.appearance_embedding != -1) {
                    samples[idx][j][6] = opt.appearance_embedding;
                }
            }
        } else if (opt.appearance_embedding != -1) {
            for (int j = 0; j < opt.samples_per_corner; j++) {
                samples[idx][j][3] = opt.appearance_embedding;
            }
        }

        int grid_1;
        int grid_2;

        for (int j = 0; j < opt.samples_per_corner; j++) {
            grid_1 = (int) fmax(
                    fmin((samples[idx][j][1] - min_position[1]) / range[1] * grid_dim[0],
                         grid_dim[0] - 1.0f),
                    0.0f);
            grid_2 = (int) fmax(
                    fmin((samples[idx][j][2] - min_position[2]) / range[2] * grid_dim[1],
                         grid_dim[1] - 1.0f),
                    0.0f);
            cluster_indices[idx][j] = grid_1 * grid_dim[1] + grid_2;
        }
    }

    __global__ static void add_children_and_generate_samples_kernel(
            TreeSpec tree,
            const RenderOptions opt,
            const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> parent_nodes,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> samples,
            torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> cluster_indices,
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> grid_dim,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> min_position,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> range,
            const int num_parents) {
        CUDA_GET_THREAD_ID(tid, num_parents * tree.N3);
        int32_t rel_chunk_idx = tid / tree.N3;
        int32_t abs_chunk_idx = tree.capacity + rel_chunk_idx;
        int32_t child_idx = tid % tree.N3;

        int32_t parent_node_chunk_idx = parent_nodes[rel_chunk_idx][0];
        int32_t parent_node_child_idx = parent_nodes[rel_chunk_idx][1];

        if (child_idx == 0) {
            tree.child[parent_node_chunk_idx][parent_node_child_idx] = abs_chunk_idx - parent_node_chunk_idx;
            tree.parent[abs_chunk_idx] = parent_node_chunk_idx * tree.N3 + parent_node_child_idx;
            visited[abs_chunk_idx] = visited[parent_node_chunk_idx];
        }

        tree.child[abs_chunk_idx][child_idx] = 0;
        generate_samples_inner(tree, opt, samples, cluster_indices, grid_dim, min_position, range, tid, abs_chunk_idx,
                               child_idx);
    }

    __global__ static void generate_samples_kernel(
            const TreeSpec tree,
            const RenderOptions opt,
            const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> nodes,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> samples,
            torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> cluster_indices,
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> grid_dim,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> min_position,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> range,
            const int num_items) {
        CUDA_GET_THREAD_ID(tid, num_items);
        generate_samples_inner(tree, opt, samples, cluster_indices, grid_dim, min_position, range, tid, nodes[tid][0],
                               nodes[tid][1]);
    }

    __device__ __inline__ void composite_and_write(
            const int x,
            const int y,
            const RenderOptions &opt,
            cudaSurfaceObject_t surf_obj,
            float *out,
            const uint8_t *rgbx_init,
            const bool offscreen) {
        // Compositing with existing color
        const float nalpha = 1.f - out[3];
        if (offscreen) {
            const float remain = opt.background_brightness * nalpha;
            out[0] += remain;
            out[1] += remain;
            out[2] += remain;
        } else {
            out[0] += rgbx_init[0] / 255.f * nalpha;
            out[1] += rgbx_init[1] / 255.f * nalpha;
            out[2] += rgbx_init[2] / 255.f * nalpha;
        }

        // Output pixel color
        uint8_t rgbx[4] = {uint8_t(out[0] * 255), uint8_t(out[1] * 255), uint8_t(out[2] * 255), 255};

        // squelches out-of-bound writes
        surf2Dwrite(*reinterpret_cast<uint32_t *>(rgbx), surf_obj, x * 4, y, cudaBoundaryModeZero);
    }

    __global__ static void render_voxels_kernel(
            const TreeSpec tree,
            const CameraSpec cam,
            const RenderOptions opt,
            cudaSurfaceObject_t surf_obj,
            cudaSurfaceObject_t surf_obj_depth,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> to_split,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> to_sample,
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
            const bool track_visit,
            const bool offscreen) {
        CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
        const int x = idx % cam.width, y = idx / cam.width;

        float dir[3], cen[3], out[4];

        uint8_t rgbx_init[4];
        if (!offscreen) {
            // Read existing values for compositing (with meshes)
            surf2Dread(reinterpret_cast<uint32_t *>(rgbx_init), surf_obj, x * 4, y,
                       cudaBoundaryModeZero);
        }

        bool enable_draw = tree.N > 0;
        out[0] = out[1] = out[2] = out[3] = 0.f;

        if (enable_draw) {
            screen2worlddir(x, y, cam, dir, cen);

#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                cen[i] = tree.offset[i] + tree.scale[i] * cen[i];
            }

            float t_max = 1e9f;
            if (!offscreen) {
                surf2Dread(&t_max, surf_obj_depth, x * sizeof(float), y, cudaBoundaryModeZero);
            }

            float vdir[3] = {dir[0], dir[1], dir[2]};
            rodrigues(opt.rot_dirs, vdir);

            device::render_voxels_trace_ray(tree, visited, dir, vdir, cen, opt, t_max, out,
                                            &to_split[idx][1], &to_split[idx][2],
                                            &to_split[idx][0], &to_sample[idx][1],
                                            &to_sample[idx][2], &to_sample[idx][0], track_visit);
        }

        composite_and_write(x, y, opt, surf_obj, out, rgbx_init, offscreen);
    }

    __global__ static void render_nerf_results_kernel(
            const TreeSpec tree,
            const CameraSpec cam,
            const RenderOptions opt,
            cudaSurfaceObject_t surf_obj,
            const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sample_values,
            const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> z_vals,
            const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> offsets,
            const bool offscreen) {
        CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
        const int x = idx % cam.width, y = idx / cam.width;

        float dir[3], cen[3], out[4];

        uint8_t rgbx_init[4];
        if (!offscreen) {
            // Read existing values for compositing (with meshes)
            surf2Dread(reinterpret_cast<uint32_t *>(rgbx_init), surf_obj, x * 4, y,
                       cudaBoundaryModeZero);
        }

        out[0] = out[1] = out[2] = 0.f;
        out[3] = 1.0f;

        screen2worlddir(x, y, cam, dir, cen);
        float vdir[3] = {dir[0], dir[1], dir[2]};

        rodrigues(opt.rot_dirs, vdir);

        device::composite_nerf_results(tree, vdir, opt, (idx == 0 ? 0 : offsets[idx - 1]), offsets[idx], sample_values,
                                       z_vals, out);

        composite_and_write(x, y, opt, surf_obj, out, rgbx_init, offscreen);
    }

    __global__ static void get_samples_from_voxels_kernel(
            TreeSpec tree,
            CameraSpec cam,
            const RenderOptions opt,
            cudaSurfaceObject_t surf_obj_depth,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> to_split,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> to_sample,
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
            const bool track_visit,
            const bool offscreen,
            torch::PackedTensorAccessor32<short, 1, torch::RestrictPtrTraits> num_samples,
            torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> samples,
            torch::PackedTensorAccessor64<short, 2, torch::RestrictPtrTraits> cluster_indices,
            const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> grid_dim,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> min_position,
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> range) {
        CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
        const int x = idx % cam.width, y = idx / cam.width;

        float dir[3], cen[3];

        screen2worlddir(x, y, cam, dir, cen);
        float vdir[3] = {dir[0], dir[1], dir[2]};
        rodrigues(opt.rot_dirs, vdir);

        float t_max = 1e9f;
        if (!offscreen) {
            surf2Dread(&t_max, surf_obj_depth, x * sizeof(float), y, cudaBoundaryModeZero);
        }

        device::get_samples_trace_ray(tree, visited, dir, vdir, cen, opt, t_max, &to_split[idx][1], &to_split[idx][2],
                                      &to_split[idx][0], &to_sample[idx][1], &to_sample[idx][2], &to_sample[idx][0],
                                      track_visit, &num_samples[idx], samples, cluster_indices, idx, grid_dim,
                                      min_position, range);
    }

    __host__ void render_nerf_results(
            N3Tree &tree,
            const Camera &cam,
            const RenderOptions &opt,
            cudaArray_t &image_arr,
            cudaStream_t &stream,
            const torch::Tensor &sample_values,
            const torch::Tensor &z_vals,
            const torch::Tensor &offsets,
            const bool offscreen) {
        auto_cuda_threads();

        cudaSurfaceObject_t surf_obj = 0;

        {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = image_arr;
            cudaCreateSurfaceObject(&surf_obj, &res_desc);
        }

        const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, cuda_n_threads);
        render_nerf_results_kernel<<<blocks, cuda_n_threads, 0, stream>>>(
                tree, cam, opt, surf_obj,
                sample_values.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                z_vals.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                offscreen);
    }

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
            const bool track_visit,
            const bool offscreen) {
        auto_cuda_threads();

        cudaSurfaceObject_t surf_obj = 0, surf_obj_depth = 0;

        {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = image_arr;
            cudaCreateSurfaceObject(&surf_obj, &res_desc);
        }

        if (!offscreen) {
            {
                struct cudaResourceDesc res_desc;
                memset(&res_desc, 0, sizeof(res_desc));
                res_desc.resType = cudaResourceTypeArray;
                res_desc.res.array.array = depth_arr;
                cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
            }
        }

        const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, cuda_n_threads);
        render_voxels_kernel<<<blocks, cuda_n_threads, 0, stream>>>(
                tree, cam, opt, surf_obj, surf_obj_depth,
                to_split.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                to_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                visited.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                track_visit, offscreen);
    }

    __host__ void get_samples_from_voxels(
            N3Tree &tree,
            const Camera &cam,
            const RenderOptions &opt,
            cudaArray_t &depth_arr,
            cudaStream_t &stream,
            const torch::Tensor &to_split,
            const torch::Tensor &to_sample,
            const torch::Tensor &visited,
            const bool track_visit,
            const bool offscreen,
            const torch::Tensor &num_samples,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range) {
        auto_cuda_threads();

        cudaSurfaceObject_t surf_obj_depth = 0;

        if (!offscreen) {
            {
                struct cudaResourceDesc res_desc;
                memset(&res_desc, 0, sizeof(res_desc));
                res_desc.resType = cudaResourceTypeArray;
                res_desc.res.array.array = depth_arr;
                cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
            }
        }

        const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, cuda_n_threads);
        get_samples_from_voxels_kernel<<<blocks, cuda_n_threads,
        0, stream>>>(
                tree, cam, opt, surf_obj_depth,
                to_split.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                to_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                visited.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                track_visit,
                offscreen,
                num_samples.packed_accessor32<short, 1, torch::RestrictPtrTraits>(),
                samples.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
                cluster_indices.packed_accessor64<short, 2, torch::RestrictPtrTraits>(),
                grid_dim.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                min_position.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                range.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
    }

    __host__ void add_children_and_generate_samples(
            N3Tree &tree,
            const RenderOptions &opt,
            const torch::Tensor &parent_nodes,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &visited,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range) {
        auto_cuda_threads();

        const int blocks = N_BLOCKS_NEEDED(
                parent_nodes.size(0) * tree.N * tree.N * tree.N, cuda_n_threads);
        add_children_and_generate_samples_kernel<<<
        blocks, cuda_n_threads>>>(tree, opt,
                                  parent_nodes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                  samples.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                  cluster_indices.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
                                  visited.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                  grid_dim.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                  min_position.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  range.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  parent_nodes.size(0));
    }

    __host__ void generate_samples(
            N3Tree &tree,
            const RenderOptions &opt,
            const torch::Tensor &nodes,
            const torch::Tensor &samples,
            const torch::Tensor &cluster_indices,
            const torch::Tensor &grid_dim,
            const torch::Tensor &min_position,
            const torch::Tensor &range) {
        auto_cuda_threads();

        const int blocks = N_BLOCKS_NEEDED(nodes.size(0), cuda_n_threads);
        generate_samples_kernel<<<blocks, cuda_n_threads>>>(
                tree, opt,
                nodes.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                samples.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                cluster_indices
                        .packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
                grid_dim.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                min_position.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                range.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                nodes.size(0));
    }

    __host__ void adjust_parents_and_children(
            N3Tree &tree,
            const int first_shift_index,
            const torch::Tensor &to_delete,
            const torch::Tensor &index_shifts) {
        auto_cuda_threads();

        const int blocks =
                N_BLOCKS_NEEDED(tree.capacity - first_shift_index, cuda_n_threads);
        adjust_parents_and_children_kernel<<<blocks, cuda_n_threads>>>(
                tree, first_shift_index,
                to_delete.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                index_shifts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
    }
}  // namespace viewer