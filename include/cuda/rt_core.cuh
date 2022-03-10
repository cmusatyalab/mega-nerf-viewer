#pragma once

#include "../../include/cuda/rt_core.cuh"
#include "../data_spec.hpp"
#include "../render_options.hpp"

namespace viewer {
    namespace device {

        namespace {

            template<typename scalar_t>
            __device__ __inline__ void maybe_precalc_basis(
                    const internal::TreeSpec &__restrict__ tree,
                    const scalar_t *__restrict__ dir,
                    scalar_t *__restrict__ out) {
                const int basis_dim = tree.data_format.basis_dim;
                switch (tree.data_format.format) {
                    case DataFormat::SH: {
                        // SH Coefficients from
                        // https://github.com/google/spherical-harmonics
                        out[0] = 0.28209479177387814;
                        const scalar_t x = dir[0], y = dir[1], z = dir[2];
                        const scalar_t xx = x * x, yy = y * y, zz = z * z;
                        const scalar_t xy = x * y, yz = y * z, xz = x * z;
                        switch (basis_dim) {
                            case 25:
                                out[16] = 2.5033429417967046 * xy * (xx - yy);
                                out[17] = -1.7701307697799304 * yz * (3 * xx - yy);
                                out[18] = 0.9461746957575601 * xy * (7 * zz - 1.f);
                                out[19] = -0.6690465435572892 * yz * (7 * zz - 3.f);
                                out[20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3);
                                out[21] = -0.6690465435572892 * xz * (7 * zz - 3);
                                out[22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.f);
                                out[23] = -1.7701307697799304 * xz * (xx - 3 * yy);
                                out[24] = 0.6258357354491761 *
                                          (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
                                [[fallthrough]];
                            case 16:
                                out[9] = -0.5900435899266435 * y * (3 * xx - yy);
                                out[10] = 2.890611442640554 * xy * z;
                                out[11] = -0.4570457994644658 * y * (4 * zz - xx - yy);
                                out[12] =
                                        0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
                                out[13] = -0.4570457994644658 * x * (4 * zz - xx - yy);
                                out[14] = 1.445305721320277 * z * (xx - yy);
                                out[15] = -0.5900435899266435 * x * (xx - 3 * yy);
                                [[fallthrough]];
                            case 9:
                                out[4] = 1.0925484305920792 * xy;
                                out[5] = -1.0925484305920792 * yz;
                                out[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
                                out[7] = -1.0925484305920792 * xz;
                                out[8] = 0.5462742152960396 * (xx - yy);
                                [[fallthrough]];
                            case 4:
                                out[1] = -0.4886025119029199 * y;
                                out[2] = 0.4886025119029199 * z;
                                out[3] = -0.4886025119029199 * x;
                        }
                    }  // SH
                        break;

                    default:
                        // Do nothing
                        break;
                }  // switch
            }

            template<typename scalar_t>
            __device__ __inline__ void _dda_world(const scalar_t *__restrict__ cen,
                                                  const scalar_t *__restrict__ _invdir,
                                                  scalar_t *__restrict__ tmin,
                                                  scalar_t *__restrict__ tmax,
                                                  const float *__restrict__ render_bbox) {
                scalar_t t1, t2;
                *tmin = 0.0;
                *tmax = 1e4;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    t1 = (render_bbox[i] + 1e-6 - cen[i]) * _invdir[i];
                    t2 = (render_bbox[i + 3] - 1e-6 - cen[i]) * _invdir[i];
                    *tmin = max(*tmin, min(t1, t2));
                    *tmax = min(*tmax, max(t1, t2));
                }
            }

            template<typename scalar_t>
            __device__ __inline__ scalar_t _dda_unit(const scalar_t *__restrict__ cen,
                                                     const scalar_t *__restrict__ _invdir) {
                scalar_t t1, t2;
                scalar_t tmax = 1e4;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    t1 = -cen[i] * _invdir[i];
                    t2 = t1 + _invdir[i];
                    tmax = min(tmax, max(t1, t2));
                }
                return tmax;
            }

            template<typename scalar_t>
            __device__ __inline__ scalar_t _get_delta_scale(
                    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
                    scaling,
                    scalar_t *__restrict__ dir) {
                dir[0] *= scaling[0];
                dir[1] *= scaling[1];
                dir[2] *= scaling[2];
                scalar_t delta_scale = 1.f / _norm(dir);
                dir[0] *= delta_scale;
                dir[1] *= delta_scale;
                dir[2] *= delta_scale;
                return delta_scale;
            }

            __device__ __inline__ void query_single_from_root(
                    const internal::TreeSpec &tree,
                    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
                    float *__restrict__ xyz,
                    int32_t *__restrict__ chunk_idx,
                    int32_t *__restrict__ child_idx,
                    int32_t *__restrict__ depth,
                    const bool track_visit) {
                xyz[0] = max(min(xyz[0], 1.f - 1e-6f), 0.f);
                xyz[1] = max(min(xyz[1], 1.f - 1e-6f), 0.f);
                xyz[2] = max(min(xyz[2], 1.f - 1e-6f), 0.f);

                int32_t cur_chunk_idx = 0;
                *depth = 1;
                while (true) {
                    if (track_visit) {
                        atomicCAS(&visited[cur_chunk_idx], 0, 1);
                    }

                    int cur_child_idx = 0;
#pragma unroll 3
                    for (int i = 0; i < 3; ++i) {
                        xyz[i] *= tree.N;
                        const float idx_dimi = floorf(xyz[i]);
                        cur_child_idx = cur_child_idx * tree.N + idx_dimi;
                        xyz[i] -= idx_dimi;
                    }

                    // Find child offset
                    const int32_t skip = tree.child[cur_chunk_idx][cur_child_idx];

                    // Add to output
                    if (skip == 0 /*|| *cube_sz >= max_cube_sz*/) {
                        *chunk_idx = cur_chunk_idx;
                        *child_idx = cur_child_idx;
                        break;
                    }

                    *depth += 1;

                    cur_chunk_idx += skip;
                }
            }
        }  // namespace

        template<typename scalar_t>
        __device__ __inline__ void render_voxels_trace_ray(
                const internal::TreeSpec &__restrict__ tree,
                torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
                scalar_t *__restrict__ dir,
                const scalar_t *__restrict__ vdir,
                const scalar_t *__restrict__ cen,
                RenderOptions opt,
                float tmax_bg,
                scalar_t *__restrict__ out,
                float *__restrict__ to_split_chunk_idx,
                float *__restrict__ to_split_child_idx,
                float *__restrict__ to_split_priority,
                float *__restrict__ to_sample_chunk_idx,
                float *__restrict__ to_sample_child_idx,
                float *__restrict__ to_sample_priority,
                bool track_visit) {
            *to_split_priority = opt.max_depth + 1;
            *to_sample_priority = opt.max_sample_count + 1;

            const float delta_scale = _get_delta_scale(tree.scale, /*modifies*/ dir);
            tmax_bg /= delta_scale;

            scalar_t tmin, tmax;
            scalar_t invdir[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                invdir[i] = 1.f / (dir[i] + 1e-9);
            }
            _dda_world(cen, invdir, &tmin, &tmax, opt.render_bbox);
            tmax = min(tmax, tmax_bg);

            if (tmax < 0 || tmin > tmax) {
                // Ray doesn't hit box
                if (opt.render_depth) out[3] = 1.f;
                return;
            } else {
                scalar_t pos[3], tmp;
                scalar_t basis_fn[VIEWER_GLOBAL_BASIS_MAX];
                maybe_precalc_basis(tree, vdir, basis_fn);

                for (int i = 0; i < opt.basis_minmax[0]; ++i) {
                    basis_fn[i] = 0.f;
                }

                for (int i = opt.basis_minmax[1] + 1; i < VIEWER_GLOBAL_BASIS_MAX; ++i) {
                    basis_fn[i] = 0.f;
                }

                scalar_t light_intensity = 1.f;
                scalar_t t = tmin;
                scalar_t max_weight = -1;
                scalar_t max_sample_weight = -1;

                int chunk_idx;
                int child_idx;
                int depth;

                while (t < tmax) {
                    pos[0] = cen[0] + t * dir[0];
                    pos[1] = cen[1] + t * dir[1];
                    pos[2] = cen[2] + t * dir[2];

                    query_single_from_root(tree, visited, pos, &chunk_idx, &child_idx, &depth, track_visit);
                    float cube_size = powf(tree.N, depth);

                    scalar_t att;
                    const scalar_t t_subcube = _dda_unit(pos, invdir) / cube_size;
                    const scalar_t delta_t = t_subcube + opt.step_size;
                    scalar_t sigma = tree.data[chunk_idx][child_idx][tree.data_dim - 1];

                    if (sigma > opt.sigma_thresh) {
                        att = expf(-delta_t * delta_scale * sigma);
                        const scalar_t weight = light_intensity * (1.f - att);

                        if (weight > max_weight && depth < opt.max_depth) {
                            *to_split_chunk_idx = chunk_idx;
                            *to_split_child_idx = child_idx;
                            *to_split_priority = depth;

                            max_weight = weight;
                        }

                        if (weight > max_sample_weight &&
                            tree.sample_counts[chunk_idx][child_idx] < opt.max_sample_count) {
                            *to_sample_chunk_idx = chunk_idx;
                            *to_sample_child_idx = child_idx;
                            *to_sample_priority = tree.sample_counts[chunk_idx][child_idx];

                            max_sample_weight = weight;
                        }

                        if (opt.render_depth) {
                            out[0] += weight * t;
                        } else {
                            if (tree.data_format.basis_dim >= 0) {
                                int off = 0;
#define MUL_BASIS_I(t) basis_fn[t] * tree.data[chunk_idx][child_idx][off + t]
#pragma unroll 3
                                for (int t = 0; t < 3; ++t) {
                                    tmp = basis_fn[0] * tree.data[chunk_idx][child_idx][off];
                                    switch (tree.data_format.basis_dim) {
                                        case 25:
                                            tmp += MUL_BASIS_I(16) + MUL_BASIS_I(17) + MUL_BASIS_I(18) +
                                                   MUL_BASIS_I(19) + MUL_BASIS_I(20) + MUL_BASIS_I(21) +
                                                   MUL_BASIS_I(22) + MUL_BASIS_I(23) + MUL_BASIS_I(24);
                                        case 16:
                                            tmp += MUL_BASIS_I(9) + MUL_BASIS_I(10) + MUL_BASIS_I(11) +
                                                   MUL_BASIS_I(12) + MUL_BASIS_I(13) + MUL_BASIS_I(14) +
                                                   MUL_BASIS_I(15);

                                        case 9:
                                            tmp += MUL_BASIS_I(4) + MUL_BASIS_I(5) + MUL_BASIS_I(6) + MUL_BASIS_I(7) +
                                                   MUL_BASIS_I(8);

                                        case 4:
                                            tmp += MUL_BASIS_I(1) + MUL_BASIS_I(2) + MUL_BASIS_I(3);
                                    }

                                    out[t] += weight / (1.f + expf(-tmp));
                                    off += tree.data_format.basis_dim;
                                }
#undef MUL_BASIS_I
                            } else {
#pragma unroll 3
                                for (int j = 0; j < 3; ++j) {
                                    out[j] += tree.data[chunk_idx][child_idx][j] * weight;
                                }
                            }
                        }

                        light_intensity *= att;

                        if (light_intensity < opt.stop_thresh) {
                            // Almost full opacity, stop
                            if (opt.render_depth) {
                                out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
                            }

                            scalar_t scale = 1.f / (1.f - light_intensity);
                            out[0] *= scale;
                            out[1] *= scale;
                            out[2] *= scale;
                            out[3] = 1.f;
                            return;
                        }
                    } else {
                        if (max_weight == -1 && depth < opt.max_depth) {
                            *to_split_chunk_idx = chunk_idx;
                            *to_split_child_idx = child_idx;
                            *to_split_priority = depth;
                        }

                        if (max_sample_weight == -1 &&
                            tree.sample_counts[chunk_idx][child_idx] < opt.max_sample_count) {
                            *to_sample_chunk_idx = chunk_idx;
                            *to_sample_child_idx = child_idx;
                            *to_sample_priority = tree.sample_counts[chunk_idx][child_idx];
                        }
                    }

                    t += delta_t;
                }
                if (opt.render_depth) {
                    out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
                    out[3] = 1.f;
                } else {
                    out[3] = 1.f - light_intensity;
                }
            }
        }

        template<typename scalar_t>
        __device__ __inline__ void composite_nerf_results(
                const internal::TreeSpec &__restrict__ tree,
                const scalar_t *__restrict__ vdir,
                RenderOptions opt,
                const int64_t start,
                const int64_t end,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sample_values,
                const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> z_vals,
                scalar_t *__restrict__ out) {
            if (start == end) {
                return;
            }

            scalar_t basis_fn[VIEWER_GLOBAL_BASIS_MAX];
            maybe_precalc_basis(tree, vdir, basis_fn);

            for (int i = 0; i < opt.basis_minmax[0]; ++i) {
                basis_fn[i] = 0.f;
            }

            for (int i = opt.basis_minmax[1] + 1; i < VIEWER_GLOBAL_BASIS_MAX; ++i) {
                basis_fn[i] = 0.f;
            }

            scalar_t ti = 1;

            scalar_t delta_i, weight_component, weight, tmp;
            for (int64_t i = start; i < end; i++) {
                if (i < end - 1) {
                    delta_i = z_vals[i + 1] - z_vals[i];
                    weight_component = expf(-sample_values[i][3] * delta_i);
                    weight = ti * (1.0f - weight_component);
                } else {
                    weight = ti;
                }

                if (opt.render_depth) {
                    out[0] += weight * ti;
                } else {
                    if (tree.data_format.basis_dim >= 0) {
                        int off = 0;
#define MUL_BASIS_I(t) basis_fn[t] * sample_values[i][off + t]
#pragma unroll 3
                        for (int t = 0; t < 3; ++t) {
                            tmp = basis_fn[0] * sample_values[i][off];
                            switch (tree.data_format.basis_dim) {
                                case 25:
                                    tmp += MUL_BASIS_I(16) + MUL_BASIS_I(17) + MUL_BASIS_I(18) +
                                           MUL_BASIS_I(19) + MUL_BASIS_I(20) + MUL_BASIS_I(21) +
                                           MUL_BASIS_I(22) + MUL_BASIS_I(23) + MUL_BASIS_I(24);
                                case 16:
                                    tmp += MUL_BASIS_I(9) + MUL_BASIS_I(10) + MUL_BASIS_I(11) +
                                           MUL_BASIS_I(12) + MUL_BASIS_I(13) + MUL_BASIS_I(14) +
                                           MUL_BASIS_I(15);

                                case 9:
                                    tmp += MUL_BASIS_I(4) + MUL_BASIS_I(5) + MUL_BASIS_I(6) + MUL_BASIS_I(7) +
                                           MUL_BASIS_I(8);

                                case 4:
                                    tmp += MUL_BASIS_I(1) + MUL_BASIS_I(2) + MUL_BASIS_I(3);
                            }

                            out[t] += weight / (1.f + expf(-tmp));
                            off += tree.data_format.basis_dim;
                        }
#undef MUL_BASIS_I
                    } else {
#pragma unroll 3
                        for (int j = 0; j < 3; ++j) {
                            out[j] += weight * sample_values[i][j];
                        }
                    }
                }

                ti *= weight_component;
            }

            if (opt.render_depth) {
                out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
            }
        }

        template<typename scalar_t>
        __device__ __inline__ void get_samples_trace_ray(
                const internal::TreeSpec &__restrict__ tree,
                torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> visited,
                scalar_t *__restrict__ true_dir,
                const scalar_t *__restrict__ vdir,
                const scalar_t *__restrict__ true_cen,
                RenderOptions opt,
                float tmax_bg,
                float *__restrict__ to_split_chunk_idx,
                float *__restrict__ to_split_child_idx,
                float *__restrict__ to_split_priority,
                float *__restrict__ to_sample_chunk_idx,
                float *__restrict__ to_sample_child_idx,
                float *__restrict__ to_sample_priority,
                bool track_visit,
                short *__restrict__ num_samples,
                torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> samples,
                torch::PackedTensorAccessor64<short, 2, torch::RestrictPtrTraits> cluster_indices,
                int sample_index,
                torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> grid_dim,
                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> min_position,
                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> range) {
            *to_split_priority = opt.max_depth + 1;
            *to_sample_priority = opt.max_sample_count + 1;

            scalar_t cen[3];
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                cen[i] = tree.offset[i] + tree.scale[i] * true_cen[i];
            }

            float dir[] = {true_dir[0], true_dir[1], true_dir[2]};
            const float delta_scale = _get_delta_scale(tree.scale, /*modifies*/ dir);
            tmax_bg /= delta_scale;

            scalar_t tmin, tmax;
            scalar_t invdir[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                invdir[i] = 1.f / (dir[i] + 1e-9);
            }
            _dda_world(cen, invdir, &tmin, &tmax, opt.render_bbox);
            tmax = min(tmax, tmax_bg);

            if (tmax < 0 || tmin > tmax) {
                // Ray doesn't hit box
                return;
            } else {
                scalar_t pos[3], true_z[3];

                scalar_t light_intensity = 1.f;
                scalar_t t = tmin;
                scalar_t max_weight = -1;
                scalar_t max_sample_weight = -1;

                int chunk_idx;
                int child_idx;
                int depth;
                int grid_1;
                int grid_2;

                while (t < tmax) {
                    pos[0] = cen[0] + t * dir[0];
                    pos[1] = cen[1] + t * dir[1];
                    pos[2] = cen[2] + t * dir[2];

                    query_single_from_root(tree, visited, pos, &chunk_idx, &child_idx, &depth, track_visit);
                    float cube_size = powf(tree.N, depth);

                    scalar_t att;
                    const scalar_t t_subcube = _dda_unit(pos, invdir) / cube_size;
                    const scalar_t delta_t = t_subcube + opt.step_size;
                    scalar_t sigma = tree.data[chunk_idx][child_idx][tree.data_dim - 1];

                    if (sigma > opt.sigma_thresh) {
                        att = expf(-delta_t * delta_scale * sigma);
                        const scalar_t weight = light_intensity * (1.f - att);

                        if (weight > max_weight && depth < opt.max_depth) {
                            *to_split_chunk_idx = chunk_idx;
                            *to_split_child_idx = child_idx;
                            *to_split_priority = depth;

                            max_weight = weight;
                        }

                        if (weight > max_sample_weight &&
                            tree.sample_counts[chunk_idx][child_idx] < opt.max_sample_count) {
                            *to_sample_chunk_idx = chunk_idx;
                            *to_sample_child_idx = child_idx;
                            *to_sample_priority = tree.sample_counts[chunk_idx][child_idx];

                            max_sample_weight = weight;
                        }

                        if (*num_samples < opt.max_guided_samples) {
                            true_z[0] = t * dir[0] / tree.scale[0];
                            true_z[1] = t * dir[1] / tree.scale[1];
                            true_z[2] = t * dir[2] / tree.scale[2];

                            samples[sample_index][*num_samples][0] = _norm(true_z);

                            samples[sample_index][*num_samples][1] = true_cen[0] + true_dir[0] *
                                                                                   samples[sample_index][*num_samples][0];

                            samples[sample_index][*num_samples][2] = true_cen[1] + true_dir[1] *
                                                                                   samples[sample_index][*num_samples][0];

                            samples[sample_index][*num_samples][3] = true_cen[2] + true_dir[2] *
                                                                                   samples[sample_index][*num_samples][0];

                            if (opt.need_viewdir) {
                                samples[sample_index][*num_samples][4] = vdir[0];
                                samples[sample_index][*num_samples][5] = vdir[1];
                                samples[sample_index][*num_samples][6] = vdir[2];
                                if (opt.appearance_embedding != -1) {
                                    samples[sample_index][*num_samples][7] = opt.appearance_embedding;
                                }
                            } else if (opt.appearance_embedding != -1) {
                                samples[sample_index][*num_samples][4] = opt.appearance_embedding;
                            }

                            grid_1 = (int) fmax(
                                    fmin((samples[sample_index][*num_samples][2] - min_position[1]) / range[1] *
                                         grid_dim[0], grid_dim[0] - 1.0f), 0.0f);

                            grid_2 = (int) fmax(
                                    fmin((samples[sample_index][*num_samples][3] - min_position[2]) / range[2] *
                                         grid_dim[1], grid_dim[1] - 1.0f), 0.0f);

                            cluster_indices[sample_index][*num_samples] = grid_1 * grid_dim[1] + grid_2;
                            *num_samples += 1;
                        }

                        light_intensity *= att;

                        if (light_intensity < opt.stop_thresh) {
                            return;
                        }
                    } else {
                        if (max_weight == -1 && depth < opt.max_depth) {
                            *to_split_chunk_idx = chunk_idx;
                            *to_split_child_idx = child_idx;
                            *to_split_priority = depth;
                        }

                        if (max_sample_weight == -1 &&
                            tree.sample_counts[chunk_idx][child_idx] < opt.max_sample_count) {
                            *to_sample_chunk_idx = chunk_idx;
                            *to_sample_child_idx = child_idx;
                            *to_sample_priority = tree.sample_counts[chunk_idx][child_idx];
                        }
                    }

                    t += delta_t;
                }
            }
        }
    }  // namespace device
}  // namespace viewer
