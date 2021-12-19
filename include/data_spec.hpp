#pragma once

#include "camera.hpp"
#include "n3tree/n3tree.hpp"

namespace viewer::internal {
    namespace {

        struct CameraSpec {
            const int width;
            const int height;
            const float fx, fy;
            const float cx, cy;
            const float *__restrict__ transform;

            CameraSpec(const Camera &camera) : width(camera.width),
                                               height(camera.height),
                                               fx(camera.fx),
                                               fy(camera.fy),
                                               cx(camera.cx),
                                               cy(camera.cy),
                                               transform(camera.device.transform) {}
        };

        struct TreeSpec {
            torch::PackedTensorAccessor64<at::Half, 3, torch::RestrictPtrTraits> data;
            torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> child;
            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> parent;
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> offset;
            const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> scale;
            const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> sample_counts;
            const int N;
            const int N3;
            const int data_dim;
            const DataFormat data_format;
            const int capacity;

            TreeSpec(N3Tree &tree)
                    : data(tree.data.packed_accessor64<at::Half, 3, torch::RestrictPtrTraits>()),
                      child(tree.child.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()),
                      parent(tree.parent.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()),
                      offset(tree.offset.packed_accessor32<float, 1, torch::RestrictPtrTraits>()),
                      scale(tree.scale.packed_accessor32<float, 1, torch::RestrictPtrTraits>()),
                      sample_counts(tree.sample_counts.packed_accessor32<short, 2, torch::RestrictPtrTraits>()),
                      N(tree.N),
                      N3(tree.N * tree.N * tree.N),
                      data_dim(tree.data_dim),
                      data_format(tree.data_format),
                      capacity(tree.capacity) {}
        };

    }  // namespace
}  // namespace viewer::internal
