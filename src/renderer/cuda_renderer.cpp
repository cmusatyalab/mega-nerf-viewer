#include <GL/glew.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_gl_interop.h>

#include <array>
#include <memory>

#include "../../include/cuda/common.cuh"
#include "../../include/cuda/renderer_kernel.hpp"
#include "../../include/mesh.hpp"
#include "../../include/renderer/renderer.hpp"
#include "ATen/autocast_mode.h"
#include "torch/script.h"

namespace viewer {

    int PRUNE_CHUNK_SIZE = 100000;

// Starting CUDA/OpenGL interop code from
// https://gist.github.com/allanmac/4ff11985c3562830989f

    struct VolumeRenderer::Impl {
        Impl(Camera &camera, RenderOptions &options)
                : camera(camera), options(options), buf_index(0) {
            wire_.face_size = 2;
            wire_.unlit = true;
        }

        ~Impl() {
            // Unregister CUDA resources
            for (int index = 0; index < cgr.size(); index++) {
                if (cgr[index] != nullptr)
                    cuda_assert(cudaGraphicsUnregisterResource(cgr[index]),
                                __FILE__, __LINE__);
            }
            glDeleteRenderbuffers(2, rb.data());
            glDeleteRenderbuffers(2, depth_rb.data());
            glDeleteRenderbuffers(2, depth_buf_rb.data());
            glDeleteFramebuffers(2, fb.data());
            cuda_assert(cudaStreamDestroy(stream), __FILE__, __LINE__);
        }

        void start() {
            if (started_) return;
            cuda_assert(cudaStreamCreateWithFlags(&stream, cudaStreamDefault),
                        __FILE__, __LINE__);

            glCreateRenderbuffers(2, rb.data());
            // Depth buffer cannot be read in CUDA,
            // have to write fake depth buffer manually
            glCreateRenderbuffers(2, depth_rb.data());
            glCreateRenderbuffers(2, depth_buf_rb.data());
            glCreateFramebuffers(2, fb.data());

            // Attach rbo to fbo
            for (int index = 0; index < 2; index++) {
                glNamedFramebufferRenderbuffer(fb[index], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[index]);
                glNamedFramebufferRenderbuffer(fb[index], GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, depth_rb[index]);
                glNamedFramebufferRenderbuffer(fb[index], GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buf_rb[index]);
                const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
                glNamedFramebufferDrawBuffers(fb[index], 2, attach_buffers);
            }

            init_split_tracker();
            started_ = true;
        }

        void render() {
            start();
            GLfloat clear_color[] = {options.background_brightness, options.background_brightness,
                                     options.background_brightness, 1.f};

            GLfloat depth_inf = 1e9, zero = 0;
            glClearDepth(1.f);
            glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 0, clear_color);
            glClearNamedFramebufferfv(fb[buf_index], GL_COLOR, 1, &depth_inf);
            glClearNamedFramebufferfv(fb[buf_index], GL_DEPTH, 0, &depth_inf);

            camera._update();

            if (options.show_grid) {
                maybe_gen_wire(options.grid_max_depth);
            }

            glDepthMask(GL_TRUE);
            glBindFramebuffer(GL_FRAMEBUFFER, fb[buf_index]);

            if (options.show_grid) {
                wire_.draw(camera.w2c, camera.K);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            if (tree != nullptr) {
                cuda_assert(cudaGraphicsMapResources(2, &cgr[buf_index * 2], stream), __FILE__, __LINE__);

                split_tracker.fill_(-1);
                sample_tracker.fill_(-1);
                bool camera_has_changed = camera.has_changed();
                bool track_visit =
                        (camera_has_changed && tree->capacity > max_tree_capacity * 3 / 4) || prune_happened;

                if (camera_has_changed) {
                    can_reuse_results = false;
                }

                if (options.use_guided_sampling && !camera.is_dragging()) {
                    if (!can_reuse_results) {
                        num_samples.fill_(0);
                        guided_samples.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).fill_(-1);

                        get_samples_from_voxels(*tree, camera, options, ca[buf_index * 2 + 1], stream, split_tracker,
                                                sample_tracker, visit_tracker, track_visit, false, num_samples,
                                                guided_samples, cluster_indices, grid_dim, min_position, range);

                        offsets = torch::cumsum(num_samples, 0);
                        auto valid_samples = guided_samples.view({-1, guided_samples.size(-1)});
                        auto valid_cluster_indices = cluster_indices.view({-1}).index(
                                {valid_samples.index({torch::indexing::Slice(), 0}) >= 0});
                        valid_samples = valid_samples.index({valid_samples.index({torch::indexing::Slice(), 0}) >= 0});

                        auto start = std::chrono::high_resolution_clock::now();

                        auto buffer = nerf_result_buffer.slice(0, 0, valid_samples.size(0));

                        query_submodules(valid_cluster_indices, valid_samples.slice(1, 1),
                                         buffer, 32);

                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

                        std::cout << "Guided sampling finished in " << duration.count() << " ms" << std::endl;
                        z_vals = valid_samples.index({torch::indexing::Slice(), 0});
                        can_reuse_results = true;
                    }

                    auto sample_values = nerf_result_buffer.slice(0, 0, z_vals.size(0));
                    render_nerf_results(*tree, camera, options, ca[buf_index * 2], stream, sample_values, z_vals,
                                        offsets, false);
                } else {
                    render_voxels(*tree, camera, options, ca[buf_index * 2], ca[buf_index * 2 + 1], stream,
                                  split_tracker, sample_tracker, visit_tracker, track_visit, false);
                }

                if (options.use_splitting && !camera.is_dragging()) {
                    expand_voxels();
                }

                if (max_tree_capacity - tree->capacity < options.split_batch_size) {
                    prune_tree();
                    prune_happened = true;
                } else {
                    prune_happened = false;
                }

                cuda_assert(cudaGraphicsUnmapResources(2, &cgr[buf_index * 2], stream), __FILE__, __LINE__);
            }

            glNamedFramebufferReadBuffer(fb[buf_index], GL_COLOR_ATTACHMENT0);
            glBlitNamedFramebuffer(fb[buf_index], 0, 0, 0, camera.width, camera.height, 0, camera.height, camera.width,
                                   0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            buf_index ^= 1;
        }

        void query_submodules(const torch::Tensor &cluster_indices, const torch::Tensor &samples,
                              torch::Tensor &result_buffer, int batch_mult) {
            c10::InferenceMode guard;

            auto cluster_indices_and_inverse = torch::sort(cluster_indices);
            torch::Tensor sorted_cluster_indices = std::get<0>(cluster_indices_and_inverse);
            torch::Tensor inverse_indices = std::get<1>(cluster_indices_and_inverse);

            auto unique_cluster_indices_and_counts = torch::unique_consecutive(sorted_cluster_indices, false, true);

            torch::Tensor unique_cluster_indices = std::get<0>(unique_cluster_indices_and_counts);
            torch::Tensor cluster_counts = std::get<2>(unique_cluster_indices_and_counts);

            int offset = 0;

            int batch_size = options.nerf_batch_size * batch_mult;
            for (int cluster_index = 0; cluster_index < unique_cluster_indices.size(0); cluster_index++) {
                int cluster_count = cluster_counts[cluster_index].item().toInt();
                for (int chunk_index = 0; chunk_index < cluster_count; chunk_index += batch_size) {
                    int start = offset + chunk_index;
                    torch::Tensor query_indices = inverse_indices.slice(0, start, std::min(start + batch_size,
                                                                                           offset + cluster_count));

                    at::autocast::set_enabled(true);
                    torch::Tensor input = samples.index({query_indices});
                    torch::Tensor nerf_results = nerfs[unique_cluster_indices[cluster_index].item().toInt()].forward(
                            {input, false}).toTensor();
                    at::autocast::clear_cache();
                    at::autocast::set_enabled(false);

                    result_buffer.scatter_(0, query_indices.unsqueeze(-1).repeat({1, nerf_results.size(1)}),
                                           nerf_results);
                }

                offset += cluster_count;
            }


        }

        void expand_voxels() {
            torch::Tensor split_candidates = split_tracker.index(
                    {split_tracker.index({torch::indexing::Slice(), 1}) >= 0});

            auto split_candidates_unique = torch::unique_dim(split_candidates, 0, false, false, true);
            torch::Tensor to_split = std::get<0>(split_candidates_unique);
            torch::Tensor to_split_counts = std::get<2>(split_candidates_unique).to(torch::kInt32).unsqueeze_(-1);

            // Negative to make sort stable with other factors
            to_split = torch::cat({-to_split_counts, to_split}, -1);
            to_split = to_split.index({to_split.index({torch::indexing::Slice(), 0}) < -1});
            auto to_split_sorted = torch::unique_dim(to_split, 0);
            to_split = std::get<0>(to_split_sorted);

            std::cout << "Split candidates: " << to_split.size(0) << std::endl;

            if (to_split.size(0) == 0) {
                get_more_samples();
                return;
            }

            to_split = to_split.slice(0, 0, options.split_batch_size).slice(1, 2).to(torch::kInt32);

            if (tree->capacity + to_split.size(0) > max_tree_capacity) {
                std::cout << "Full" << std::endl;
                return;
            }

            int N3 = tree->N * tree->N * tree->N;

            int64_t num_children = to_split.size(0) * N3;
            torch::Tensor cluster_indices = torch::empty({num_children, options.samples_per_corner},
                                                         torch::TensorOptions()
                                                                 .device(torch::kCUDA)
                                                                 .dtype(torch::kInt16));
            int rand_dim = 3;

            if (options.need_viewdir) {
                rand_dim += 3;
            }

            if (options.appearance_embedding != -1) {
                rand_dim += 1;
            }

            torch::Tensor rand_sample = torch::rand({num_children, options.samples_per_corner, rand_dim},
                                                    torch::TensorOptions()
                                                            .device(torch::kCUDA)
                                                            .dtype(torch::kFloat32));

            add_children_and_generate_samples(*tree, options, to_split, rand_sample, cluster_indices, visit_tracker,
                                              grid_dim, min_position, range);

            torch::Tensor results = torch::empty({num_children, options.samples_per_corner, tree->data_dim + 1},
                                                 torch::TensorOptions()
                                                         .device(torch::kCUDA)
                                                         .dtype(torch::kFloat32));

            auto buffer = results.view({-1, tree->data_dim + 1});
            query_submodules(cluster_indices.view({-1}), rand_sample.view({-1, rand_dim}), buffer, 1);

            torch::Tensor new_data_slice = tree->data.view({-1, tree->data_dim}).slice(0, tree->capacity * N3,
                                                                                       tree->capacity * N3 +
                                                                                       num_children);

            torch::mean_out(new_data_slice, results.slice(2, 0, tree->data_dim), 1);

            tree->sample_counts.slice(0, tree->capacity, tree->capacity + to_split.size(0)).fill_(
                    options.samples_per_corner);

            tree->capacity += to_split.size(0);
            std::cout << "Added: " << to_split.size(0) << ", total size: " << tree->capacity << std::endl;
            can_reuse_results = false;
        }

        void get_more_samples() {
            torch::Tensor sample_candidates = sample_tracker.index(
                    {sample_tracker.index({torch::indexing::Slice(), 1}) >= 0});

            if (sample_candidates.size(0) == 0) {
                return;
            }

            auto sample_candidates_sorted = torch::unique_dim(sample_candidates, 0);
            sample_candidates = std::get<0>(sample_candidates_sorted);

            std::cout << "Sample candidates: " << sample_candidates.size(0) << std::endl;

            torch::Tensor to_sample = sample_candidates.slice(0, 0, options.split_batch_size).slice(1, 1).to(
                    torch::kInt32);

            torch::Tensor cluster_indices = torch::empty({to_sample.size(0), options.samples_per_corner},
                                                         torch::TensorOptions()
                                                                 .device(torch::kCUDA)
                                                                 .dtype(torch::kInt16));

            torch::Tensor rand_sample = torch::rand({to_sample.size(0), options.samples_per_corner, 3},
                                                    torch::TensorOptions()
                                                            .device(torch::kCUDA)
                                                            .dtype(torch::kFloat32));

            generate_samples(*tree, options, to_sample, rand_sample, cluster_indices, grid_dim, min_position, range);

            torch::Tensor results = torch::empty({to_sample.size(0), options.samples_per_corner, tree->data_dim + 1},
                                                 torch::TensorOptions()
                                                         .device(torch::kCUDA)
                                                         .dtype(torch::kFloat32));

            auto buffer = results.view({-1, tree->data_dim + 1});
            query_submodules(cluster_indices.view({-1}), rand_sample.view({-1, 3}), buffer, 1);

            int N3 = tree->N * tree->N * tree->N;

            torch::Tensor sample_dest_indices = (to_sample.index({torch::indexing::Slice(), 0}) * N3 +
                                                 to_sample.index({torch::indexing::Slice(), 1})).to(torch::kInt64);

            torch::Tensor new_sample_counts = tree->sample_counts.view({-1}).index({sample_dest_indices}) +
                                              options.samples_per_corner;

            torch::Tensor new_data = torch::sum(results.slice(2, 0, tree->data_dim), 1);

            // new average = old average + (next data - old average) / next count
            torch::Tensor data_update = (new_data - options.samples_per_corner *
                                                    tree->data.view({-1, tree->data_dim})
                                                            .index({sample_dest_indices})) /
                                        new_sample_counts.unsqueeze(-1);

            tree->data.view({-1, tree->data_dim}).index_add_(0, sample_dest_indices, data_update);

            tree->sample_counts.view({-1}).index_add_(0, sample_dest_indices,
                                                      torch::ones({sample_dest_indices.size(0)},
                                                                  torch::TensorOptions()
                                                                          .device(torch::kCUDA)
                                                                          .dtype(torch::kInt16)) *
                                                      options.samples_per_corner);
            can_reuse_results = false;
        }

        void prune_tree() {
            std::cout << "Pruning" << std::endl;
            torch::Tensor to_delete = visit_tracker.slice(0, 0, tree->capacity) == 0;

            int num_to_delete = to_delete.sum().item().toInt();

            if (num_to_delete == 0) {
                std::cout << "Nothing can be pruned" << std::endl;
                visit_tracker.slice(0, 1, max_tree_capacity).zero_();
                return;
            }

            torch::Tensor index_shifts = torch::cumsum(to_delete, 0, torch::kInt32);

            int first_shift_index = index_shifts.argmin().item().toInt();
            adjust_parents_and_children(*tree, first_shift_index, to_delete, index_shifts);

            torch::Tensor to_delete_shifted = to_delete.slice(0, first_shift_index, tree->capacity);
            torch::Tensor copy_indices =
                    torch::arange(first_shift_index, tree->capacity)
                            .index({to_delete_shifted == false});
            for (int i = 0; i < copy_indices.size(0); i += PRUNE_CHUNK_SIZE) {
                int start = first_shift_index + i;
                int end = std::min(start + PRUNE_CHUNK_SIZE, (int) copy_indices.size(0));
                tree->data.slice(0, start, end) = tree->data.index({copy_indices.slice(0, i, i + PRUNE_CHUNK_SIZE)})
                        .clone();

                tree->child.slice(0, start, end) = tree->child.index(
                        {copy_indices.slice(0, i, i + PRUNE_CHUNK_SIZE)}).clone();

                tree->parent.slice(0, start, end) = tree->parent.index(
                        {copy_indices.slice(0, i, i + PRUNE_CHUNK_SIZE)}).clone();
            }

            tree->capacity -= num_to_delete;

            visit_tracker.slice(0, 1, max_tree_capacity).zero_();
            std::cout << "Pruning finished - reclaimed: " << num_to_delete << std::endl;
        }

        void resize(const int width, const int height) {
            if (camera.width == width && camera.height == height) return;
            start();

            float width_ratio = (float) width / camera.width;
            float height_ratio = (float) height / camera.height;

            // There seems to be an initial resize triggered from 256x256
            // to the actual specified camera dimensions for some reason
            if (!initial_resize) {
                camera.fx *= width_ratio;
                camera.default_fx *= width_ratio;

                camera.fy *= height_ratio;
                camera.default_fy *= height_ratio;
                camera.cy *= height_ratio;

                if (camera.default_cx != -1) {
                    camera.cx *= width_ratio;
                }

                if (camera.default_cy != -1) {
                    camera.cy *= height_ratio;
                }
            } else {
                initial_resize = false;
            }

            if (camera.default_cx == -1) {
                camera.cx = width / 2;
            }

            if (camera.default_cy == -1) {
                camera.cy = height / 2;
            }

            // save new size
            camera.width = width;
            camera.height = height;

            init_split_tracker();
            init_sample_tensor();

            // unregister resource
            for (int index = 0; index < cgr.size(); index++) {
                if (cgr[index] != nullptr)
                    cuda_assert(cudaGraphicsUnregisterResource(cgr[index]), __FILE__, __LINE__);
            }

            // resize color buffer
            for (int index = 0; index < 2; index++) {
                // resize rbo
                glNamedRenderbufferStorage(rb[index], GL_RGBA8, width, height);
                glNamedRenderbufferStorage(depth_rb[index], GL_R32F, width, height);
                glNamedRenderbufferStorage(depth_buf_rb[index], GL_DEPTH_COMPONENT32F, width, height);
                const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
                glNamedFramebufferDrawBuffers(fb[index], 2, attach_buffers);

                // register rbo
                cuda_assert(cudaGraphicsGLRegisterImage(&cgr[index * 2], rb[index], GL_RENDERBUFFER,
                                                        cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                                        cudaGraphicsRegisterFlagsWriteDiscard), __FILE__, __LINE__);
                cuda_assert(cudaGraphicsGLRegisterImage(&cgr[index * 2 + 1], depth_rb[index],
                                                        GL_RENDERBUFFER,
                                                        cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                                        cudaGraphicsRegisterFlagsWriteDiscard), __FILE__, __LINE__);
            }

            cuda_assert(cudaGraphicsMapResources(cgr.size(), cgr.data(), 0), __FILE__, __LINE__);

            for (int index = 0; index < cgr.size(); index++) {
                cuda_assert(cudaGraphicsSubResourceGetMappedArray(&ca[index], cgr[index], 0, 0), __FILE__, __LINE__);
            }

            cuda_assert(cudaGraphicsUnmapResources(cgr.size(), cgr.data(), 0), __FILE__, __LINE__);
        }

        void init_split_tracker() {
            torch::TensorOptions tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

            split_tracker = torch::ones({camera.height * camera.width, 3}, tensorOptions) * -1;
            sample_tracker = torch::ones({camera.height * camera.width, 3}, tensorOptions) * -1;
            num_samples = torch::empty({camera.height * camera.width},
                                       torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));
            cluster_indices = torch::empty({camera.height * camera.width, options.max_guided_samples},
                                           torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));
        }

        void init_sample_tensor() {
            if (nerfs.empty() || this->tree == nullptr) {
                return;
            }

            torch::TensorOptions tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

            int samples_dim = 4; // Add one for z_vals

            if (options.need_viewdir) {
                samples_dim += 3;
            }

            if (options.appearance_embedding != -1) {
                samples_dim += 1;
            }

            guided_samples = torch::ones({camera.height * camera.width, options.max_guided_samples, samples_dim},
                                         tensorOptions) * -1;

            nerf_result_buffer = torch::empty(
                    {camera.height * camera.width * options.max_guided_samples, tree->data_dim + 1},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            can_reuse_results = false;
        }

        void set(N3Tree &tree, long max_tree_capacity) {
            start();
            tree.move_to_device(max_tree_capacity, true, true);

            this->tree = &tree;
            this->max_tree_capacity = max_tree_capacity;

            visit_tracker = torch::zeros({max_tree_capacity},
                                         torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            visit_tracker[0] = 1;

            wire_.vert.clear();
            wire_.faces.clear();
            options.basis_minmax[0] = 0;
            options.basis_minmax[1] = std::max(tree.data_format.basis_dim - 1, 0);
            last_wire_depth_ = -1;

            init_sample_tensor();
        }

        void load_model(const std::filesystem::path &container_path) {
            c10::InferenceMode guard;

            device = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
            std::cout << "Loading model from: " << container_path << std::endl;

            torch::jit::script::Module container = torch::jit::load(container_path);
            grid_dim = container.attr("grid_dim").toTensor().to(device);
            min_position = container.attr("min_position").toTensor().to(device);
            range = container.attr("max_position").toTensor().to(device) - min_position;

            for (int i = 0; i < container.attr("centroids").toTensor().size(0); i++) {
                torch::jit::script::Module nerf = container.attr("sub_module_" + std::to_string(i)).toModule();
                nerf.to(device);
                nerf.eval();
                nerfs.push_back(torch::jit::freeze(nerf));
            }

            options.need_viewdir = container.attr("need_viewdir").toBool();
            if (options.appearance_embedding == -1 and container.attr("need_appearance_embedding").toBool()) {
                options.appearance_embedding = 0;
            }

            init_sample_tensor();
            std::cout << "Model loaded" << std::endl;
        }

        void maybe_gen_wire(int depth) {
            if (last_wire_depth_ != depth) {
                wire_.vert = tree->gen_wireframe(depth);
                wire_.update();
                last_wire_depth_ = depth;
            }
        }

        N3Tree *tree = nullptr;

    private:
        Camera &camera;
        RenderOptions &options;
        int buf_index;

        // GL buffers
        std::array<GLuint, 2> fb, rb, depth_rb, depth_buf_rb;

        // CUDA resources
        std::array<cudaGraphicsResource_t, 4> cgr = {{0}};
        std::array<cudaArray_t, 4> ca;

        Mesh wire_;
        // The depth level of the octree wireframe; -1 = not yet generated
        int last_wire_depth_ = -1;

        cudaStream_t stream;
        bool started_ = false;

        std::vector<torch::jit::script::Module> nerfs;
        torch::Tensor grid_dim;
        torch::Tensor min_position;
        torch::Tensor range;

        torch::Device device = torch::Device(torch::kCPU);
        torch::Tensor split_tracker;
        torch::Tensor sample_tracker;

        torch::Tensor num_samples;
        torch::Tensor guided_samples;
        torch::Tensor cluster_indices;

        torch::Tensor nerf_result_buffer;
        torch::Tensor z_vals;
        torch::Tensor offsets;

        bool can_reuse_results = false;


        torch::Tensor visit_tracker;

        bool initial_resize = true;
        bool prune_happened = false;
        size_t max_tree_capacity;
    };

    VolumeRenderer::VolumeRenderer()
            : impl_(std::make_unique<Impl>(camera, options)) {}

    VolumeRenderer::~VolumeRenderer() {}

    void VolumeRenderer::render() { impl_->render(); }

    void VolumeRenderer::set(N3Tree &tree, long max_tree_capacity) {
        impl_->set(tree, max_tree_capacity);
    }

    void VolumeRenderer::clear() { impl_->tree = nullptr; }

    void VolumeRenderer::load_model(const std::filesystem::path &modelPath) {
        impl_->load_model(modelPath);
    }

    void VolumeRenderer::resize(int width, int height) {
        impl_->resize(width, height);
    }

    const char *VolumeRenderer::get_backend() { return "CUDA"; }

}  // namespace viewer
