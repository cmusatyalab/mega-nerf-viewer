#pragma once

#include <filesystem>
#include "../camera.hpp"
#include "../render_options.hpp"

namespace viewer {
// Volume renderer using CUDA or compute shader
struct VolumeRenderer {
    explicit VolumeRenderer();
    ~VolumeRenderer();

    // Render the currently set tree
    void render();

    // Set volumetric data to render
    void set(N3Tree& tree, long max_tree_capacity);

    // Set volumetric data to render
    void load_model(const std::filesystem::path& modelPath);

    // Clear the volumetric data
    void clear();

    // Resize the buffer
    void resize(int width, int height);

    // Get name identifying the renderer backend used e.g. CUDA
    const char* get_backend();

    // Camera instance
    Camera camera;

    // Rendering options
    RenderOptions options;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer
