#pragma once

#include <string>
#include <vector>

#include "glm/mat4x4.hpp"

namespace viewer {

struct Mesh {
    explicit Mesh(int n_verts = 0,
                  int n_faces = 0,
                  int face_size = 3,
                  bool unshaded = false);

    // Upload to GPU
    void update();

    // Draw the mesh
    void draw(const glm::mat4x4& V, glm::mat4x4 K, bool y_up = true) const;

    // Vertex positions
    std::vector<float> vert;
    // Triangle indices
    std::vector<unsigned int> faces;

    // Model transform, rotation is axis-angle
    glm::vec3 rotation, translation;
    float scale = 1.f;

    // Computed transform
    mutable glm::mat4 transform_;

    int face_size;
    bool visible = true;
    bool unlit = false;

private:
    unsigned int vao_, vbo_, ebo_;
};

}  // namespace viewer
