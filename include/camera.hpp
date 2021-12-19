#pragma once

#include <memory>

#include "glm/mat4x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

namespace viewer {

struct Camera {
    Camera(int width = 256,
           int height = 256,
           float fx = 1111.f,
           float fy = -1.f,
           float cx = -1.f,
           float cy = -1.f);
    ~Camera();

    /** Drag helpers **/
    void begin_drag(float x, float y, bool is_pan, bool about_origin);
    void drag_update(float x, float y);
    void end_drag();
    bool is_dragging() const;
    /** Move center by +=xyz, correctly handling drag **/
    void move(const glm::vec3& xyz);

    bool has_changed();

    /** Camera params **/
    // Camera pose model, you can modify these
    glm::vec3 v_back, v_world_up, center;

    // Origin for about-origin rotation
    glm::vec3 origin;

    // Vectors below are automatically updated
    glm::vec3 v_up, v_right;

    // 4x3 C2W transform used for volume rendering, automatically updated
    glm::mat4x3 transform;

    // 4x4 projection matrix for triangle rendering
    glm::mat4x4 K;

    // 4x4 W2C transform
    glm::mat4x4 w2c;

    // Image size
    int width, height;

    // Focal length
    float fx, fy;

    // Camera centers
    float cx, cy;

    // "Default" focal length
    float default_fx, default_fy;

    // "Default" camera centers
    float default_cx, default_cy;

    // GUI movement speed
    float movement_speed = 1.f;

    // CUDA memory used in kernel
    struct {
        float* transform = nullptr;
    } device;

    // Update the transform after modifying v_right/v_forward/center
    // (internal)
    void _update(bool transform_from_vecs = true, bool copy_cuda = true);

private:
    // For dragging
    struct DragState;
    std::unique_ptr<DragState> drag_state_;
    bool has_changed_ = true;
    bool transform_changed_ = false;
    float last_fx;
    float last_fy;
    int last_width;
    int last_height;
};

}  // namespace viewer
