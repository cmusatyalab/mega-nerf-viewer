#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cxxopts.hpp>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "include/imwrite.h"
#include "include/n3tree/n3tree.hpp"
#include "include/opts.hpp"
#include "include/renderer/renderer.hpp"

// clang-format off
// @formatter:off
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
// clang-format on
// @formatter:on

#include "ImGuizmo.h"
#include "imfilebrowser.h"

namespace viewer {

namespace {

#define GET_RENDERER(window) \
    (*((VolumeRenderer *)glfwGetWindowUserPointer(window)))

void glfw_update_title(GLFWwindow *window) {
    // static fps counters
    // Source: http://antongerdelan.net/opengl/glcontext2.html
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        char tmp[128];
        sprintf(tmp, "mega-nerf viewer - FPS: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        frame_count = 0;
    }

    frame_count++;
}

int gizmo_mesh_op = ImGuizmo::TRANSLATE;
int gizmo_mesh_space = ImGuizmo::LOCAL;

void draw_imgui(VolumeRenderer &rend,
                N3Tree &tree,
                long max_tree_capacity,
                std::string &model_path_str) {
    auto &cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // BEGIN gizmo handling
    // clang-format off
        static glm::mat4 camera_persp_prj(1.f, 0.f, 0.f, 0.f,
                                          0.f, 1.f, 0.f, 0.f,
                                          0.f, 0.f, -1.f, -1.f,
                                          0.f, 0.f, -0.001f, 0.f);
    // clang-format on
    ImGuiIO &io = ImGui::GetIO();

    camera_persp_prj[0][0] = cam.fx / cam.width * 2.0;
    camera_persp_prj[1][1] = cam.fy / cam.height * 2.0;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetGizmoSizeClipSpace(0.05f);

    ImGuizmo::BeginFrame();

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    glm::mat4 w2c = glm::affineInverse(glm::mat4(cam.transform));
    // END gizmo handling

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(340.f, 480.f), ImGuiCond_Once);

    static char title[128] = {0};
    if (title[0] == 0) {
        sprintf(title, "viewer backend: %s", rend.get_backend());
    }

    // Begin window
    ImGui::Begin(title);

    static ImGui::FileBrowser open_obj_mesh_dialog(
            ImGuiFileBrowserFlags_MultipleSelection);

    static ImGui::FileBrowser open_tree_dialog,
            save_screenshot_dialog(ImGuiFileBrowserFlags_EnterNewFilename);

    if (open_tree_dialog.GetTitle().empty()) {
        open_tree_dialog.SetTypeFilters({".npz"});
        open_tree_dialog.SetTitle("Load N3Tree npz from svox");
    }

    if (save_screenshot_dialog.GetTitle().empty()) {
        save_screenshot_dialog.SetTypeFilters({".png"});
        save_screenshot_dialog.SetTitle("Save screenshot (png)");
    }

    if (ImGui::Button("Open Tree")) {
        open_tree_dialog.Open();
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Screenshot")) {
        save_screenshot_dialog.Open();
    }

    open_tree_dialog.Display();
    if (open_tree_dialog.HasSelected()) {
        // Load octree
        std::string path = open_tree_dialog.GetSelected().string();
        printf("Load N3Tree npz: %s\n", path.c_str());
        tree.open(path);
        rend.set(tree, max_tree_capacity);
        open_tree_dialog.ClearSelected();
    }

    save_screenshot_dialog.Display();
    if (save_screenshot_dialog.HasSelected()) {
        // Save screenshot
        std::string path = save_screenshot_dialog.GetSelected().string();
        save_screenshot_dialog.ClearSelected();
        int width = rend.camera.width, height = rend.camera.height;
        std::vector<unsigned char> windowPixels(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     &windowPixels[0]);

        std::vector<unsigned char> flippedPixels(4 * width * height);
        for (int row = 0; row < height; ++row)
            memcpy(&flippedPixels[row * width * 4],
                   &windowPixels[(height - row - 1) * width * 4], 4 * width);

        if (path.size() < 4 ||
            path.compare(path.size() - 4, 4, ".png", 0, 4) != 0) {
            path.append(".png");
        }
        if (write_png_file(path, flippedPixels.data(), width, height, width)) {
            printf("Wrote %s", path.c_str());
        } else {
            printf("Failed to save screenshot\n");
        }
    }

    ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Camera")) {
        // Update vectors indirectly since we need to normalize on change
        // (press update button) and it would be too confusing to keep
        // normalizing
        static glm::vec3 world_up_tmp = rend.camera.v_world_up;
        static glm::vec3 world_down_prev = rend.camera.v_world_up;
        static glm::vec3 back_tmp = rend.camera.v_back;
        static glm::vec3 forward_prev = rend.camera.v_back;
        if (cam.v_world_up != world_down_prev)
            world_up_tmp = world_down_prev = cam.v_world_up;
        if (cam.v_back != forward_prev) back_tmp = forward_prev = cam.v_back;

        ImGui::InputFloat3("center", glm::value_ptr(cam.center));
        ImGui::InputFloat3("origin", glm::value_ptr(cam.origin));
        static bool lock_fx_fy = false;
        ImGui::Checkbox("fx=fy", &lock_fx_fy);
        if (lock_fx_fy) {
            if (ImGui::SliderFloat("focal", &cam.fx, 300.f, 7000.f)) {
                cam.fy = cam.fx;
            }
        } else {
            ImGui::SliderFloat("fx", &cam.fx, 300.f, 7000.f);
            ImGui::SliderFloat("fy", &cam.fy, 300.f, 7000.f);
        }
        if (ImGui::TreeNode("Directions")) {
            ImGui::InputFloat3("world_up", glm::value_ptr(world_up_tmp));
            ImGui::InputFloat3("back", glm::value_ptr(back_tmp));
            if (ImGui::Button("normalize & update dirs")) {
                cam.v_world_up = glm::normalize(world_up_tmp);
                cam.v_back = glm::normalize(back_tmp);
            }
            ImGui::TreePop();
        }
    }  // End camera node

    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Render")) {
        static float inv_step_size = 1.0f / rend.options.step_size;
        if (ImGui::SliderFloat("1/eps", &inv_step_size, 128.f, 20000.f)) {
            rend.options.step_size = 1.f / inv_step_size;
        }
        ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f,
                           100.0f);
        ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f,
                           0.4f);
        ImGui::SliderFloat("bg_brightness", &rend.options.background_brightness,
                           0.f, 1.0f);

    }  // End render node
    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Visualization")) {
        ImGui::PushItemWidth(230);
        ImGui::SliderFloat3("bb_min", rend.options.render_bbox, 0.0, 1.0);
        ImGui::SliderFloat3("bb_max", rend.options.render_bbox + 3, 0.0, 1.0);
        ImGui::SliderInt2("decomp", rend.options.basis_minmax, 0,
                          std::max(tree.data_format.basis_dim - 1, 0));
        ImGui::SliderFloat3("viewdir shift", rend.options.rot_dirs, -M_PI / 4,
                            M_PI / 4);
        ImGui::PopItemWidth();
        if (ImGui::Button("Reset Viewdir Shift")) {
            for (int i = 0; i < 3; ++i) rend.options.rot_dirs[i] = 0.f;
        }

        ImGui::SameLine();

        ImGui::Checkbox("Show Grid", &rend.options.show_grid);
        ImGui::SameLine();
        ImGui::Checkbox("Render Depth", &rend.options.render_depth);
        ImGui::SameLine();
        ImGui::Checkbox("Dynamic Splitting", &rend.options.use_splitting);

        if (rend.options.show_grid) {
            ImGui::SliderInt("grid max depth", &rend.options.grid_max_depth, 0, 4);
        }
    }

    ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Computation")) {
        static int max_tree_depth = 20;
        ImGui::SliderInt("Tree Max Depth", &rend.options.max_depth, 1, 31);
        ImGui::SliderInt("Voxel Max Samples", &rend.options.max_sample_count, 1,
                         2048);

        ImGui::SliderInt("Samples per corner", &rend.options.samples_per_corner,
                         1, 1024);

        ImGui::SliderInt("Split batch size", &rend.options.split_batch_size, 1,
                         128 * 1024);

        ImGui::SliderInt("Model batch size", &rend.options.nerf_batch_size, 1,
                         16 * 1024);
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void glfw_error_callback(int error, const char *description) {
    fputs(description, stderr);
}

void glfw_key_callback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        auto &rend = GET_RENDERER(window);
        auto &cam = rend.camera;
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_W:
            case GLFW_KEY_S:
            case GLFW_KEY_A:
            case GLFW_KEY_D:
            case GLFW_KEY_E:
            case GLFW_KEY_Q: {
                // Camera movement
                float speed = 0.002f;
                if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                    speed = -speed;
                const auto &vec =
                        (key == GLFW_KEY_A || key == GLFW_KEY_D)   ? cam.v_right
                        : (key == GLFW_KEY_W || key == GLFW_KEY_S) ? -cam.v_back
                                                                   : -cam.v_up;
                cam.move(vec * speed);
            } break;

            case GLFW_KEY_C: {
                // Print C2W matrix
                puts("C2W:\n");
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        if (j) puts(" ");
                        printf("%.10f", cam.transform[j][i]);
                    }
                    puts("\n");
                }
                fflush(stdout);
            } break;

            case GLFW_KEY_Z: {
                // Cycle gizmo op
                if (gizmo_mesh_op == ImGuizmo::TRANSLATE)
                    gizmo_mesh_op = ImGuizmo::ROTATE;
                else if (gizmo_mesh_op == ImGuizmo::ROTATE)
                    gizmo_mesh_op = ImGuizmo::SCALE_Z;
                else
                    gizmo_mesh_op = ImGuizmo::TRANSLATE;
            } break;

            case GLFW_KEY_X: {
                // Cycle gizmo space
                if (gizmo_mesh_space == ImGuizmo::LOCAL)
                    gizmo_mesh_space = ImGuizmo::WORLD;
                else
                    gizmo_mesh_space = ImGuizmo::LOCAL;
            } break;

            case GLFW_KEY_M:
                rend.options.use_splitting = !rend.options.use_splitting;
                break;

            case GLFW_KEY_R:
                rend.options.use_guided_sampling = !rend.options.use_guided_sampling;
                break;

            case GLFW_KEY_G:
                rend.options.grid_max_depth += 1;
                break;

            case GLFW_KEY_F:
                rend.options.grid_max_depth -= 1;
                break;

            case GLFW_KEY_MINUS:
                cam.fx *= 0.99f;
                cam.fy *= 0.99f;
                break;

            case GLFW_KEY_EQUAL:
                cam.fx *= 1.01f;
                cam.fy *= 1.01f;
                break;

            case GLFW_KEY_0:
                cam.fx = cam.default_fx;
                cam.fy = cam.default_fy;
                break;

            case GLFW_KEY_1:
                cam.v_world_up = glm::vec3(0.f, 0.f, 1.f);
                break;

            case GLFW_KEY_2:
                cam.v_world_up = glm::vec3(0.f, 0.f, -1.f);
                break;

            case GLFW_KEY_3:
                cam.v_world_up = glm::vec3(0.f, 1.f, 0.f);
                break;

            case GLFW_KEY_4:
                cam.v_world_up = glm::vec3(0.f, -1.f, 0.f);
                break;

            case GLFW_KEY_5:
                cam.v_world_up = glm::vec3(1.f, 0.f, 0.f);
                break;

            case GLFW_KEY_6:
                cam.v_world_up = glm::vec3(-1.f, 0.f, 0.f);
                break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow *window,
                                int button,
                                int action,
                                int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto &rend = GET_RENDERER(window);
    auto &cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        const bool SHIFT = mods & GLFW_MOD_SHIFT;
        cam.begin_drag(x, y, SHIFT || button == GLFW_MOUSE_BUTTON_MIDDLE,
                       button == GLFW_MOUSE_BUTTON_RIGHT ||
                               (button == GLFW_MOUSE_BUTTON_MIDDLE && SHIFT));
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void glfw_cursor_pos_callback(GLFWwindow *window, double x, double y) {
    GET_RENDERER(window).camera.drag_update(x, y);
}

void glfw_scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto &cam = GET_RENDERER(window).camera;
    // Focal length adjusting was very annoying so changed it to movement in z
    // cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

GLFWwindow *glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) std::exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window =
            glfwCreateWindow(width, height, "nerf viewer", NULL, NULL);

    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (window == nullptr) {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char *glsl_version = NULL;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace viewer

cxxopts::ParseResult parse_options(int argc, char *argv[]) {
    cxxopts::Options cxxoptions("nerf-viewer", "OpenGL NeRF Viewer");
    viewer::internal::add_common_opts(cxxoptions);

    // clang-format off
    // @formatter:off
    cxxoptions.add_options()
    ("w,width", "image width", cxxopts::value<int>()->default_value("800"))
    ("h,height", "image height", cxxopts::value<int>()->default_value("800"))
    ("fx", "focal length in x direction", cxxopts::value<float>()->default_value("1111"))
    ("fy", "focal length in y direction; -1 = use fx", cxxopts::value<float>()->default_value("-1.0"))
    ("cx", "camera center in x direction; -1 = use width / 2", cxxopts::value<float>()->default_value("-1"))
    ("cy", "camera center in y direction; -1 = use height / 2", cxxopts::value<float>()->default_value("-1.0"))
    ("center", "camera center position (world)",
    cxxopts::value<std::vector<float>>()->default_value("-3.5,0,3.5"))
    ("back", "camera's back direction unit vector (world) for orientation",
    cxxopts::value<std::vector<float>>()->default_value("-0.7071068,0,0.7071068"))
    ("origin", "origin for right click rotation controls",
    cxxopts::value<std::vector<float>>()->default_value("0,0,0"))
    ("world_up", "world up direction for rotating controls e.g. 0,0,1=blender;",
    cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
    ("grid", "show grid with given max resolution (4 is reasonable)", cxxopts::value<int>());
    // @formatter:on
    // clang-format on

    cxxoptions.positional_help("npz_file");

    cxxopts::ParseResult args =
            viewer::internal::parse_options(cxxoptions, argc, argv);

    return args;
}

int main(int argc, char *argv[]) {
    using namespace viewer;

    cxxopts::ParseResult args = parse_options(argc, argv);

    torch::manual_seed(42);

    N3Tree tree;
    bool init_loaded = false;
    if (args.count("file")) {
        init_loaded = true;
        tree.open(args["file"].as<std::string>());
        if (args["bounds_only"].as<bool>()) {
            tree.capacity = 1;
            tree.data = tree.data.slice(0, 0, 1);
            tree.child = tree.child.slice(0, 0, 1);
            tree.parent = tree.parent.slice(0, 0, 1);

            for (int i = 0; i < tree.N * tree.N * tree.N; i++) {
                tree.child[0][i] = 0;
            }
        }
    }
    int width = args["width"].as<int>();
    int height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    float fy = args["fy"].as<float>();
    float cx = args["cx"].as<float>();
    float cy = args["cy"].as<float>();

    GLFWwindow *window = glfw_init(width, height);

    {
        VolumeRenderer rend;
        if (fx > 0.f) {
            rend.camera.fx = fx;
        }

        rend.options = viewer::internal::render_options_from_args(args);
        auto cen = args["center"].as<std::vector<float>>();
        rend.camera.center = glm::vec3(cen[0], cen[1], cen[2]);
        auto origin = args["origin"].as<std::vector<float>>();
        rend.camera.origin = glm::vec3(origin[0], origin[1], origin[2]);
        auto world_up = args["world_up"].as<std::vector<float>>();
        rend.camera.v_world_up =
                glm::vec3(world_up[0], world_up[1], world_up[2]);
        auto back = args["back"].as<std::vector<float>>();
        rend.camera.v_back = glm::vec3(back[0], back[1], back[2]);

        if (fy <= 0.f) {
            rend.camera.fy = rend.camera.fx;
        } else {
            rend.camera.fy = fy;
        }

        if (cx <= 0.f) {
            rend.camera.cx = rend.camera.width / 2;
        } else {
            rend.camera.cx = cx;
        }

        if (cy <= 0.f) {
            rend.camera.cy = rend.camera.height / 2;
        } else {
            rend.camera.cy = cy;
        }

        glfwGetFramebufferSize(window, &width, &height);

        std::string model_path_str = args["model_path"].as<std::string>();
        if (!model_path_str.empty()) {
            rend.load_model(std::filesystem::path(model_path_str));
        }

        long max_tree_capacity = args["max_tree_capacity"].as<size_t>();
        rend.set(tree, max_tree_capacity);
        rend.resize(width, height);

        // Set user pointer and callbacks
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glPointSize(4.f);
            glfw_update_title(window);

            rend.render();

            draw_imgui(rend, tree, max_tree_capacity, model_path_str);

            glfwSwapBuffers(window);
            glFinish();
            glfwPollEvents();
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
