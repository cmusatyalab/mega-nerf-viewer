# On Ubuntu 18.04, get the latest CMake from https://apt.kitware.com/.
set(CMAKE_VERBOSE_MAKEFILE on)
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
MESSAGE(STATUS "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")

project(mega-nerf-viewer LANGUAGES CUDA CXX)

if (POLICY CMP0091)
    # https://stackoverflow.com/a/56490614
    cmake_policy(SET CMP0091 NEW)
endif ()


find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if (GIT_SUBMODULE)
        message(STATUS "Running git submodule update --init --recursive")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
        message(STATUS "Submodule update done")
    endif ()
endif ()

set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

include_directories("${VENDOR_DIR}/cnpy")
file(GLOB CNPY_SOURCES ${VENDOR_DIR}/cnpy/*.cpp)

include_directories("${VENDOR_DIR}/cxxopts")

include_directories("${VENDOR_DIR}/glm")

include_directories("${VENDOR_DIR}/imgui")
file(GLOB IMGUI_SOURCES ${VENDOR_DIR}/imgui/*.cpp)

include_directories("${VENDOR_DIR}/ImGuizmo")
file(GLOB IMGUIZMO_SOURCES ${VENDOR_DIR}/ImGuizmo/*.cpp)

include_directories("${VENDOR_DIR}/imgui-filebrowser")

include_directories("${VENDOR_DIR}/zstream-cpp")

include_directories("${CUDA_INCLUDE_DIRS}")

file(GLOB_RECURSE VIEWER_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
file(GLOB_RECURSE VIEWER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

add_executable(mega-nerf-viewer ${VIEWER_HEADERS} ${VIEWER_SOURCES} ${CNPY_SOURCES} ${IMGUI_SOURCES} ${IMGUIZMO_SOURCES} main.cpp)
target_link_libraries(mega-nerf-viewer glfw OpenGL::GL ${PNG_LIBRARIES} ${GLEW_LIBRARIES} ${TORCH_LIBRARIES} ${ZLIB_LIBRARIES})

# On Windows, when BUILD_SHARED_LIBS, copy .dll to the executable directory
if (WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET mega-nerf-viewer
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:mega-nerf-viewer>)
endif ()
