#include "../include/mesh.hpp"

#include <GL/glew.h>

#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>

#include "../include/shader.hpp"

const int VERT_SZ = 9;
unsigned int program = -1;
unsigned int u_K, u_MV, u_M, u_cam_pos, u_unlit;

const char* VERT_SHADER_SRC =
        R"glsl(
uniform mat4x4 K;
uniform mat4x4 MV;
uniform mat4x4 M;

in vec3 aPos;
in vec3 aColor;
in vec3 aNormal;

out lowp vec3 VertColor;
out highp vec4 FragPos;
out highp vec3 Normal;

void main()
{
    FragPos = MV * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    gl_Position = K * FragPos;
    VertColor = aColor;
    Normal = normalize(mat3x3(M) * aNormal);
}
        )glsl";

const char* FRAG_SHADER_SRC =
        R"glsl(
precision highp float;
in lowp vec3 VertColor;
in vec4 FragPos;
in vec3 Normal;

uniform bool unlit;
uniform vec3 camPos;

layout(location = 0) out lowp vec4 FragColor;
layout(location = 1) out float Depth;

void main()
{
    if (unlit) {
        FragColor = vec4(VertColor, 1);
    } else {
        // FIXME make these uniforms, whatever for now
        float ambient = 0.3;
        float specularStrength = 0.6;
        float diffuseStrength = 0.7;
        float diffuse2Strength = 0.2;
        vec3 lightDir = normalize(vec3(0.5, 0.2, 1));
        vec3 lightDir2 = normalize(vec3(-0.5, -1.0, -0.5));

        float diffuse = diffuseStrength * max(dot(lightDir, Normal), 0.0);
        float diffuse2 = diffuse2Strength * max(dot(lightDir2, Normal), 0.0);

        vec3 viewDir = normalize(camPos - vec3(FragPos));
        vec3 reflectDir = reflect(-lightDir, Normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float specular = specularStrength * spec;

        FragColor = (ambient + diffuse + diffuse2 + specular) * vec4(VertColor, 1);
    }

    Depth = length(FragPos.xyz);
}
        )glsl";

GLenum get_gl_ele_type(int face_size) {
    switch (face_size) {
        case 1:
            return GL_POINTS;
        case 2:
            return GL_LINES;
        case 3:
            return GL_TRIANGLES;
        default:
            throw std::invalid_argument("Unsupported mesh face size");
    }
}

namespace viewer {

Mesh::Mesh(int n_verts, int n_faces, int face_size, bool unlit)
    : vert(n_verts * 9),
      faces(n_faces * face_size),
      rotation(0),
      translation(0),
      face_size(face_size),
      unlit(unlit) {
    if (program == -1) {
        program = create_shader_program(VERT_SHADER_SRC, FRAG_SHADER_SRC);
        u_MV = glGetUniformLocation(program, "MV");
        u_M = glGetUniformLocation(program, "M");
        u_K = glGetUniformLocation(program, "K");
        u_cam_pos = glGetUniformLocation(program, "camPos");
        u_unlit = glGetUniformLocation(program, "unlit");
    }
}

void Mesh::update() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(vert[0]), vert.data(),
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERT_SZ * sizeof(float),
                          (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]),
                 faces.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void Mesh::draw(const glm::mat4x4& V, glm::mat4x4 K, bool y_up) const {
    if (!visible) return;
    float norm = glm::length(rotation);
    if (norm < 1e-3) {
        transform_ = glm::mat4(1.0);
    } else {
        glm::quat rot = glm::angleAxis(norm, rotation / norm);
        transform_ = glm::mat4_cast(rot);
    }
    transform_ *= scale;
    glm::vec3 cam_pos = -glm::transpose(glm::mat3x3(V)) * glm::vec3(V[3]);
    if (!y_up) {
        K[1][1] *= -1.0;
    }

    transform_[3] = glm::vec4(translation, 1);
    glm::mat4x4 MV = V * transform_;
    glUniformMatrix4fv(u_MV, 1, GL_FALSE, glm::value_ptr(MV));
    glUniformMatrix4fv(u_M, 1, GL_FALSE, glm::value_ptr(transform_));
    glUniformMatrix4fv(u_K, 1, GL_FALSE, glm::value_ptr(K));
    glUniform3fv(u_cam_pos, 1, glm::value_ptr(cam_pos));
    glUniform1i(u_unlit, unlit);
    glBindVertexArray(vao_);
    if (faces.empty()) {
        glDrawArrays(get_gl_ele_type(face_size), 0, vert.size() / VERT_SZ);
    } else {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        glDrawElements(get_gl_ele_type(face_size), faces.size(),
                       GL_UNSIGNED_INT, (void*)0);
    }
    glBindVertexArray(0);
}

}  // namespace viewer