#include "vec.h"

#include "common.h"
#include "rand.h"
#include <cmath>
#include <cstdio>
#include <math.h>

__host__ __device__ float3 vec_new(const float x, const float y, const float z) {
    return (float3) {.x = x, .y = y, .z = z};
}

__host__ __device__ float3 vec_add(const float3 a, const float3 b) {
    return (float3) { .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
}

__host__ __device__ float3 vec_sub(const float3 a, const float3 b) {
    return (float3) { .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
}

__host__ __device__ float3 vec_scale(const float3 a, const float k) { 
    return (float3) { .x = k * a.x, .y = k * a.y, .z = k * a.z };
}

__host__ __device__ float3 vec_negate(const float3 vec) {
    return (float3) { .x = -vec.x, .y = -vec.y, .z = -vec.z };
}

__host__ __device__ float3 vec_mult(const float3 a, const float3 b) {
    return (float3) { .x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z };
}

__host__ __device__ float3 vec_cross(const float3 a, const float3 b) {
    return (float3) {
        .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - b.y * b.x
    };
}

__host__ __device__ float vec_dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float vec_length(const float3 a) {
    return sqrt(vec_dot(a, a));
}

__host__ __device__ float3 vec_norm(const float3 a) {
    return vec_scale(a, 1.0f / vec_length(a));
}

__device__ float3 random_unit_vector() {
    while (1) {
        float3 vec = {RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1)};
        float length_sq = vec_dot(vec, vec);
        if (1e-6f < length_sq && length_sq <= 1.0f) {
            return vec_scale(vec, 1.0f / sqrtf(length_sq));
        }
    }
}

__host__ __device__ float3 vector_reflect(const float3 incoming, const float3 normal) {
    return vec_sub(incoming, vec_scale(normal, 2.0f * vec_dot(incoming, normal)));
}

__host__ __device__ float3 vector_refract(float3 incoming, const float3 normal, float refract_coefficient) {
    float cos_theta = -vec_dot(incoming, normal);
    if (cos_theta > 1.0f) { cos_theta = 1.0f; }

    const float3 out_perp = vec_scale(vec_add(incoming, vec_scale(normal, cos_theta)), refract_coefficient);
    const float3 out_para = vec_scale(normal, -sqrt(fabs(1.0f - vec_dot(out_perp, out_perp))));

    return vec_add(out_perp, out_para);
}

__host__ __device__ float4 quaternion_from_euler(const float rx, const float ry, const float rz) {
    const float SX = sin(M_PI * rx / 360.0f), SY = sin(M_PI * ry / 360.0f), SZ = sin(M_PI * rz / 360.0f),
                 CX = cos(M_PI * rx / 360.0f), CY = cos(M_PI * ry / 360.0f), CZ = cos(M_PI * rz / 360.0f);
    return (float4) {
        .x = (SX * CY * CZ) - (CX * SY * SZ),
        .y = (CX * SY * CZ) + (SX * CY * SZ),
        .z = (CX * CY * SZ) - (SX * SY * CZ),
        .w = (CX * CY * CZ) + (SX * SY * SZ),
    };
}

__host__ __device__ float3 vec_rotate(const float3 vec, const float4 quat) {
    float3 u = vec_new(quat.x, quat.y, quat.z);
    float s = quat.w;

    return vec_add(vec_add(
                vec_scale(u, 2.0f * vec_dot(u, vec)),
                vec_scale(vec, s * s - vec_dot(u, u))),
                vec_scale(vec_cross(u, vec), 2.0f * s));
}
