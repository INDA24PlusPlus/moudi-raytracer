#pragma once

#define Vec3(x, y, z) ((float3) {x, y, z})
#define ExpandVec3(vec) (vec).x, (vec).y, (vec).z
#define ExpandVec3F(vec, f) f((vec).x), f((vec).y), f((vec).z)

#define DEFAULT_VEC Vec3(0.0, 0.0, 1.0)
#define DEFAULT_VEC_UP Vec(0.0, 0.0, 1.0)

#define VEC_ZERO Vec3(0.0, 0.0, 0.0)
#define VEC_ONE Vec3(1.0, 1.0, 1.0)

__host__ __device__ float3 vec_new(const float x, const float y, const float z);
__host__ __device__ float3 vec_add(const float3 a, const float3 b);
__host__ __device__ float3 vec_sub(const float3 a, const float3 b);
__host__ __device__ float3 vec_scale(const float3 a, const float k);
__host__ __device__ float3 vec_negate(const float3 vec);
__host__ __device__ float3 vec_mult(const float3 a, const float3 b);
__host__ __device__ float3 vec_cross(const float3 a, const float3 b);
__host__ __device__ float vec_dot(const float3 a, const float3 b);
__host__ __device__ float vec_length(const float3 a);
__host__ __device__ float3 vec_norm(const float3 a);

__device__ float3 random_unit_vector();
__host__ __device__ float3 vector_reflect(const float3 incoming, const float3 normal);
__host__ __device__ float3 vector_refract(const float3 incoming, const float3 normal, float refract_coefficient);

__host__ __device__ float4 quaternion_from_euler(const float rx, const float ry, const float rz);
__host__ __device__ float3 vec_rotate(const float3 vec, const float4 quat);
