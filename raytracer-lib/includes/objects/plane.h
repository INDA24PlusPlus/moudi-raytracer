#pragma once
#include "common.h"
#include "material.h"
#include "ray.h"
#include "object.h"

__host__ Plane new_plane(float3 pos, float3 dimensions);

__host__ __device__ void plane_set_position(Plane * plane, float3 new_pos);
__host__ __device__ void plane_set_rotation(Plane * plane, float3 new_rot);
__host__ __device__ void plane_set_color(Plane * plane, float3 color);
__host__ __device__ void plane_set_material(Plane * plane, Material_t mat);
__host__ __device__ void plane_set_size(Plane * plane, float3 dimensions);

__host__ __device__ void plane_update_normal(Plane * plane);
__device__ float plane_intersects_ray(Plane plane, Ray_t ray);
