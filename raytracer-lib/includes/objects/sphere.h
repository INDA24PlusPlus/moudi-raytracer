#pragma once

#include "common.h"
#include "ray.h"
#include "object.h"

__host__ Sphere new_sphere(float3 position, float radius);
__host__ __device__ char * sphere_to_string(Sphere sphere);

__host__ __device__ void set_sphere_color(Sphere * sphere, float3 color);
__host__ __device__ void set_sphere_material(Sphere * sphere, Material_t mat);

__device__ float sphere_intersects_ray(Sphere sphere, Ray_t ray);
