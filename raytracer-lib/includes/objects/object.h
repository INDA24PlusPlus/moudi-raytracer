#pragma once


enum obj_type {
    PLANE,
    SPHERE
};

#include "common.h"
#include "material.h"

typedef struct plane {
    float3 position;
    float3 rotation;
    float3 dimensions;

    float3 color;
    Material_t material;

    float3 normal;
} Plane;

typedef struct sphere {
    float3 position;
    float radius;

    float3 color;
    Material_t material;
} Sphere;

typedef struct object {
    union value {
        Sphere sphere;
        Plane plane;
    } value;

    enum obj_type type;
} Object;

#include "ray.h"

__device__ float obj_intersects_ray(Object obj, Ray_t ray);
__device__ float3 obj_get_normal(Object obj, float3 point_of_collision);
__device__ float3 obj_get_color(Object obj);
__device__ Material_t obj_get_material(Object obj);
