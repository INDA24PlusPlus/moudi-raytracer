#pragma once
#include "common.h"

typedef struct ray Ray_t;
typedef struct hit_record Hit_record;


struct ray {
    float3 position;
    float3 direction;
};

#include "material.h"

struct hit_record {
    float3 point;
    float3 normal;
    float t;
    Material_t material;
    char front_face;
};

#include "scene.h"

__device__ Ray_t new_ray(float3 start, float3 dir);
__device__ Ray_t get_ray(Camera_t * camera, size_t i, size_t j);

__device__ float3 ray_at(Ray_t ray, float t);
__device__ float3 ray_color(Scene * scene, Ray_t ray);
