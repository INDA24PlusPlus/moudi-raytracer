#pragma once
#include "common.h"

typedef struct ray Ray_t;
typedef struct hit_record Hit_record;

#include "material.h"

struct ray {
    Vector3 position;
    Vector3 direction;
};

struct hit_record {
    Vector3 point;
    Vector3 normal;
    double t;
    Material_t material;
};

#include "scene.h"

Ray_t new_ray(Vector3 start, Vector3 dir);
Ray_t get_ray(Camera_t * camera, size_t i, size_t j);

Vector3 ray_at(Ray_t ray, double t);
Vector3 ray_color(Scene * scene, Ray_t ray, size_t depth);
