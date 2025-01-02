#pragma once

#include "common.h"

typedef struct ray {
    Vector3 position;
    Vector3 direction;

    Vector3 color;
    int depth;
} Ray_t;

Ray_t new_ray(Vector3 start, Vector3 dir);

Vector3 ray_at(Ray_t ray, double t);
Vector3 ray_color(Ray_t ray);
