#pragma once

#include "common.h"
#include "material.h"

typedef struct plane {
    Vector3 position;
    Vector3 rotation;
    Vector3 dimensions;

    Vector3 color;
    Material_t material;

    Vector3 normal;
} Plane;

typedef struct sphere {
    Vector3 position;
    double radius;

    Vector3 color;
    Material_t material;
} Sphere;

typedef struct object {
    union value {
        Sphere sphere;
        Plane plane;
    } value;

    enum obj_type {
        PLANE,
        SPHERE
    } type;
} Object;

#include "ray.h"

double obj_intersects_ray(Object obj, Ray_t ray);
Vector3 obj_get_normal(Object obj, Vector3 point_of_collision);
Vector3 obj_get_color(Object obj);
Material_t obj_get_material(Object obj);
