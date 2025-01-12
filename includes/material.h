#pragma once

#include "common.h"

typedef struct material {
    Vector3 albedo;
    double fuzziness;
    double reflectance;
} Material_t;

#include "ray.h"

Material_t new_material(Vector3 color, double reflectance);
Ray_t material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec);
