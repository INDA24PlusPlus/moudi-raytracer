#pragma once

#include "common.h"

typedef struct material {
    Vector3 albedo;
    double fuzziness;
    double reflectance;
    double refraction_index;
} Material_t;

#include "ray.h"

Material_t new_material(Vector3 color, double reflectance, double refraction_index);
char material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec, Ray_t * out_ray);
