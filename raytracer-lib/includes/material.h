#pragma once

#include "common.h"

typedef struct material {
    float3 albedo;
    float fuzziness;
    float reflectance;
    float refraction_index;
} Material_t;

#include "ray.h"

Material_t new_material(float3 color, float fuzziness, float reflectance, float refraction_index);
__device__ char material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec, float3 * attenuation, Ray_t * out_ray);
