#pragma once

#include "quaternion.h"
#define VEC_NUMERIC_TYPE double

typedef struct vec3 {
    VEC_NUMERIC_TYPE x, y, z;
} Vec3;

Vec3 vec_new(VEC_NUMERIC_TYPE x, VEC_NUMERIC_TYPE y, VEC_NUMERIC_TYPE z);

Vec3 vec_add(Vec3 a, Vec3 b);
Vec3 vec_sub(Vec3 a, Vec3 b);
Vec3 vec_scalar(Vec3 a, VEC_NUMERIC_TYPE k);

VEC_NUMERIC_TYPE vec_dot(Vec3 a, Vec3 b);
Vec3 vec_cross(Vec3 a, Vec3 b);

/* static inline Vec3 vec_reduce(Vec3 a); */
static inline VEC_NUMERIC_TYPE vec_length(Vec3 a);

Vec3 vec_rotate(Vec3 vec, Quaternion rotation);
