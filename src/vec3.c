#include "vec3.h"
#include "quaternion.h"
#include <math.h>

Vec3 vec_new(VEC_NUMERIC_TYPE x, VEC_NUMERIC_TYPE y, VEC_NUMERIC_TYPE z) {
    return (Vec3) {x, y, z};
}

Vec3 vec_add(Vec3 a, Vec3 b) {
    return (Vec3) { .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
}

Vec3 vec_sub(Vec3 a, Vec3 b) {
    return (Vec3) { .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
}

Vec3 vec_scalar(Vec3 a, VEC_NUMERIC_TYPE k) {
    return (Vec3) {.x = a.x * k, .y = a.y * k, .z = a.z * k};
}

VEC_NUMERIC_TYPE vec_dot(Vec3 a, Vec3 b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

Vec3 vec_cross(Vec3 a, Vec3 b) {
    return (Vec3) {
        .x = (a.y * b.z) - (a.z * b.y),
        .y = (a.x * b.z) - (a.z * b.x),
        .z = (a.x * b.y) - (a.y * b.x)
    };
}

/* VEC_NUMERIC_TYPE gcd(VEC_NUMERIC_TYPE a, VEC_NUMERIC_TYPE b) { */
/*     if (a < 0) { a = -a; } */
/*     if (b < 0) { b = -b; } */
/*  */
/*     if (a == 0) { return b; } */
/*     if (b == 0) { return a; } */
/*  */
/*     int i = log2(a & (-a)); a >>= i; */
/*     int j = log2(b & (-b)); b >>= j; */
/*     int k = (i < j) ? i : j; */
/*  */
/*     while (1) { */
/*         if (a > b) { */
/*             VEC_NUMERIC_TYPE temp = a; */
/*             a = b; */
/*             b = temp; */
/*         } */
/*  */
/*         b -= a; */
/*         if (b == 0) { */
/*             return a << k; */
/*         } */
/*  */
/*         b >>= (int) log2(b & (-b)); */
/*     } */
/* } */
/*  */
/* Vec3 vec_reduce(Vec3 a) { */
/*     VEC_NUMERIC_TYPE d = gcd(a.x, gcd(a.y, a.z)); */
/*     return (Vec3) {.x = a.x / d, .y = a.y / d, .z = a.z / d}; */
/* } */

VEC_NUMERIC_TYPE vec_length(Vec3 a) {
    return vec_dot(a, a);
}

Vec3 vec_rotate(Vec3 vec, Quaternion rot) {
    Vec3 u = vec_new(rot.x, rot.y, rot.z);
    VEC_NUMERIC_TYPE s = rot.w;

    return vec_add(vec_add(
                vec_scalar(u, 2.0 * vec_dot(u, vec)),
                vec_scalar(vec, s * s - vec_dot(u, u))),
                vec_scalar(vec_cross(u, vec), 2.0 * s));
}
