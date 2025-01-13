#pragma once

#include <raylib.h>

#define Vec3(x, y, z) ((Vector3) {x, y, z})
#define ExpandVec3(vec) (vec).x, (vec).y, (vec).z
#define ExpandVec3F(vec, f) f((vec).x), f((vec).y), f((vec).z)

#define DEFAULT_VEC Vec3(0.0, 0.0, 1.0)

Vector3 random_unit_vector();
Vector3 vector_reflect(const Vector3 incoming, const Vector3 normal);
Vector3 vector_refract(const Vector3 incoming, const Vector3 normal, double refract_coefficient);
