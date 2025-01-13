#include "plane.h"
#include "common.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include <stdlib.h>

#include "raymath.h"
#include "vec.h"

#define ROT_QUAT

Plane new_plane(Vector3 pos, Vector3 dimensions) {
    return (Plane) {.position = pos, .normal = DEFAULT_VEC, .dimensions = dimensions};
}

void plane_set_position(Plane * plane, Vector3 pos) {
    plane->position = pos;
}

void plane_set_rotation(Plane * plane, Vector3 rot) {
    plane->rotation = rot;
    plane_update_normal(plane);
}

void plane_set_size(Plane * plane, Vector3 dimensions) {
    plane->dimensions = dimensions;
}

void plane_set_material(Plane * plane, Material_t mat) {
    plane->material = mat;
}

void plane_update_normal(Plane * plane) {
    plane->normal = Vector3RotateByQuaternion(DEFAULT_VEC, QuaternionFromEuler(ExpandVec3F(plane->rotation, DEGREE_TO_RADIAN)));
}

double plane_intersects_ray(Plane plane, Ray_t ray) {
    double a = Vector3DotProduct(plane.normal, Vector3Subtract(plane.position, ray.position));
    double b = Vector3DotProduct(plane.normal, ray.direction);

    if (b == 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    double t = -a / b;

    if (t < 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    Vector3 intersection_point = ray_at(ray, t);
    Vector3 diff = Vector3Subtract(intersection_point, plane.position);

    // is not within the plane bounds
    if (!(-plane.dimensions.x <= diff.x && diff.x <= plane.dimensions.x) ||
        !(-plane.dimensions.y <= diff.y && diff.y <= plane.dimensions.y)) {
        return INTERSECTION_NOT_FOUND;
    }

    return t;
}
