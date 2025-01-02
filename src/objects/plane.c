#include "plane.h"
#include "common.h"
#include "ray.h"
#include "scene.h"
#include <stdlib.h>

#include "raymath.h"

#define ROT_QUAT

Plane new_plane(Vector3 pos, Vector3 dimensions) {
    return (Plane) {.position = pos, .dimensions = dimensions};
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

void plane_set_color(Plane * plane, Vector3 color) {
    plane->color = color;
}

void plane_update_normal(Plane * plane) {
    plane->normal = Vector3RotateByQuaternion((Vector3) {0.0, 0.0, 1.0}, QuaternionFromEuler(plane->rotation.x, plane->rotation.y, plane->rotation.z));
}

double plane_intersects_ray(Plane plane, Ray_t * ray) {
    double a = Vector3DotProduct(plane.normal, Vector3Subtract(plane.position, ray->position));
    double b = Vector3DotProduct(plane.normal, ray->direction);

    if (b == 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    double t = a / b;

    Vector3 intersection_point = ray_at(*ray, t);
    Vector3 diff = Vector3Subtract(intersection_point, plane.position);

    // is not within the plane bounds
    if (!(abs(diff.x) < plane.dimensions.x && abs(diff.y) < plane.dimensions.y)) {
        return INTERSECTION_NOT_FOUND;
    }

    return t;
}
