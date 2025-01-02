#include "ray.h"
#include <raymath.h>

const Vector3 START_COLOR = Vec3(255.0, 255.0, 255.0);
const Vector3 END_COLOR = Vec3(127.5, 178.5, 255.0);

Ray_t new_ray(Vector3 start, Vector3 direction) {
    return (Ray_t) {.position = start, .direction = direction};
}

Vector3 ray_at(Ray_t ray, double t) {
    return Vector3Add(ray.position, Vector3Scale(ray.direction, t));
}

Vector3 ray_color(Ray_t ray) {
    if (ray.depth == 0) {
        Vector3 unit_dir = Vector3Normalize(ray.direction);
        double a = 0.5 * (unit_dir.y + 1.0);
        return Vector3Add(Vector3Scale(START_COLOR, 1.0 - a), Vector3Scale(END_COLOR, a));
    }

    return ray.color;
}
