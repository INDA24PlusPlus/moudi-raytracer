#include "ray.h"
#include "camera.h"
#include "common.h"
#include "material.h"
#include "scene.h"
#include "vec.h"
#include <raymath.h>

const Vector3 START_COLOR = Vec3(1.0, 1.0, 1.0);
const Vector3 END_COLOR = Vec3(0.5, 0.7, 1.0);

Vector2 sample_square() {
    return (Vector2) {.x = RANDOM_DOUBLE() - 0.5, .y = RANDOM_DOUBLE() - 0.5};
}

Ray_t get_ray(Camera_t * camera, size_t i, size_t j) {
    Vector2 offset = sample_square();
    Vector3 pixel_center =  Vector3Add(camera->viewport_start,
                                Vector3Add(Vector3Scale(camera->pixel_delta_h, i + offset.x),
                                            Vector3Scale(camera->pixel_delta_v, j + offset.y)));
    Vector3 ray_direction = Vector3Normalize(Vector3Subtract(pixel_center, camera->position));

    return new_ray(camera->position, ray_direction);
}

Ray_t new_ray(Vector3 start, Vector3 direction) {
    return (Ray_t) {.position = start, .direction = direction};
}

Vector3 ray_at(Ray_t ray, double t) {
    return Vector3Add(ray.position, Vector3Scale(ray.direction, t));
}

Vector3 ray_color(Scene * scene, Ray_t ray, size_t depth) {
    if (MAX_RAY_COLLISIONS <= depth) {
        return Vector3Zero();
    }

    Hit_record rec = scene_find_ray_intersection(scene, ray);

    if (rec.t != INTERSECTION_NOT_FOUND) {
        Ray_t scattered;
        if (material_scatter_ray(rec.material, ray, rec, &scattered)) {
            return Vector3Multiply(ray_color(scene, scattered, depth + 1), rec.material.albedo);
        }
        return Vector3Zero();
    }

    double a = 0.5 * (ray.direction.y + 1.0);
    return Vector3Add(Vector3Scale(START_COLOR, 1.0 - a), Vector3Scale(END_COLOR, a));
}
