#include "ray.h"

#include "camera.h"
#include "rand.h"
#include "common.h"
#include "material.h"
#include "scene.h"
#include "vec.h"

const float3 START_COLOR = Vec3(1.0, 1.0, 1.0);
const float3 END_COLOR = Vec3(0.5, 0.7, 1.0);

__device__ float2 sample_square() {
    return (float2) {.x = rand_float() - 0.5, .y = rand_float() - 0.5};
}

__device__ Ray_t get_ray(Camera_t * camera, size_t i, size_t j) {
    float2 offset = sample_square();
    float3 pixel_center =  vec_add(camera->viewport_start,
                                vec_add(vec_scale(camera->pixel_delta_h, i + offset.x),
                                            vec_scale(camera->pixel_delta_v, j + offset.y)));
    float3 ray_direction = vec_norm(vec_sub(pixel_center, camera->position));
    // printf("ray fin: [%d, %d]\n", i, j);

    return new_ray(camera->position, ray_direction);
}

__device__ Ray_t new_ray(float3 start, float3 direction) {
    return (Ray_t) {.position = start, .direction = direction};
}

__device__ float3 ray_at(Ray_t ray, float t) {
    return vec_add(ray.position, vec_scale(ray.direction, t));
}

__device__ float3 ray_color(Scene * scene, Ray_t ray) {
    float3 color = VEC_ONE;
    Hit_record rec;
 
    for (size_t i = 0; i < MAX_RAY_COLLISIONS; ++i) {
        rec = scene_find_ray_intersection(scene, ray);
        if (rec.t == INTERSECTION_NOT_FOUND) {
            float3 unit_dir = vec_norm(ray.direction);
            float a = 0.5f * (unit_dir.y + 1.0f);
            float3 background = vec_add(vec_scale(START_COLOR, 1.0f - a), vec_scale(END_COLOR, a));
            return vec_mult(color, background);
        }

        Ray_t scattered;
        float3 attenuation;
        if (material_scatter_ray(rec.material, ray, rec, &attenuation, &scattered)) {
            ray = scattered;
            color = vec_mult(color, attenuation);
        } else {
            return VEC_ZERO;
        }
    }

    return VEC_ZERO;
}
