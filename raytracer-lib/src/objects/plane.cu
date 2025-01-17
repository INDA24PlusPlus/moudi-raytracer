#include "plane.h"
#include "common.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include <stdlib.h>

#include "vec.h"

#define ROT_QUAT

__host__ Plane new_plane(float3 pos, float3 dimensions) {
    return (Plane) {.position = pos, .dimensions = dimensions, .normal = DEFAULT_VEC};
}

__host__ __device__ void plane_set_position(Plane * plane, float3 pos) {
    plane->position = pos;
}

__host__ __device__ void plane_set_rotation(Plane * plane, float3 rot) {
    plane->rotation = rot;
    plane_update_normal(plane);
}

__host__ __device__ void plane_set_size(Plane * plane, float3 dimensions) {
    plane->dimensions = dimensions;
}

__host__ __device__ void plane_set_material(Plane * plane, Material_t mat) {
    plane->material = mat;
}

__host__ __device__ void plane_update_normal(Plane * plane) {
    plane->normal = vec_rotate(DEFAULT_VEC, quaternion_from_euler(ExpandVec3F(plane->rotation, DEGREE_TO_RADIAN)));
}

__device__ float plane_intersects_ray(Plane plane, Ray_t ray) {
    float a = vec_dot(plane.normal, vec_sub(plane.position, ray.position));
    float b = vec_dot(plane.normal, ray.direction);

    if (b == 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    float t = -a / b;

    if (t < 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    float3 intersection_point = ray_at(ray, t);
    float3 diff = vec_sub(intersection_point, plane.position);

    // is not within the plane bounds
    if (!(-plane.dimensions.x <= diff.x && diff.x <= plane.dimensions.x) ||
        !(-plane.dimensions.y <= diff.y && diff.y <= plane.dimensions.y)) {
        return INTERSECTION_NOT_FOUND;
    }

    return t;
}
