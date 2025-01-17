#include "sphere.h"

#include "common.h"
#include "material.h"
#include "ray.h"
#include "scene.h"

__host__ Sphere new_sphere(float3 position, float radius) {
    return (Sphere) {.position = position, .radius = radius};
}

__host__ __device__ char * sphere_to_string(Sphere sphere) {
    char * str;
    asprintf(&str, "Sphere %lf: (%lf, %lf, %lf)", sphere.radius, ExpandVec3(sphere.position));
    return str;
}

__host__ __device__ void set_sphere_color(Sphere * sphere, float3 color) {
    sphere->color = color;
}

__host__ __device__ void set_sphere_material(Sphere * sphere, Material_t mat) {
    sphere->material = mat;
}

__device__ float sphere_intersects_ray(const Sphere sphere, const Ray_t ray) {
    float a = vec_dot(ray.direction, ray.direction);
    
    if (a == 0.0) {
        return INTERSECTION_NOT_FOUND;
    }
 
    float3 cam_to_sphere = vec_sub(sphere.position, ray.position);

    float h = vec_dot(ray.direction, cam_to_sphere);
    float c = vec_dot(cam_to_sphere, cam_to_sphere) - sphere.radius * sphere.radius;

    float disc = h * h - a * c;

    // if discriminant is 0 then the intersection point is a tangent and less than 0 means no intersection
    if (disc <= 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    float disc_sqrt = sqrt(disc);
    float t = (h - disc_sqrt) / a;
    
    if (t < 0.0) {
        t = (h + disc_sqrt) / a;
        if (t < 0.0) {
            return INTERSECTION_NOT_FOUND;
        }
    }

    return t;
}
