#include "material.h"
#include "common.h"
#include "ray.h"
#include "vec.h"
#include <raymath.h>

Material_t new_material(Vector3 color, double reflectance) {
    return (Material_t) { .albedo = color, .reflectance = reflectance };
}

Ray_t material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec) {
    Vector3 scatter_direction;
    /* if (RANDOM_DOUBLE() <= material.reflectance) { // reflect */
    /*     scatter_direction = Vector3Normalize(Vector3Subtract(ray.direction, Vector3Scale(rec.normal, 2.0 * Vector3DotProduct(ray.direction, rec.normal)))); */
    /* } else { // scatter */
        scatter_direction = Vector3Normalize(Vector3Add(rec.normal, random_unit_vector()));
    /* } */

    return new_ray(rec.point, scatter_direction);
}
