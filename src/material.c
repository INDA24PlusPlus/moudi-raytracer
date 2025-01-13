#include "material.h"
#include "common.h"
#include "ray.h"
#include "vec.h"
#include <math.h>
#include <raymath.h>


Material_t new_material(Vector3 color, double reflectance, double refraction_index) {
    return (Material_t) { .albedo = color, .reflectance = reflectance, .refraction_index = refraction_index };
}

static double reflectance(double cosine, double refraction_index) {
    double r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

char material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec, Ray_t * out_ray) {
    Vector3 direction;

    if (material.refraction_index != 0.0) {
        double refraction_index = rec.front_face ? (1.0 / material.refraction_index) : material.refraction_index;

        double cos_theta = -Vector3DotProduct(ray.direction, rec.normal);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        char cannot_refract = (refraction_index * sin_theta) > 1.0;

        if (cannot_refract || reflectance(cos_theta, refraction_index) > RANDOM_DOUBLE()) {
            direction = Vector3Normalize(vector_reflect(ray.direction, rec.normal));
        } else {
            direction = Vector3Normalize(vector_refract(ray.direction, rec.normal, refraction_index));
        }

        *out_ray = new_ray(direction, ray.position);
        return 1;
    }

    Vector3 reflect = vector_reflect(ray.direction, rec.normal);
    if (RANDOM_DOUBLE() <= material.reflectance) { // reflect
        direction = Vector3Normalize(reflect);
    } else { // scatter
        direction = Vector3Normalize(Vector3Add(reflect, Vector3Scale(random_unit_vector(), 1.0 - material.fuzziness)));
    }

    *out_ray = new_ray(rec.point, direction);
    return Vector3DotProduct(direction, rec.normal) > 0.0;
}
