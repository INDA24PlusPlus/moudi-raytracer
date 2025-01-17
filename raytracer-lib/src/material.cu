#include "material.h"

#include "common.h"
#include "rand.h"
#include <math.h>

Material_t new_material(float3 color, float fuzziness, float reflectance, float refraction_index) {
    if (refraction_index != 0.0f) {
        color = VEC_ONE;
    }
    return (Material_t) { .albedo = color, .fuzziness = fuzziness, .reflectance = reflectance, .refraction_index = refraction_index };
}

__device__ static float reflectance(float cosine, float refraction_index) {
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ char material_scatter_ray(Material_t material, Ray_t ray, Hit_record rec, float3 * attenuation, Ray_t * out_ray) {
    float3 direction;

    if (material.refraction_index != 0.0f) {
        *attenuation = VEC_ONE;
        float refraction_index = rec.front_face ? (1.0f / material.refraction_index) : material.refraction_index;

        float cos_theta = -vec_dot(ray.direction, rec.normal);
        if (cos_theta > 1.0f) { cos_theta = 1.0f; }
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        char cannot_refract = (refraction_index * sin_theta) > 1.0f;

        if (cannot_refract || reflectance(cos_theta, refraction_index) > rand_float()) {
            direction = vec_norm(vector_reflect(ray.direction, rec.normal));
        } else {
            direction = vec_norm(vector_refract(ray.direction, rec.normal, refraction_index));
        }

        *out_ray = new_ray(direction, ray.position);
        return 1;
    }

    float3 reflect = vec_norm(vector_reflect(ray.direction, rec.normal));
    float rand = rand_float();
    if (rand <= material.reflectance) { // reflect
        direction = reflect;
    } else { // scatter
        direction = vec_norm(vec_add(reflect, vec_scale(random_unit_vector(), 1.0f - material.fuzziness)));
    }

    *attenuation = material.albedo;

    *out_ray = new_ray(rec.point, direction);
    return vec_dot(direction, rec.normal) > 0.0f;
}
