#include "vec.h"
#include "common.h"
#include <math.h>
#include <raymath.h>

Vector3 random_unit_vector() {
    while (1) {
        Vector3 vec = {RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1)};
        double length_sq = Vector3LengthSqr(vec);
        if (1e-6 < length_sq && length_sq <= 1) {
            return Vector3Scale(vec, 1.0 / sqrt(length_sq));
        }
    }
}

Vector3 vector_reflect(const Vector3 incoming, const Vector3 normal) {
    return Vector3Subtract(incoming, Vector3Scale(normal, 2.0 * Vector3DotProduct(incoming, normal)));
}

Vector3 vector_refract(const Vector3 incoming, const Vector3 normal, double refract_coefficient) {
    double cos_theta = -Vector3DotProduct(incoming, normal);

    const Vector3 out_perp = Vector3Scale(Vector3Add(incoming, Vector3Scale(normal, cos_theta)), refract_coefficient);
    const Vector3 out_para = Vector3Scale(normal, -sqrt(fabs(1.0 - Vector3LengthSqr(out_perp))));

    return Vector3Add(out_perp, out_para);
}
