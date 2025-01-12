#include "vec.h"
#include "common.h"
#include <raymath.h>

Vector3 random_unit_vector() {
    while (1) {
        Vector3 vec = {RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1), RANDOM_RANGE(-1, 1)};
        double length_sq = Vector3LengthSqr(vec);
        if (1e-10 < length_sq && length_sq <= 1) {
            return Vector3Scale(vec, 1.0 / sqrt(length_sq));
        }
    }
}

Vector3 hemisphere_random_vector(Vector3 normal) {
    Vector3 random_vec = random_unit_vector();
    if (Vector3DotProduct(normal, random_vec) > 0.0) {
        return random_vec;
    } else {
        return Vector3Negate(random_vec);
    }
}
