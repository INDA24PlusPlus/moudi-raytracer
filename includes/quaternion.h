#pragma once

typedef struct quaternion {
    double w, x, y, z;
} Quaternion;

Quaternion quaternion_from_euler(double x, double y, double z);
#define quaternion_from_euler_vec(vec) quaternion_from_euler((vec).x, (vec).y, (vec).z)

Quaternion quaternion_conjugate(Quaternion);
Quaternion quaternion_multiply(Quaternion, Quaternion);
