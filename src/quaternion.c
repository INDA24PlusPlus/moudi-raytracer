#include "quaternion.h"
#include "common.h"

Quaternion quaternion_from_euler(double x, double y, double z) {
    const double SX = sin(M_PI * x / 360.0), SY = sin(M_PI * y / 360.0), SZ = sin(M_PI * z / 360.0),
                 CX = cos(M_PI * x / 360.0), CY = cos(M_PI * y / 360.0), CZ = cos(M_PI * z / 360.0);

    return (Quaternion) {
        .w = (CX * CY * CZ) + (SX * SY * SZ),
        .x = (SX * CY * CZ) - (CX * SY * SZ),
        .y = (CX * SY * CZ) + (SX * CY * SZ),
        .z = (CX * CY * SZ) - (SX * SY * CZ),
    };
}

Quaternion quaternion_conjugate(Quaternion quaternion) {
    return (Quaternion) {
        .w = quaternion.w,
        .x = -quaternion.x,
        .y = -quaternion.y,
        .z = -quaternion.z,
    };
}

Quaternion quaternion_multiply(Quaternion a, Quaternion b) {
    return (Quaternion) {
        .w = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z),
        .x = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y),
        .y = (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x),
        .z = (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w),
    };
}
