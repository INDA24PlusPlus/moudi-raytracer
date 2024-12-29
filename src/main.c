#include "common.h"

#include "plane.h"
#include "ppm.h"

int main() {

    Plane plane = {0};
    Vec3 rotation = vec_new(90.0, 0.0, 0.0);
    plane_set_rotation(&plane, rotation);

    println("[{d}, {d}, {d}]", plane.normal.x, plane.normal.y, plane.normal.z);

}
