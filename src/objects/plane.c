#include "plane.h"
#include "common.h"
#include "quaternion.h"
#include "vec3.h"

#define ROT_QUAT

void plane_set_position(Plane * plane, Vec3 pos) {
    plane->position = pos;

}

void plane_set_rotation(Plane * plane, Vec3 rot) {
    plane->rotation = rot;
    plane_update_normal(plane);
}

void plane_set_size(Plane * plane, Vec3 dimensions) {
    plane->dimensions = dimensions;
}

void plane_set_color(Plane * plane, Vec3 color) {
    plane->color = color;
}

void plane_update_normal(Plane * plane) {
#ifdef ROT_QUAT
    Quaternion vec_quat = {.w = 0.0, .x = 0.0, .y = 0.0, .z = 1.0}; // default normal
    Quaternion rotation_quat = quaternion_from_euler(plane->rotation.x, plane->rotation.y, plane->rotation.z);
    Quaternion rotation_quat_conj = quaternion_conjugate(rotation_quat);
    Quaternion rotated_vec = quaternion_multiply(quaternion_multiply(rotation_quat, vec_quat), rotation_quat_conj);
    plane->normal = (Vec3) {.x = rotated_vec.x, .y = rotated_vec.y, .z = rotated_vec.z};
#elifdef ROT_XY
        plane->normal = (Vec3) {.x = cos(plane->rotation.x)*sin(plane->rotation.y), .y = -sin(plane->rotation.x), .z = cos(plane->rotation.x) * cos(plane->rotation.y)};
#elifdef ROT_YX
        plane->normal = (Vec3) {.x = sin(plane->rotation.y), .y = -sin(plane->rotation.x) * cos(plane->rotation.y), .z = cos(plane->rotation.x) * cos(plane->rotation.y)};
#endif
}
