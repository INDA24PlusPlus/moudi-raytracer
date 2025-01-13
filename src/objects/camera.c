#include "camera.h"
#include "common.h"
#include "vec.h"
#include <raymath.h>

Camera_t new_camera(double FOV_y, double focal_length) {
    return (Camera_t) {.FOV_y = FOV_y, .focal_length = focal_length };
}

void camera_set_position(Camera_t * camera, Vector3 position) {
    camera->position = position;
}

void camera_set_rotation(Camera_t * camera, Vector3 rotation) {
    camera->rotation = rotation;
}

void camera_set_focal_length(Camera_t * camera, double focal_length) {
    camera->focal_length = focal_length;
}

void camera_set_FOV_y(Camera_t * camera, double FOV_y) {
    camera->FOV_y = FOV_y;
}

void camera_update(Camera_t * camera) {
    double height = 2 * tan(M_PI * camera->FOV_y / 360.0);
    camera->viewport = (Vector2) {.x = (height * IMAGE_WIDTH) / IMAGE_HEIGHT, .y = height};

    Quaternion rotation_quaternion = QuaternionFromEuler(ExpandVec3F(camera->rotation, DEGREE_TO_RADIAN));
    // this should be normalized for the viewport calculations to work
    camera->normal = Vector3Normalize(Vector3RotateByQuaternion(DEFAULT_VEC, rotation_quaternion));

    camera->viewport_h = Vector3RotateByQuaternion(Vec3(camera->viewport.x, 0, 0.0), rotation_quaternion);
    camera->viewport_v = Vector3RotateByQuaternion(Vec3(0, -camera->viewport.y, 0.0), rotation_quaternion);

    camera->pixel_delta_h = Vector3Scale(camera->viewport_h, 1.0 / IMAGE_WIDTH);
    camera->pixel_delta_v = Vector3Scale(camera->viewport_v, 1.0 / IMAGE_HEIGHT);

    Vector3 viewport_center = Vector3Add(camera->position, Vector3Scale(camera->normal, camera->focal_length));
    Vector3 offset_to_viewport_top_left = Vector3Scale(Vector3Add(camera->viewport_h, camera->viewport_v), -0.5);

    camera->viewport_start = Vector3Add(viewport_center, offset_to_viewport_top_left);
}
