#include "camera.h"
#include "common.h"

Camera_t new_camera(double focal_length, double FOV_y) {
    return (Camera_t) {.focal_length = focal_length, .FOV_y = FOV_y};
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
    camera->normal = Vector3RotateByQuaternion(DEFAULT_VEC, QuaternionFromEuler(camera->rotation.x, camera->rotation.y, camera->rotation.z));
}
