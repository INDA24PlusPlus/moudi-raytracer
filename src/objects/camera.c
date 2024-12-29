#include "camera.h"

Camera new_camera(int FOV) {
    return (Camera) {.FOV = FOV};
}

void camera_set_pos(Camera * camera, int x, int y, int z) {
    camera->position = (Vec3) {.x = x, .y = y, .z = z};
}

void camera_set_rot(Camera * camera, int x, int y, int z) {
    camera->rotation = (Vec3) {.x = x, .y = y, .z = z};
}
