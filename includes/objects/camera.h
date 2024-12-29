#pragma once

#include "common.h"

typedef struct camera {
    Vec3 position;
    Vec3 rotation;
    int FOV;
} Camera;

Camera new_camera(int FOV);
void camera_set_pos(Camera * camera, int x, int y, int z);
void camera_set_rot(Camera * camera, int x, int y, int z);
