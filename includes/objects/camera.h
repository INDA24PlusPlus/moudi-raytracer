#pragma once

#include <raylib.h>

typedef struct camera {
    Vector3 position, rotation, normal;
    double focal_length, FOV_y;

    Vector2 viewport;
} Camera_t;

Camera_t new_camera(double focal_length, double FOV_y);

void camera_set_position(Camera_t * camera, Vector3 position);
void camera_set_rotation(Camera_t * camera, Vector3 rotation);

void camera_set_focal_length(Camera_t * camera, double focal_length);
void camera_set_FOV_y(Camera_t * camera, double FOV_y);

void camera_update(Camera_t * camera);
