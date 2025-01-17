#pragma once

typedef struct camera {
    float3 position, rotation, normal;
    float FOV_y, focal_length;

    float2 viewport;
    float3 viewport_start, viewport_h, viewport_v;
    float3 pixel_delta_h, pixel_delta_v;
} Camera_t;

Camera_t new_camera(float focal_length, float FOV_y);

void camera_set_position(Camera_t * camera, float3 position);
void camera_set_rotation(Camera_t * camera, float3 rotation);

void camera_set_focal_length(Camera_t * camera, float focal_length);
void camera_set_FOV_y(Camera_t * camera, float FOV_y);

__host__ __device__ void camera_update(Camera_t * camera);
