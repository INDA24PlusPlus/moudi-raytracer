#include "camera.h"
#include "common.h"
#include "vec.h"

Camera_t new_camera(float FOV_y, float focal_length) {
    return (Camera_t) {.FOV_y = FOV_y, .focal_length = focal_length };
}

void camera_set_position(Camera_t * camera, float3 position) {
    camera->position = position;
}

void camera_set_rotation(Camera_t * camera, float3 rotation) {
    camera->rotation = rotation;
}

void camera_set_focal_length(Camera_t * camera, float focal_length) {
    camera->focal_length = focal_length;
}

void camera_set_FOV_y(Camera_t * camera, float FOV_y) {
    camera->FOV_y = FOV_y;
}

__host__ __device__ void camera_update(Camera_t * camera) {
    float height = 2 * tan(M_PI * camera->FOV_y / 360.0);
    camera->viewport = (float2) {.x = (height * IMAGE_WIDTH) / IMAGE_HEIGHT, .y = height};

    float4 rotation_quaternion = quaternion_from_euler(ExpandVec3F(camera->rotation, DEGREE_TO_RADIAN));
    // this should be normalized for the viewport calculations to work
    camera->normal = vec_norm(vec_rotate(DEFAULT_VEC, rotation_quaternion));

    camera->viewport_h = vec_rotate(Vec3(camera->viewport.x, 0, 0.0), rotation_quaternion);
    camera->viewport_v = vec_rotate(Vec3(0, -camera->viewport.y, 0.0), rotation_quaternion);

    camera->pixel_delta_h = vec_scale(camera->viewport_h, 1.0 / IMAGE_WIDTH);
    camera->pixel_delta_v = vec_scale(camera->viewport_v, 1.0 / IMAGE_HEIGHT);

    float3 viewport_center = vec_add(camera->position, vec_scale(camera->normal, camera->focal_length));
    float3 offset_to_viewport_top_left = vec_scale(vec_add(camera->viewport_h, camera->viewport_v), -0.5);

    camera->viewport_start = vec_add(viewport_center, offset_to_viewport_top_left);
}
