#pragma once

#include "camera.h"
#include "list.h"
#include <cuda/atomic>

#define INTERSECTION_NOT_FOUND -1.0

typedef struct scene {
    Camera_t camera;
    List objects;
    bool rendered;
    int rendering_progress;

    float3 image[IMAGE_HEIGHT][IMAGE_WIDTH];
} Scene;

#include "ray.h"
#include "object.h"

void init_scene(Scene * scene, float FOV, float focal_length);

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr);
void ** obj_list_to_device(List list);

__device__ Hit_record scene_find_ray_intersection(Scene * scene, Ray_t ray);
void scene_render(Scene * scene, Scene * device_scene, void ** host_objects, void ** device_objects);
