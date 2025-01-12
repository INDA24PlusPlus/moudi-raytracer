#pragma once

#include "camera.h"
#include "list.h"
#include <stdatomic.h>

#define INTERSECTION_NOT_FOUND -1.0

typedef struct scene {
    Camera_t camera;
    List objects;
    atomic_bool rendered;
    atomic_int rendering_progress;

    Vector3 image[IMAGE_HEIGHT][IMAGE_WIDTH];
} Scene;

#include "ray.h"
#include "object.h"

void init_scene(Scene * scene, double FOV, double focal_length);

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr);

Hit_record scene_find_ray_intersection(Scene * scene, Ray_t ray);
void scene_render(Scene * scene);
