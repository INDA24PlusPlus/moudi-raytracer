#pragma once

#include "camera.h"
#include "common.h"
#include "list.h"
#include <stdatomic.h>

#define INTERSECTION_NOT_FOUND -1.0
#define MAX_RAY_COLLISIONS 1

typedef struct object {
    void * ptr;
    enum obj_type {
        PLANE,
        SPHERE
    } type;
} Object;

typedef struct scene {
    Camera_t camera;
    List objects;
    atomic_bool rendered;
    atomic_int rendering_progress;

    Vector3 image[IMAGE_HEIGHT][IMAGE_WIDTH];
} Scene;

void init_scene(Scene * scene, double FOV, double focal_length);

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr);

void scene_render(Scene * scene);
