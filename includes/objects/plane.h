#pragma once

#include "common.h"
#include "ray.h"

typedef struct plane {
    Vector3 color;

    Vector3 position;
    Vector3 rotation;
    Vector3 dimensions;

    Vector3 normal;
} Plane;

Plane new_plane(Vector3 pos, Vector3 dimensions);

void plane_set_position(Plane * plane, Vector3 new_pos);
void plane_set_rotation(Plane * plane, Vector3 new_rot);
void plane_set_color(Plane * plane, Vector3 color);
void plane_set_size(Plane * plane, Vector3 dimensions);

void plane_update_normal(Plane * plane);
double plane_intersects_ray(Plane plane, Ray_t * ray);
