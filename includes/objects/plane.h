#pragma once

#include "vec3.h"

typedef struct plane {
    Vec3 color;

    Vec3 position;
    Vec3 rotation;
    Vec3 dimensions;

    Vec3 normal;
} Plane;

void plane_set_position(Plane * plane, Vec3 new_pos);
void plane_set_rotation(Plane * plane, Vec3 new_rot);
void plane_set_color(Plane * plane, Vec3 color);
void plane_set_size(Plane * plane, Vec3 dimensions);

void plane_update_normal(Plane * plane);
