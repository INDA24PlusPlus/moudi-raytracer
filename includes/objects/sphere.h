#pragma once

#include "common.h"
#include "ray.h"
#include "object.h"

Sphere new_sphere(Vector3 position, double radius);
char * sphere_to_string(Sphere sphere);

void set_sphere_color(Sphere * sphere, Vector3 color);
void set_sphere_material(Sphere * sphere, Material_t mat);

double sphere_intersects_ray(Sphere sphere, Ray_t ray);

