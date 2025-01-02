#pragma once
#include "ray.h"

typedef struct sphere {
    Vector3 color;
    Vector3 position;
    double radius;
} Sphere;

Sphere new_sphere(Vector3 position, double radius);
char * sphere_to_string(Sphere sphere);

void set_sphere_color(Sphere * sphere, Vector3 color);
double sphere_intersects_ray(Sphere sphere, Ray_t * ray);
