#include "sphere.h"
#include "common.h"
#include "ray.h"
#include "scene.h"
#include <raymath.h>

Sphere new_sphere(Vector3 position, double radius) {
    return (Sphere) {.position = position, .radius = radius};
}

char * sphere_to_string(Sphere sphere) {
    return format("Sphere {d}: ({3d:, })", sphere.radius, ExpandVec3(sphere.position));
}

void set_sphere_color(Sphere * sphere, Vector3 color) {
    sphere->color = color;
}

double sphere_intersects_ray(Sphere sphere, Ray_t * ray) {
    
    /* println("Sphere: [{3d:, }], Ray: [{3d:, }] + t[{3d:, }]",  */
    /*         ExpandVec3(sphere.position), */
    /*         ExpandVec3(ray->position), */
    /*         ExpandVec3(ray->direction)); */
    
    double a = Vector3DotProduct(ray->direction, ray->direction);
    
    if (a == 0.0) {
        return INTERSECTION_NOT_FOUND;
    }
 
    Vector3 cam_to_sphere = Vector3Subtract(sphere.position, ray->position);

    double h = Vector3DotProduct(ray->direction, cam_to_sphere);
    double c = Vector3DotProduct(cam_to_sphere, cam_to_sphere) - sphere.radius * sphere.radius;

    double disc = h * h - a * c;

    // if discriminant is 0 then the intersection point is a tangent
    if (disc <= 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    double disc_sqrt = sqrt(disc);
    double t1 = (h + disc_sqrt) / a,
           t2 = (h - disc_sqrt) / a;
    
    if (t2 >= 0.0 && t2 < t1) {
        t1 = t2;
    }

    if (t1 < 0.0) {
        return INTERSECTION_NOT_FOUND;
    }

    return t1;
}
