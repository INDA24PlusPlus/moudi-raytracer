#include "object.h"

#include "plane.h"
#include "sphere.h"

double obj_intersects_ray(Object obj, Ray_t ray) {
    switch (obj.type) {
        case PLANE:
            return plane_intersects_ray(obj.value.plane, ray);
        case SPHERE:
            return sphere_intersects_ray(obj.value.sphere, ray);
    } 

    println("Invalid object type for obj_intersects_ray: {i}", obj.type);
    exit(1);
}

Vector3 obj_get_normal(Object obj, Vector3 point_of_collision) {
    switch (obj.type) {
        case PLANE:
            return obj.value.plane.normal;
        case SPHERE:
            return Vector3Subtract(point_of_collision, obj.value.sphere.position);
    } 

    println("Invalid object type for obj_intersects_ray: {i}", obj.type);
    exit(1);
}

Vector3 obj_get_color(Object obj) {
    switch (obj.type) {
        case PLANE:
            return obj.value.plane.color;
        case SPHERE:
            return obj.value.sphere.color;
    }

    println("Invalid object type for obj_get_color: {i}", obj.type);
    exit(1);
}

Material_t obj_get_material(Object obj) {
    switch (obj.type) {
        case PLANE:
            return obj.value.plane.material;
        case SPHERE:
            return obj.value.sphere.material;
    }

    print("Invalid object type for obj_get_material: {i}", obj.type);
    exit(1);
}
