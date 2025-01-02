#include "scene.h"

#include "common.h"
#include "camera.h"
#include "list.h"
#include "plane.h"
#include "ray.h"
#include "sphere.h"
#include <raymath.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

void init_scene(Scene * scene, double FOV, double focal_length) {
    scene->camera = new_camera(FOV, focal_length);
    scene->objects = new_list(sizeof(Object *));
}

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr) {
    Object * object = malloc(sizeof(*object));
    *object = (Object) {.ptr = ptr, .type = type};
    list_push(&scene->objects, object);
}

double obj_intersects_ray(struct object obj, Ray_t * ray) {
    switch (obj.type) {
        case PLANE:
            return plane_intersects_ray(*(Plane *) obj.ptr, ray);
        case SPHERE:
            return sphere_intersects_ray(*((Sphere *) obj.ptr), ray);
    } 

    println("Invalid object type for obj_intersects_ray: {i}", obj.type);
    exit(1);
}

Vector3 obj_get_normal(struct object obj, Vector3 point_of_collision) {
    switch (obj.type) {
        case PLANE:
            return ((Plane *) obj.ptr)->normal;
        case SPHERE:
            return Vector3Subtract(point_of_collision, ((Sphere *) obj.ptr)->position);
    } 

    println("Invalid object type for obj_intersects_ray: {i}", obj.type);
    exit(1);
}

Vector3 obj_get_color(struct object obj) {
    switch (obj.type) {
        case PLANE:
            return ((Plane *) obj.ptr)->color;
        case SPHERE:
            return ((Sphere *) obj.ptr)->color;
    }

    println("Invalid object type for obj_get_color: {i}", obj.type);
    exit(1);
}

void scene_find_ray_intersection(Scene * scene, Ray_t * ray) {
    while (1) {
        double t = INTERSECTION_NOT_FOUND;
        struct object collision_obj;

        for (size_t i = 0; i < scene->objects.count; ++i) {
            Object obj = *(Object *) list_at(scene->objects, i);
            double t_temp = obj_intersects_ray(obj, ray);

            // if an intersection was found and the intersection is earlier than the previous intersection found
            if (t_temp != INTERSECTION_NOT_FOUND && (t_temp < t || t == INTERSECTION_NOT_FOUND)) {
                t = t_temp;
                collision_obj = obj;
            }
        }

        if (t == INTERSECTION_NOT_FOUND) {
            return;
        }

        /* println("intersection"); */

        ray->depth += 1;

        Vector3 point_of_collision = ray_at(*ray, t);
        Vector3 normal = obj_get_normal(collision_obj, point_of_collision);
        char front_face = Vector3DotProduct(ray->direction, normal) > 0.0;
        
        if (!front_face) {
            normal = Vector3Negate(normal);
        }

        ray->color = Vector3Scale(Vector3Add(normal, Vec3(1.0, 1.0, 1.0)), 0.5);
        /* println("color = {s}", vec_to_string(ray->color)); */

        if (MAX_RAY_COLLISIONS <= ray->depth) {
            return;
        }

        ray->position = ray_at(*ray, t);
        ray->direction = Vector3Add(Vector3Scale(Vector3Project(ray->direction, normal), 2.0), ray->direction);
    }
}

void scene_render(Scene * scene) {
    Vector2 viewport = scene->camera.viewport;
    Vector3 viewport_h = Vec3(viewport.x, 0.0, 0.0),
         viewport_v = Vec3(0.0, -viewport.y, 0.0);

    Vector3 pixel_delta_h = Vector3Scale(viewport_h, 1.0 / IMAGE_WIDTH),
         pixel_delta_v = Vector3Scale(viewport_v, 1.0 / IMAGE_HEIGHT);
    
    Vector3 camera_normal = Vector3RotateByQuaternion(DEFAULT_VEC, QuaternionFromEuler(ExpandVec3(scene->camera.rotation)));
    Vector3 camera_to_viewport = Vector3Scale(camera_normal, scene->camera.focal_length);

    Vector3 viewport_start = Vector3Subtract(
                                Vector3Add(scene->camera.position, camera_to_viewport),
                                Vector3Scale(Vector3Add(viewport_h, viewport_v), 0.5));

    Vector3 pixel_center_offset = Vector3Add(viewport_start, Vector3Scale(Vector3Add(pixel_delta_v, pixel_delta_h), 0.5));
    
    println("Scene: Objects [{i}]", scene->objects.count);

    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        atomic_store_explicit(&scene->rendering_progress, (100 * (y + 1) / IMAGE_HEIGHT), memory_order_relaxed);
        Vector3 pixel_v = Vector3Scale(pixel_delta_v, y);
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            Vector3 pixel_h = Vector3Scale(pixel_delta_h, x);
            Vector3 pixel_center = Vector3Add(pixel_center_offset, Vector3Add(pixel_v, pixel_h));
            Vector3 ray_direction = Vector3Subtract(pixel_center, scene->camera.position);

            Ray_t ray = new_ray(scene->camera.position, ray_direction);

            scene_find_ray_intersection(scene, &ray);

            scene->image[y][x] = ray_color(ray);
        }
    }
    atomic_store_explicit(&scene->rendered, 1, memory_order_relaxed);
    putchar('\n');
}
