#include "scene.h"

#include "common.h"
#include "camera.h"
#include "list.h"
#include "object.h"
#include "plane.h"
#include "ray.h"
#include "sphere.h"
#include "vec.h"
#include <raymath.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

void init_scene(Scene * scene, double FOV, double focal_length) {
    scene->camera = new_camera(FOV, focal_length);
    scene->objects = new_list(sizeof(Object *));
    camera_update(&scene->camera);

    println("Camera: viewport [{2d:, }]", scene->camera.viewport.x, scene->camera.viewport.y);
}

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr) {
    Object * object = (Object *) malloc(sizeof(*object));
    object->type = type;

    switch (type) {
        case PLANE:
            object->value.plane = *(Plane *) ptr; break;
        case SPHERE:
            object->value.sphere = *(Sphere *) ptr; break;
    }

    list_push(&scene->objects, object);
}

Hit_record scene_find_ray_intersection(Scene * scene, Ray_t ray) {
    Hit_record rec = { .t = INTERSECTION_NOT_FOUND };
    Object collision_obj;

    for (size_t i = 0; i < scene->objects.count; ++i) {
        Object obj = *(Object *) list_at(scene->objects, i);
        double t_temp = obj_intersects_ray(obj, ray);

        // if an intersection was found and the intersection is earlier than the previous intersection found
        if (t_temp != INTERSECTION_NOT_FOUND && ((T_MIN_CUTOFF < t_temp && t_temp < rec.t) || rec.t == INTERSECTION_NOT_FOUND)) {
            rec.t = t_temp;
            collision_obj = obj;
        }
    }

    if (rec.t == INTERSECTION_NOT_FOUND) {
        return rec;
    }

    rec.point = ray_at(ray, rec.t);
    rec.normal = Vector3Normalize(obj_get_normal(collision_obj, rec.point));
    rec.material = obj_get_material(collision_obj);

    if (Vector3DotProduct(ray.direction, rec.normal) > 0.0) {
        rec.normal = Vector3Negate(rec.normal);
        rec.front_face = 0;
    } else {
        rec.front_face = 1;
    }

    return rec;
}

void scene_render(Scene * scene) {
    camera_update(&scene->camera);
    
    println("Scene: Objects [{i}]", scene->objects.count);

    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        atomic_store_explicit(&scene->rendering_progress, (100 * (y + 1) / IMAGE_HEIGHT), memory_order_relaxed);
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {

            Vector3 pixel_color = Vector3Zero();
            for (size_t sample = 0; sample < SAMPLES_PER_PIXEL; ++sample) {
                Ray_t ray = get_ray(&scene->camera, x, y);
                pixel_color = Vector3Add(pixel_color, ray_color(scene, ray, 0));
            }

            pixel_color = Vector3Scale(pixel_color, 1.0 / SAMPLES_PER_PIXEL);
            scene->image[y][x] = (Vector3) {ExpandVec3F(pixel_color, LINEAR_TO_GAMMA)};
        }
    }

    atomic_store_explicit(&scene->rendered, 1, memory_order_relaxed);
    putchar('\n');
}
