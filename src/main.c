#include "camera.h"
#include "common.h"
#include "list.h"
#include "material.h"
#include "object.h"
#include "plane.h"
#include "scene.h"
#include "sphere.h"
#include "stddef.h"
#include "vec.h"
#include <pthread.h>
#include <raylib.h>
#include <stdatomic.h>
#include <stdlib.h>

Scene scene;
char * progress_cache[101];
#define TEXT_SIZE 50

pthread_t thread_ID = {0};
enum { STATE_START_RENDER, STATE_RENDERING, STATE_RENDERED } state = STATE_START_RENDER; 

void render_next_scene();
void draw_scene();

int main() {
    srand(time(NULL));

    for (size_t i = 0; i <= 100; ++i) {
        progress_cache[i] = format("Waiting for render: {i}%", i);
    }

    InitWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Raytracer");
    init_scene(&scene, 60.0, 1.0);
    camera_set_position(&scene.camera, Vec3(0.0, 0.0, -2.0));
    camera_set_rotation(&scene.camera, Vec3(0.0, 0.0, 0.0));

    Sphere sphere = new_sphere(Vec3(0.0, 0.0, 1.0), 0.5);
    set_sphere_material(&sphere, new_material(Vec3(0.1, 0.2, 0.5), 0.0, 0.0));
    add_object_to_scene(&scene, SPHERE, &sphere);
    
    /* Sphere big_sphere = new_sphere(Vec3(0.0, -100.5, 1.0), 100.0); */
    /* set_sphere_material(&big_sphere, new_material(Vec3(0.5, 0.5, 0.8), 0.0)); */
    /* add_object_to_scene(&scene, SPHERE, &big_sphere); */

    Sphere left_sphere = new_sphere(Vec3(-1.1, 0.0, 1.0), 0.5);
    set_sphere_material(&left_sphere, new_material(Vec3(1.0, 1.0, 1.0), 0.0, 1.0 / 1.5));
    add_object_to_scene(&scene, SPHERE, &left_sphere);
    
    Sphere right_sphere = new_sphere(Vec3(1.1, 0.0, 1.0), 0.5);
    set_sphere_material(&right_sphere, new_material(Vec3(0.8, 0.6, 0.2), 1.0, 0.0));
    add_object_to_scene(&scene, SPHERE, &right_sphere);

    Plane ground = new_plane(Vec3(0.0, 0.5, 0.0), Vec3(10.0, 10.0, 0.0));
    plane_set_material(&ground, new_material(Vec3(0.6, 1.0, 0.6), 0.0, 0.0));
    plane_set_rotation(&ground, Vec3(90.0, 0.0, 0.0));
    add_object_to_scene(&scene, PLANE, &ground);

    SetTargetFPS(60);
    SetExitKey(KEY_Q);

    while (!WindowShouldClose()) {
        switch (state) {
            case STATE_START_RENDER:
                render_next_scene();
                state = STATE_RENDERING;
                break;
            case STATE_RENDERING:
                if (atomic_load_explicit(&scene.rendered, memory_order_relaxed)) {
                    int error = pthread_join(thread_ID, NULL);
                    if (error != 0) {
                        TraceLog(LOG_ERROR, "Error joining rendering thread");
                        exit(1);
                    }
                    state = STATE_RENDERED;
                }
            break;
            case STATE_RENDERED:
                if (IsKeyPressed(KEY_A)) {
                    scene.camera.position.x -= 1.0;
                } else if (IsKeyPressed(KEY_D)) {
                    scene.camera.position.x += 1.0;
                } else if (IsKeyPressed(KEY_W)) {
                    scene.camera.position.z += 1.0;
                } else if (IsKeyPressed(KEY_S)) {
                    scene.camera.position.z -= 1.0;
                }
                else if (IsKeyPressed(KEY_LEFT)) {
                    scene.camera.rotation.y += 5.0;
                } else if (IsKeyPressed(KEY_RIGHT)) {
                    scene.camera.rotation.y -= 5.0;
                } else if (IsKeyPressed(KEY_UP)) {
                    scene.camera.rotation.x -= 5.0;
                } else if (IsKeyPressed(KEY_DOWN)) {
                    scene.camera.rotation.x += 5.0;
                }
                else if (IsKeyPressed(KEY_ENTER)) {
                    println("Camera pos: {3d:, }", ExpandVec3(scene.camera.position));
                    println("Camera rot: {3d:, }", ExpandVec3(scene.camera.rotation));
                    state = STATE_START_RENDER;
                }
            break;
        }

        BeginDrawing();
        
        switch (state) {
            case STATE_START_RENDER: break;
            case STATE_RENDERING: {
                ClearBackground(BLACK);
                int progress = atomic_load_explicit(&scene.rendering_progress, memory_order_relaxed);
                const int x = (IMAGE_WIDTH - MeasureText(progress_cache[progress], TEXT_SIZE)) / 2;
                DrawText(progress_cache[progress], x, IMAGE_HEIGHT / 2, TEXT_SIZE, RAYWHITE);
            } break;
            case STATE_RENDERED:
                draw_scene();
                break;
        }

        EndDrawing();
    }

    CloseWindow();

}

void render_next_scene() {
    atomic_store_explicit(&scene.rendering_progress, 0, memory_order_relaxed);
    atomic_store_explicit(&scene.rendered, 0, memory_order_relaxed);
    int error = pthread_create(&thread_ID, NULL, (void *(*)(void *)) (&scene_render), &scene);

    if (error != 0) {
        TraceLog(LOG_ERROR, "Error creating render thread");
        exit(1);
    }

    state = STATE_RENDERING;
}

void draw_scene() {
    Vector3 pixel;
    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            pixel = scene.image[y][x];
            DrawPixel(x, y, (Color) {(unsigned int) pixel.x, (unsigned int) pixel.y, (unsigned int) pixel.z, (unsigned int) 255});
        }
    }
}
