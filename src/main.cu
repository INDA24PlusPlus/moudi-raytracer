#include "camera.h"
#include "common.h"
#include "material.h"
#include "object.h"
#include "plane.h"
#include "scene.h"
#include "sphere.h"
#include "stddef.h"
#include <cstdio>
#include <pthread.h>
#include <raylib.h>
#include <stdlib.h>

#include "gpu.h"
#define RANDOM_RANGE(min, max) (((float) rand() / INT_MAX) * (max - min) + min)

Scene scene, * device_scene;

void ** host_objects, ** device_objects;

char * progress_cache[101];
#define TEXT_SIZE 50

pthread_t thread_ID = {0};
enum { STATE_START_RENDER, STATE_RENDERING, STATE_RENDERED } state = STATE_START_RENDER;

void render_next_scene();
void draw_scene();

int main() {
    srand(time(NULL));

    for (size_t i = 0; i <= 100; ++i) {
        asprintf(&progress_cache[i], "Waiting for render: %d", i);
    }

    InitWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Raytracer");

    cudaMalloc(&device_scene, sizeof(Scene));
    init_scene(&scene, 60.0, 1.0);

    camera_set_position(&scene.camera, Vec3(0.0, 0.0, -3.0));
    camera_set_rotation(&scene.camera, Vec3(10.0, 0.0, 0.0));

    Sphere sphere = new_sphere(Vec3(0.0, 0.0, 1.0), 0.5);
    set_sphere_material(&sphere, new_material(Vec3(0.1, 0.2, 0.5), 0.0, 0.0, 0.0));
    add_object_to_scene(&scene, SPHERE, &sphere);
    
    // Sphere big_sphere = new_sphere(Vec3(0.0, -100.5, 1.0), 100.0);
    // set_sphere_material(&big_sphere, new_material(Vec3(0.5, 0.5, 0.8), 0.0, 0.0, 0.0));
    // add_object_to_scene(&scene, SPHERE, &big_sphere);

    Sphere left_sphere = new_sphere(Vec3(-1.1, 0.0, 1.0), 0.5);
    set_sphere_material(&left_sphere, new_material(Vec3(1.0, 1.0, 1.0), 0.0, 0.0, 1.0f / 1.3f));
    add_object_to_scene(&scene, SPHERE, &left_sphere);

    Sphere right_sphere = new_sphere(Vec3(1.1, 0.0, 1.0), 0.5);
    set_sphere_material(&right_sphere, new_material(Vec3(0.8, 0.6, 0.2), 0.0, 1.0, 0.0));
    add_object_to_scene(&scene, SPHERE, &right_sphere);

    Plane ground = new_plane(Vec3(0.0, -0.5, 0.0), Vec3(25.0, 25.0, 0.0));
    plane_set_material(&ground, new_material(Vec3(0.6, 1.0, 0.6), 0.0, 0.0, 0.0));
    plane_set_rotation(&ground, Vec3(90.0, 0.0, 0.0));
    add_object_to_scene(&scene, PLANE, &ground);

    // for (int a = -5; a < 5; a++) {
    //    for (int b = -5; b < 5; b++) {
    //        float choose_mat = RANDOM_RANGE(0.0f, 1.0f);
    //        printf("mat = %f\n", choose_mat);
    //        float3 center = { .x = a + 0.9f * RANDOM_RANGE(0.0f, 1.0f), .y = 0.2f, .z = b + 0.9f * RANDOM_RANGE(0.0f, 1.0f) };
    //        float3 diff = vec_sub(center, (float3) {4.0, 0.2, 0});

    //        if (vec_length(diff) > 0.9f) {
    //            Material_t material;

    //            if (choose_mat < 0.8f) { // diffuse
    //                float3 color = (float3) {RANDOM_RANGE(0.0f, 1.0f), RANDOM_RANGE(0.0f, 1.0f), RANDOM_RANGE(0.0f, 1.0f)};
    //                material = new_material(color, 0.0f, 0.0, 0.0f);
    //            } else if (choose_mat < 0.95) { // metal
    //                float3 color = {RANDOM_RANGE(0.5f, 1.0f), RANDOM_RANGE(0.5f, 1.0f), RANDOM_RANGE(0.5f, 1.0f)};
    //                float fuzz = RANDOM_RANGE(0, 0.5);
    //                material = new_material(color, fuzz, 1.0f, 0.0f);
    //            } else { // glass
    //                material = new_material(VEC_ONE, 0.0, 0.0, 1.5);
    //            }

    //            Sphere sphere = new_sphere(center, 0.2);
    //            set_sphere_material(&sphere, material);
    //            add_object_to_scene(&scene, SPHERE, &sphere);
    //        }
    //    }
    //}

    host_objects = scene.objects.items;
    device_objects = obj_list_to_device(scene.objects);
    
    SetTargetFPS(60);
    SetExitKey(KEY_Q);

    while (!WindowShouldClose()) {
        switch (state) {
            case STATE_START_RENDER:
                scene_render(&scene, device_scene, host_objects, device_objects);
                state = STATE_RENDERING;
                break;
            case STATE_RENDERING:
                if (scene.rendered) {
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
                    printf("Camera pos: %f, %f, %f\n", ExpandVec3(scene.camera.position));
                    printf("Camera rot: %f, %f, %f\n", ExpandVec3(scene.camera.rotation));
                    state = STATE_START_RENDER;
                }
            break;
        }

        BeginDrawing();
        
        switch (state) {
            case STATE_START_RENDER: break;
            case STATE_RENDERING: {
                ClearBackground(BLACK);

                int progress;
                cudaMemcpy(&progress, &device_scene->rendering_progress, sizeof(progress), cudaMemcpyDeviceToHost);
                progress = (100 * progress) / (IMAGE_WIDTH * IMAGE_HEIGHT);
                printf("progress = %d\n", progress);

                const int x = (IMAGE_WIDTH - MeasureText(progress_cache[progress], TEXT_SIZE)) / 2;
                DrawText(progress_cache[progress], x, IMAGE_HEIGHT / 2, TEXT_SIZE, RAYWHITE);

                if (progress == 100) {
                    checkCudaErrors(cudaMemcpy(&scene.image, &device_scene->image, sizeof(scene.image), cudaMemcpyDeviceToHost));
                    state = STATE_RENDERED;
                }
            } break;
            case STATE_RENDERED:
                draw_scene();
                break;
        }

        EndDrawing();
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
 
    cudaDeviceReset();
    CloseWindow();

}

void draw_scene() {
    float3 pixel;
    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            pixel = scene.image[y][x];
            DrawPixel(x, y, (Color) {(unsigned int) pixel.x, (unsigned int) pixel.y, (unsigned int) pixel.z, (unsigned int) 255});
        }
    }
}
