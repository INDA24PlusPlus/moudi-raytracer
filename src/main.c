#include "camera.h"
#include "common.h"
#include "plane.h"
#include "scene.h"
#include "sphere.h"
#include "stddef.h"
#include <pthread.h>
#include <raylib.h>
#include <stdatomic.h>

Scene scene;
char * progress_cache[101];
#define TEXT_SIZE 50

pthread_t thread_ID = {0};
enum { STATE_RENDERING, STATE_RENDERED } state = STATE_RENDERING; 

void render_next_scene();
void draw_scene();

int main() {
    for (size_t i = 0; i <= 100; ++i) {
        progress_cache[i] = format("Waiting for render: {i}%", i);
    }
    InitWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Raytracer");
    init_scene(&scene, 90, 1.0);

    Sphere sphere = new_sphere(Vec3(0.0, 0.0, 10.0), 1.0);
    set_sphere_color(&sphere, Vec3(255.0, 0.0, 0.0));
    add_object_to_scene(&scene, SPHERE, &sphere);

    Plane ground = new_plane(Vec3(0.0, 0.0, 10.0), Vec3(1.0, 1.0, 0.0));
    plane_set_rotation(&ground, Vec3(0.0, 85.0, 0.0));
    plane_set_color(&ground, Vec3(160.0, 255.0, 160.0));
    /* add_object_to_scene(scene, PLANE, &ground); */

    Plane wall = new_plane(Vec3(0.0, 0.0, 5.0), Vec3(10.0, 10.0, 0.0));
    plane_set_rotation(&wall, Vec3(0.0, 0.0, -90.0));
    plane_set_color(&wall, Vec3(255.0, 0.0, 0.0));
    /* add_object_to_scene(&scene, PLANE, &wall); */

    SetTargetFPS(60);

    render_next_scene();

    while (!WindowShouldClose()) {
        switch (state) {
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
            case STATE_RENDERED: break;
        }

        BeginDrawing();
        
        switch (state) {
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
