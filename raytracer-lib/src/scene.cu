#include "scene.h"

#include "common.h"
#include "camera.h"
#include "list.h"
#include "object.h"
#include "plane.h"
#include "ray.h"
#include "sphere.h"
#include "vec.h"

#include "gpu.h"

__host__ __device__ void init_scene(Scene * scene, float FOV, float focal_length) {
    scene->camera = new_camera(FOV, focal_length);
    scene->objects = new_list(sizeof(Object *));
    camera_update(&scene->camera);

    printf("Camera: viewport [%f, %f]\n", scene->camera.viewport.x, scene->camera.viewport.y);
}

void add_object_to_scene(Scene * scene, enum obj_type type, void * ptr) {
    Object object = { .type = type };

    switch (type) {
        case PLANE:
            object.value.plane = *(Plane *) ptr; break;
        case SPHERE:
            object.value.sphere = *(Sphere *) ptr; break;
    }

    Object * obj_ptr = (Object *) malloc(sizeof(object));
    *obj_ptr = object;

    list_push(&scene->objects, obj_ptr);
}

void ** obj_list_to_device(List list) {
    size_t objects_size = list.count * list.item_size;
    void ** objects = (void **) malloc(objects_size), ** device_objects;
    checkCudaErrors(cudaMalloc(&device_objects, objects_size));

    for (size_t i = 0; i < list.count; ++i) {
        Object * obj = (Object *) list.items[i], * device_obj;
        checkCudaErrors(cudaMalloc(&device_obj, sizeof(*device_obj)));
        checkCudaErrors(cudaMemcpy(device_obj, obj, sizeof(*device_obj), cudaMemcpyHostToDevice));
        objects[i] = device_obj;
    }

    checkCudaErrors(cudaMemcpy(device_objects, objects, objects_size, cudaMemcpyHostToDevice));
    return device_objects;
}

__device__ Hit_record scene_find_ray_intersection(Scene * scene, Ray_t ray) {
    Hit_record rec = { .t = INTERSECTION_NOT_FOUND };
    Object collision_obj;

    for (size_t i = 0; i < scene->objects.count; ++i) {
        Object obj = *(Object *) list_at(scene->objects, i);
        float t_temp = obj_intersects_ray(obj, ray);

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
    rec.normal = vec_norm(obj_get_normal(collision_obj, rec.point));
    rec.material = obj_get_material(collision_obj);

    if (vec_dot(ray.direction, rec.normal) > 0.0) {
        rec.normal = vec_negate(rec.normal);
        rec.front_face = 0;
    } else {
        rec.front_face = 1;
    }

    return rec;
}

__global__ void calculate_pixel(Scene * scene) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // printf("%d, %d\n", x, y);

    if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
        return;
    }

    float3 pixel_color = VEC_ZERO;
    for (size_t sample = 0; sample < SAMPLES_PER_PIXEL; ++sample) {
        Ray_t ray = get_ray(&scene->camera, x, y);
        pixel_color = vec_add(pixel_color, ray_color(scene, ray));
    }
    
    pixel_color = vec_scale(pixel_color, 1.0 / SAMPLES_PER_PIXEL);
    scene->image[y][x] = (float3) {ExpandVec3F(pixel_color, LINEAR_TO_GAMMA)};
    atomicAdd(&scene->rendering_progress, 1);
}

void scene_render(Scene * scene, Scene * device_scene, void ** host_objects, void ** device_objects) {
    camera_update(&scene->camera);
    scene->rendering_progress = false;
    scene->rendered = 0;
    scene->objects.items = device_objects;
    checkCudaErrors(cudaMemcpy(device_scene, scene, sizeof(*device_scene), cudaMemcpyHostToDevice));
    scene->objects.items = host_objects;
 
    printf("Scene: Objects [%d]\n", scene->objects.count);

    int thread_x = 16;
    int thread_y = 16;

    dim3 blocks((IMAGE_WIDTH / thread_x) + 1, (IMAGE_HEIGHT / thread_y) + 1);
    dim3 threads(thread_x, thread_y);
    calculate_pixel<<<blocks, threads>>>(device_scene);
}
