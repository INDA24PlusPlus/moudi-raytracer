cmake_minimum_required(VERSION 3.28)

project(RayTracer VERSION 1.0.0 LANGUAGES C CUDA)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_DEBUG "-pg -g -lm")
set(CMAKE_C_FLAGS_RELEASE "-Wall -lm")

add_subdirectory("submodules/String-Formatter")
add_subdirectory("raytracer-lib")

get_target_property(LIB_INCLUDE_DIRS raytracer-lib INCLUDE_DIRECTORIES)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/**.cu)

add_executable(raytracer ${SOURCES})

set_property(TARGET raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Includes: ${LIB_INCLUDE_DIRS}")

target_include_directories(raytracer PRIVATE ${LIB_INCLUDE_DIRS})
target_link_libraries(raytracer PRIVATE fmt raylib raytracer-lib)
