cmake_minimum_required(VERSION 3.28)

project(RayTracer-lib VERSION 1.0.0 LANGUAGES CUDA)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE INCLUDES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/includes/**.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/**.cu)

message(STATUS "CUDA: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_library(raytracer-lib STATIC ${SOURCES} ${INCLUDES})

set_target_properties(raytracer-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(raytracer-lib PRIVATE 
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes/objects
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes/utils
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
