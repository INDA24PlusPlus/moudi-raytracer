cmake_minimum_required(VERSION 3.23)

project(Raytracer LANGUAGES C CUDA VERSION 1.0.0)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_DEBUG "-pg -g -lm")
set(CMAKE_C_FLAGS_RELEASE "-Wall -lm")

message(STATUS "FLAGS: ${CMAKE_C_FLAGS_DEBUG}")

add_subdirectory("submodules/String-Formatter")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/**.c)
file(GLOB_RECURSE PARTICLES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/**.cu)

message(STATUS "SOURCES: ${SOURCES}")
message(STATUS "PARTICLES: ${PARTICLES}")

add_executable(raytracer ${SOURCES} ${PARTICLES})

# Add other flags to the compiler
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Include header files
target_include_directories(raytracer PRIVATE 
	${CMAKE_CURRENT_SOURCE_DIR}/includes
	${CMAKE_CURRENT_SOURCE_DIR}/includes/objects
	${CMAKE_CURRENT_SOURCE_DIR}/includes/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/submodule/String-Formatter/include)

set_target_properties(raytracer
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(raytracer PUBLIC cxx_std_11)
  
# Link libraries
target_link_libraries(raytracer PRIVATE fmt raylib)
