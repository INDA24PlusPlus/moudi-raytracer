cmake_minimum_required(VERSION 3.23)

project(Raytracer VERSION 1.0.0)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_DEBUG "-pg -g -lm")
set(CMAKE_C_FLAGS_RELEASE "-Wall -lm -O3")

message(STATUS "FLAGS: ${CMAKE_C_FLAGS_DEBUG}")

add_subdirectory("submodules/String-Formatter")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/**.c)

message(STATUS "SOURCES: ${SOURCES}")

add_executable(raytracer ${SOURCES})

# Add other flags to the compiler
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Include header files
target_include_directories(raytracer PRIVATE 
	${CMAKE_CURRENT_SOURCE_DIR}/includes
	${CMAKE_CURRENT_SOURCE_DIR}/includes/objects
	${CMAKE_CURRENT_SOURCE_DIR}/includes/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/submodule/String-Formatter/include)

# Link libraries
target_link_libraries(raytracer PRIVATE fmt raylib)
