#pragma once

#include <stdlib.h>
#include <raylib.h>
#include <raymath.h>
#include "fmt.h"
#include "math.h"
#include "stddef.h"

#define IMAGE_HEIGHT 720
#define IMAGE_WIDTH 1080

#define Vec3(x, y, z) ((Vector3) {x, y, z})
#define ExpandVec3(vec) (vec).x, (vec).y, (vec).z

#define DEFAULT_VEC Vec3(0.0, 0.0, 1.0)
