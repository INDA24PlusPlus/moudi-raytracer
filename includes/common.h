#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <raylib.h>
#include <raymath.h>

#include "fmt.h"
#include "vec.h"

#include "math.h"

#define IMAGE_HEIGHT 450
#define IMAGE_WIDTH 800
#define SAMPLES_PER_PIXEL 100
#define MAX_RAY_COLLISIONS 20
#define T_MIN_CUTOFF 0.001

#define COMPOSE(f, g) f(g)
#define DEGREE_TO_RADIAN(d) (M_PI * d / 180.0)
#define LINEAR_TO_GAMMA(d) (255.0 * ((d > 0.0) ? (sqrtf(d)) : (0)))

#define RANDOM_DOUBLE() (rand() / (double) RAND_MAX)
#define RANDOM_RANGE(min, max) (min + (RANDOM_DOUBLE() * (max - min)))
