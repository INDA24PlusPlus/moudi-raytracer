#pragma once

#include <stdlib.h>
#include <stddef.h>

#include "vec.h"
#include <math.h>

#define IMAGE_HEIGHT 450
#define IMAGE_WIDTH 800
#define SAMPLES_PER_PIXEL 450
#define MAX_RAY_COLLISIONS 50
#define T_MIN_CUTOFF 0.001

#define COMPOSE(f, g) f(g)
#define DEGREE_TO_RADIAN(d) (M_PI * d / 180.0)
#define LINEAR_TO_GAMMA(d) (255.0 * ((d > 0.0) ? (sqrtf(d)) : (0)))
