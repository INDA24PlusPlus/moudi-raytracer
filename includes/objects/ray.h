#pragma once

#include "common.h"

typedef struct ray {
    Vec3 dir;
    int depth;
    Vec3 acc_color;
} Ray;
