#pragma once

#define RANDOM_RANGE(min, max) (rand_float() * (max - min) + min)

__device__ float rand_float();
