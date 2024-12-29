#include "common.h"

typedef struct color {
    unsigned char R, G, B;
} Color;

void save_buf_as_ppm_format(Color image[IMAGE_HEIGHT][IMAGE_WIDTH]);
