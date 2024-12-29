#include "ppm.h"
#include "fmt.h"

#include <stdio.h>


void save_buf_as_ppm_format(Color image[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    FILE * fp = fopen("out.ppm", "w");

    writef(fp, "P3\n{i} {i}\n255", IMAGE_WIDTH, IMAGE_HEIGHT);

    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        fputc('\n', fp);
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            Color pixel = image[y][x];
            if (x == 0) {
                writef(fp, "{i} {i} {i}", pixel.R, pixel.G, pixel.B);
            } else {
                writef(fp, " {i} {i} {i}", pixel.R, pixel.G, pixel.B);
            }
        }
    }

    fclose(fp);
}
