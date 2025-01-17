#include "gpu.h"

#include "stdio.h"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        printf("CUDA error = %d at %s:%d '%s'", static_cast<unsigned int>(result), file, line, func);
        printf("Reason: %s\n", cudaGetErrorString(result));
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
