#include <stdio.h>
#include <stdlib.h>

#include "../../include/cuda/common.cuh"

namespace viewer {

cudaError_t cuda_assert(const cudaError_t code,
                        const char* const file,
                        const int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);

        cudaDeviceReset();
        exit(code);
    }

    return code;
}

}  // namespace viewer
