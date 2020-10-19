#include "main.h"
#include "log.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
    if (!graphics_init())
        return -1;

    gl_setup();
    main_loop();

    /*  CUDA_CHECK(cudaSetDevice(0));

      uint8_t* chunkDataCPU;
      uint8_t* chunkDataGPU;

      chunkDataCPU = (uint8_t*)malloc(CHUNK_VOXELS);
      memset(chunkDataCPU, 0, CHUNK_VOXELS);

      CUDA_CHECK(cudaMalloc((void**)&chunkDataGPU, CHUNK_VOXELS));
      CUDA_CHECK(cudaMemcpy(chunkDataGPU, chunkDataCPU, CHUNK_VOXELS, cudaMemcpyKind::cudaMemcpyHostToDevice));

      generateChunkKernel << <CHUNK_SIZE * CHUNK_SIZE, CHUNK_SIZE >> > (chunkDataGPU);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(chunkDataCPU, chunkDataGPU, CHUNK_VOXELS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(chunkDataGPU));

      free(chunkDataCPU);*/

    CUDA_CHECK(cudaDeviceReset());
    graphics_cleanup();

    return 0;
}
