#include "main.h"
#include "log.h"
#include "generator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
    cuda_generate_init();

    if (!graphics_init())
        return -1;

    gl_setup();
    main_loop();

    cuda_generate_clean();
    CUDA_CHECK(cudaDeviceReset());
    graphics_cleanup();

    return 0;
}
