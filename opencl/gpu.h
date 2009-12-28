#ifndef __nnocl_gpu
#define __nnocl_gpu

#include <CL/cl.h>

#define BLOCKDIM_X  16
#define BLOCKDIM_Y  32
struct db;

struct gpu
{
        cl_context context;
        cl_device_id device;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        cl_mem trdata;
        cl_mem trklasses;
        cl_mem data;
        cl_mem klasses;
};

#define gpu_check(val) if ((val) != CL_SUCCESS) \
                               gpu_fatal(__FILE__, __LINE__, val)

void        gpu_fatal         (const char *, int, cl_int);
struct gpu *create_gpu        (const char *, const char *);
void        send_nn_arguments (struct gpu *, struct db *, struct db *);
void        execute_kernel    (struct gpu *);
void        get_nn_result     (struct gpu *, int, void *);
void        destroy_gpu       (struct gpu *);

#endif /* __nnocl_gpu */

