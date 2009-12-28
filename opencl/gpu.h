#ifndef __nnocl_gpu
#define __nnocl_gpu

#include <CL/cl.h>

#define BLOCKDIM_X  16
#define BLOCKDIM_Y  32
struct db;

#define gpu_check(val) if ((val) != CL_SUCCESS) \
                               gpu_fatal(__FILE__, __LINE__, val)

void  gpu_fatal         (const char *, int, cl_int);
void  init_gpu          (const char *, const char *);
void  send_nn_arguments (struct db *, struct db *);
void  execute_kernel    (void);
void  get_nn_result     (int, void *);
void  destroy_gpu       (void);

#endif /* __nnocl_gpu */

