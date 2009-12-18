#ifndef __nnocl_gpu
#define __nnocl_gpu

#include <CL/cl.h>

#define gpu_check(val) if ((val) != CL_SUCCESS) \
                               gpu_fatal(__FILE__, __LINE__, val)

void gpu_fatal (const char *, int, cl_int);

#endif /* __nnocl_gpu */

