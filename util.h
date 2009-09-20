#ifndef __nnvect_misc
#define __nnvect_misc

#include <sys/time.h>

void warning (const char *msg, ...);
void error   (const char *msg, ...);
void fatal   (const char *msg, ...);

#define ALIGN_BOUNDARY  16U
#define LOWER_MASK      (ALIGN_BOUNDARY - 1U)
#define UPPER_MASK      (~(ALIGN_BOUNDARY - 1U))

#define malloc_aligned(bytes)  do_malloc(bytes, __FILE__, __LINE__)
void *do_malloc (size_t bytes, const char *file, int line);

static inline float elapsed_time (struct timeval *begin, struct timeval *end)
{
        return (end->tv_sec - begin->tv_sec) + (end->tv_usec - begin->tv_usec)
               * 1.0e-6f;
}

#endif /* __nnvect_misc */
