#ifndef __nnvect_misc
#define __nnvect_misc

void warning (const char *msg, ...);
void error   (const char *msg, ...);
void fatal   (const char *msg, ...);

float system_time (void);

#define ALIGN_BOUNDARY  16U
#define LOWER_MASK      (ALIGN_BOUNDARY - 1U)
#define UPPER_MASK      (~(ALIGN_BOUNDARY - 1U))

#define malloc_aligned(bytes)  do_malloc(bytes, __FILE__, __LINE__)
void *do_malloc (size_t bytes, const char *file, int line);

#endif /* __nnvect_misc */
