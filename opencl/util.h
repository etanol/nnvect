#ifndef __nnocl_util
#define __nnocl_util

#include <sys/types.h>
#include <sys/time.h>
#include <CL/cl.h>


/* Module interface */
#define warning(...)        print_message(__FILE__, __LINE__, 1, __VA_ARGS__)
#define error(...)          print_message(__FILE__, __LINE__, 2, __VA_ARGS__)
#define quit(...)           print_message(__FILE__, __LINE__, 3, __VA_ARGS__)
#define fatal(...)          print_message(__FILE__, __LINE__, 4, __VA_ARGS__)

#define xmalloc(s)     allocate_memory(__FILE__, __LINE__, (s))
#define xstrcat(a, b)  string_concat(__FILE__, __LINE__, (a), (b))

#define ocl_check(err) \
                if ((err) != CL_SUCCESS) \
                        ocl_print_error(__FILE__, __LINE__, (err), "ERROR")

#define ocl_assert(err) do { \
                if ((err) != CL_SUCCESS) \
                { \
                        ocl_print_error(__FILE__, __LINE__, (err), "FATAL"); \
                        exit(EXIT_FAILURE); \
                } \
        } while (0)

/* Do not call these functions directly, use the macro interface instead */
void  print_message   (const char *, int, int, const char *, ...);
void *allocate_memory (const char *, int, size_t);
char *string_concat   (const char *, int, const char *, const char *);
void  ocl_print_error (const char *, int, cl_int, const char *);

#endif /* __nnocl_util */