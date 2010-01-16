#ifndef __nnvect_util
#define __nnvect_util

#include <sys/types.h>
#include <sys/time.h>

/* Alignment utilities */
#ifndef ALIGNMENT
#define ALIGNMENT     16
#endif
#define PADDED(size)  (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Module interface */
#define warning(...)        print_message(__FILE__, __LINE__, 1, __VA_ARGS__)
#define error(...)          print_message(__FILE__, __LINE__, 2, __VA_ARGS__)
#define quit(...)           print_message(__FILE__, __LINE__, 3, __VA_ARGS__)
#define fatal(...)          print_message(__FILE__, __LINE__, 4, __VA_ARGS__)
#define xmalloc(s)          allocate_memory(__FILE__, __LINE__, 0, (s))
#define xmalloc_aligned(s)  allocate_memory(__FILE__, __LINE__, ALIGNMENT, (s))
#define xstrcat(a, b)       string_concat(__FILE__, __LINE__, (a), (b))

/* Debugging support */
#ifdef DEBUG
#  define debug(...)    print_message(__FILE__, __LINE__, 0, __VA_ARGS__)
#  define ensure(cond)  if (!(cond))  \
                                print_message(__FILE__, __LINE__, 3,  \
                                              "Condition '" #cond "' is false")
#else
#  define debug(...)
#  define ensure(cond)
#endif


/* Functions doing real work, but not recommended to call directly */
void  print_message   (const char *, int, int, const char *, ...);
void *allocate_memory (const char *, int, size_t, size_t);
char *string_concat   (const char *, int, const char *, const char *);

#endif /* __nnvect_util */
