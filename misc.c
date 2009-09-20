#include <sys/time.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "misc.h"


void warning (const char *msg, ...)
{
        va_list args;

        fputs("WARNING: ", stderr);
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        va_end(args);
        fputs(".\n", stderr);
}


void error (const char *msg, ...)
{
        va_list args;
        int errnum;
        char errmsg[72];

        errnum = errno;
        fputs("ERROR: ", stderr);
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        va_end(args);
        if (errnum != 0)
        {
                strerror_r(errnum, errmsg, 71);
                fprintf(stderr, ": %s.\n", errmsg);
        }
        else
                fputs(".\n", stderr);
}


void fatal (const char *msg, ...)
{
        va_list args;
        int errnum;
        char errmsg[72];

        errnum = errno;
        fputs("FATAL ERROR: ", stderr);
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        va_end(args);
        if (errnum != 0)
        {
                strerror_r(errnum, errmsg, 71);
                fprintf(stderr, ": %s.\n", errmsg);
        }
        else
                fputs(".\n", stderr);

        exit(EXIT_FAILURE);
}


float system_time (void)
{
        struct timeval t;

        gettimeofday(&t, NULL);
        return t.tv_sec * 1.0e6f + t.tv_usec;
}


void *do_malloc (size_t bytes, const char *file, int line)
{
        void *addr;

        addr = malloc(bytes + ALIGN_BOUNDARY);
        if (addr == NULL)
                fatal("Unable to allocate %d bytes at %s line %d",
                      bytes, file, line);
        memset(addr, 0, bytes + ALIGN_BOUNDARY);
        return (void *) (((unsigned int) addr + LOWER_MASK) & UPPER_MASK);
}

