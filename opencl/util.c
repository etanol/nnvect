#include "util.h"
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * NOTE: This error string list is implementation dependant (for NVIDIA
 * implementation).  A more correct way to code this is to replace the array
 * with a switch statement inside a function call and use the symbolic error
 * codes.
 */
static const char *OCL_ErrorMsg[] = {
        /*  0 */ "Success",
        /*  1 */ "Device not found",
        /*  2 */ "Device not available",
        /*  3 */ "Compiler not available",
        /*  4 */ "Memory object allocation failure",
        /*  5 */ "Out of resources",
        /*  6 */ "Out of host memory",
        /*  7 */ "Profiling information not available",
        /*  8 */ "Memory copy overlap",
        /*  9 */ "Image format mismatch",
        /* 10 */ "Image format not supported",
        /* 11 */ "Build program failure",
        /* 12 */ "Map failure",

        /* 13 */ "Unknown error",
        /* 14 */ "Unknown error",
        /* 15 */ "Unknown error",
        /* 16 */ "Unknown error",
        /* 17 */ "Unknown error",
        /* 18 */ "Unknown error",
        /* 19 */ "Unknown error",
        /* 20 */ "Unknown error",
        /* 21 */ "Unknown error",
        /* 22 */ "Unknown error",
        /* 23 */ "Unknown error",
        /* 24 */ "Unknown error",
        /* 25 */ "Unknown error",
        /* 26 */ "Unknown error",
        /* 27 */ "Unknown error",
        /* 28 */ "Unknown error",
        /* 29 */ "Unknown error",

        /* 30 */ "Invalid value",
        /* 31 */ "Invalid device type",
        /* 32 */ "Invalid platform",
        /* 33 */ "Invalid device",
        /* 34 */ "Invalid context",
        /* 35 */ "Invalid queue properties",
        /* 36 */ "Invalid command queue",
        /* 37 */ "Invalid host pointer",
        /* 38 */ "Invalid memory object",
        /* 39 */ "Invalid image format descriptor",
        /* 40 */ "Invalid imgae size",
        /* 41 */ "Invalid image sampler",
        /* 42 */ "Invalid binary",
        /* 43 */ "Invalid build options",
        /* 44 */ "Invalid program",
        /* 45 */ "Invalid program executable",
        /* 46 */ "Invalid kernel name",
        /* 47 */ "Invalid kernel definition",
        /* 48 */ "Invalid kernel",
        /* 49 */ "Invalid argument index",
        /* 50 */ "Invalid argument value",
        /* 51 */ "Invalid argument size",
        /* 52 */ "Invalid kernel arguments",
        /* 53 */ "Invalid work dimension",
        /* 54 */ "Invalid work group size",
        /* 55 */ "Invalid work item size",
        /* 56 */ "Invalid global offset",
        /* 57 */ "Invalid event wait list",
        /* 58 */ "Invalid event",
        /* 59 */ "Invalid operation",
        /* 60 */ "Invalid GL object",
        /* 61 */ "Invalid buffer size",
        /* 62 */ "Invalid MIP level",
        /* 63 */ "Invalid error code"
};


void print_message (const char *file, int line, int severity,
                    const char *message, ...)
{
        va_list args;
        char *label;
        char errmsg[72];

        errmsg[0] = '\0';
        switch (severity)
        {
        case 0: label = "DEBUG";  break;
        case 1: label = "WARNING";  break;
        case 3: label = "ABORT";  break;
        case 2:
                label = "ERROR";
                strerror_r(errno, errmsg, 71);
                break;
        case 4:
                label = "FATAL";
                strerror_r(errno, errmsg, 71);
                break;
        default:
                label = "???";
        }

        fprintf(stderr, "[%s at %s:%d] ", label, file, line);
        va_start(args, message);
        vfprintf(stderr, message, args);
        va_end(args);

        if (errno != 0 && errmsg[0] != '\0')
                fprintf(stderr, ": %s", errmsg);
        fputs(".\n", stderr);

        if (severity > 2)
                exit(EXIT_FAILURE);
}


void ocl_print_error (const char *file, int line, cl_int code, int is_fatal,
                      const char *message, ...)
{
        va_list args;
        char *label;
        int errcode;

        errcode = (code > 0 ? 63 : (int) -code);
        label = (is_fatal ? "FATAL" : "ERROR");

        fprintf(stderr, "[OpenCL %s at %s:%d] ", label, file, line);
        va_start(args, message);
        vfprintf(stderr, message, args);
        va_end(args);
        fprintf(stderr, ": %s.\n", OCL_ErrorMsg[code]);

        if (is_fatal)
                exit(EXIT_FAILURE);
}


void *allocate_memory (const char *file, int line, size_t bytes)
{
        void *mem;

        mem = malloc(bytes);
        if (mem == NULL)
                print_message(file, line, 4, "Unable to allocate %ld bytes",
                              bytes);
        return mem;
}

