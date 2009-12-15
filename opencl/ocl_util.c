#include "ocl_util.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
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


cl_device_id ocl_available_device (cl_context ctx)
{
        cl_device_id *device, ret;
        cl_int e, id;
        cl_bool available;
        size_t sz;
        int dev_count;

        e = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz);
        if (e != CL_SUCCESS)
                ocl_error(e, "Listing context devices");
        device = xmalloc(sz);
        e = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sz, device, NULL);
        if (e != CL_SUCCESS)
        {
                free(device);
                ocl_error(e, "Listing context devices");
                return NULL;
        }

        dev_count = sz / sizeof(cl_device_id);
        ret = NULL;
        for (id = 0;  id < dev_count;  id++)
        {

                e = clGetDeviceInfo(device[id], CL_DEVICE_AVAILABLE,
                                    sizeof(cl_bool), &available, NULL);
                if (e != CL_SUCCESS)
                        ocl_error(e, "Querying device %d", id);
                else
                        if (available == CL_TRUE)
                        {
                                ret = device[id];
                                break;
                        }
        }

        free(device);
        return ret;
}


cl_program ocl_make_program (cl_context ctx, cl_device_id dev, const char *file)
{
        struct stat st;
        char *data;
        size_t filesize;
        int e, fd;
        cl_program prog;
        cl_int cle;

        fd = open(file, O_RDONLY);
        if (fd == -1)
                fatal("Could not open '%s'", file);
        e = fstat(fd, &st);
        if (e == -1)
                fatal("Could not stat '%s'", file);
        data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
                fatal("Could not map '%s'", file);

        filesize = (size_t) st.st_size;
        prog = clCreateProgramWithSource(ctx, 1, (const char **) &data,
                                         &filesize, &cle);
        if (cle != CL_SUCCESS)
                ocl_fatal(cle, "Could not create program");
        cle = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
        if (cle != CL_SUCCESS)
                ocl_fatal(cle, "Could not build program");
        clUnloadCompiler();

        e = munmap(data, st.st_size);
        if (e == -1)
                error("Unmpaaing file '%s'", file);
        e = close(fd);
        if (e == -1)
                error("Closing file '%s'", file);

        return prog;
}

