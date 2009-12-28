#include "gpu.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "db.h"
#include "util.h"

struct _GPUData
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

static struct _GPUData GPU;


void init_gpu (const char *filename, const char *kernelname)
{
        int fd, e;
        struct stat st;
        size_t filesize, sz;
        char *data, *errors, buildflags[128];
        cl_device_id *devices;
        cl_build_status status;
        cl_int ce;

        memset(&GPU, 0, sizeof(struct _GPUData));

        /* Create context */
        GPU.context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL,
                                              NULL, &ce);  gpu_check(ce);
        /* Get first device in context list */
        ce = clGetContextInfo(GPU.context , CL_CONTEXT_DEVICES, 0, NULL, &sz);  gpu_check(ce);
        devices = xmalloc(sz);
        ce = clGetContextInfo(GPU.context, CL_CONTEXT_DEVICES, sz, devices, NULL);  gpu_check(ce);
        GPU.device = devices[0];
        free(devices);
        /* Create command queue */
        GPU.queue = clCreateCommandQueue(GPU.context, GPU.device, 0, &ce);  gpu_check(ce);

        /* Load program code */
        fd = open(filename, O_RDONLY);
        if (fd == -1)
                fatal("Could not open '%s'", filename);
        e = fstat(fd, &st);
        if (e == -1)
                fatal("Could not stat '%s'", filename);
        data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
                fatal("Could not map '%s'", filename);

        /* Create and compile the program (print errors on failure) */
        filesize = (size_t) st.st_size;
        GPU.program = clCreateProgramWithSource(GPU.context, 1,
                                                (const char **) &data, &filesize,
                                                &ce);  gpu_check(ce);
        /* Ignore errors in clBuildProgram() as they will be detected and
         * reported in more detail later */
        snprintf(buildflags, 128, "-DBLOCKDIM_X=%d -DBLOCKDIM_Y=%d", BLOCKDIM_X,
                 BLOCKDIM_Y);
        clBuildProgram(GPU.program, 0, NULL, buildflags, NULL, NULL);
        do {
                sleep(1);
                ce = clGetProgramBuildInfo(GPU.program, GPU.device,
                                           CL_PROGRAM_BUILD_STATUS,
                                           sizeof(cl_build_status), &status,
                                           NULL);  gpu_check(ce);
        } while (status == CL_BUILD_IN_PROGRESS);
        if (status != CL_BUILD_SUCCESS)
        {
                ce = clGetProgramBuildInfo(GPU.program, GPU.device,
                                           CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);  gpu_check(ce);
                errors = xmalloc(sz);
                ce = clGetProgramBuildInfo(GPU.program, GPU.device,
                                           CL_PROGRAM_BUILD_LOG, sz, errors, NULL);  gpu_check(ce);
                destroy_gpu();
                quit("Errors during '%s' compilation:\n%s", filename, errors);
        }

        /* Create the corresponding kernel */
        GPU.kernel = clCreateKernel(GPU.program, kernelname, &ce);  gpu_check(ce);

        /* Unload the program source code */
        e = munmap(data, st.st_size);
        if (e == -1)
                error("Unmapping file '%s'", filename);
        e = close(fd);
        if (e == -1)
                error("Closing file '%s'", filename);
}


void send_nn_arguments (struct db *trdb, struct db *db)
{
        cl_int e;

        GPU.trdata = clCreateBuffer(GPU.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     trdb->count * trdb->dimensions * trdb->typesize,
                                     trdb->data, &e);  gpu_check(e);
        GPU.data = clCreateBuffer(GPU.context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   db->count * db->dimensions * db->typesize,
                                   db->data, &e);  gpu_check(e);
        GPU.trklasses = clCreateBuffer(GPU.context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        trdb->count * sizeof(int), trdb->klass,
                                        &e);  gpu_check(e);
        GPU.klasses = clCreateBuffer(GPU.context, CL_MEM_WRITE_ONLY,
                                      db->count * sizeof(int), NULL, &e);  gpu_check(e);

        e = clSetKernelArg(GPU.kernel, 0, sizeof(int), &db->dimensions);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 1, sizeof(int), &trdb->count);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 2, sizeof(cl_mem), &GPU.trdata);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 3, sizeof(cl_mem), &GPU.trklasses);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 4, sizeof(int), &db->count);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 5, sizeof(cl_mem), &GPU.data);  gpu_check(e);
        e = clSetKernelArg(GPU.kernel, 6, sizeof(cl_mem), &GPU.klasses);  gpu_check(e);
}


void execute_kernel (void)
{
        size_t local[2], global[2];
        cl_int e;

        local[0] = global[0] = BLOCKDIM_X;
        local[1] = BLOCKDIM_Y;
        global[1] = BLOCKDIM_Y * 30;

        e = clEnqueueNDRangeKernel(GPU.queue, GPU.kernel, 2, NULL, global,
                                   local, 0, NULL, NULL);  gpu_check(e);
        e = clFinish(GPU.queue);  gpu_check(e);
}


void get_nn_result (int size, void *data)
{
        cl_int e;

        e = clEnqueueReadBuffer(GPU.queue, GPU.klasses, CL_TRUE, 0, size,
                                data, 0, NULL, NULL);  gpu_check(e);
}


void destroy_gpu (void)
{
        if (GPU.trdata)
                clReleaseMemObject(GPU.trdata);
        if (GPU.trklasses)
                clReleaseMemObject(GPU.trklasses);
        if (GPU.data)
                clReleaseMemObject(GPU.data);
        if (GPU.klasses)
                clReleaseMemObject(GPU.klasses);
        if (GPU.kernel)
                clReleaseKernel(GPU.kernel);
        if (GPU.program)
                clReleaseProgram(GPU.program);
        if (GPU.queue)
                clReleaseCommandQueue(GPU.queue);
        if (GPU.kernel)
                clReleaseContext(GPU.context);
}


void gpu_fatal (const char *file, int line, cl_int errcode)
{
        char *errmsg;

        switch (errcode)
        {
        case CL_SUCCESS                        : errmsg = "Success";                              break;
        case CL_DEVICE_NOT_FOUND               : errmsg = "Device not found";                     break;
        case CL_DEVICE_NOT_AVAILABLE           : errmsg = "Device not available";                 break;
        case CL_COMPILER_NOT_AVAILABLE         : errmsg = "Compiler not available";               break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE  : errmsg = "Memory object allocation failure";     break;
        case CL_OUT_OF_RESOURCES               : errmsg = "Out of resources";                     break;
        case CL_OUT_OF_HOST_MEMORY             : errmsg = "Out of host memory";                   break;
        case CL_PROFILING_INFO_NOT_AVAILABLE   : errmsg = "Profiling information not available";  break;
        case CL_MEM_COPY_OVERLAP               : errmsg = "Memory copy overlap";                  break;
        case CL_IMAGE_FORMAT_MISMATCH          : errmsg = "Image format mismatch";                break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED     : errmsg = "Image format not supported";           break;
        case CL_BUILD_PROGRAM_FAILURE          : errmsg = "Build program failure";                break;
        case CL_MAP_FAILURE                    : errmsg = "Map failure";                          break;
        case CL_INVALID_VALUE                  : errmsg = "Invalid value";                        break;
        case CL_INVALID_DEVICE_TYPE            : errmsg = "Invalid device type";                  break;
        case CL_INVALID_PLATFORM               : errmsg = "Invalid platform";                     break;
        case CL_INVALID_DEVICE                 : errmsg = "Invalid device";                       break;
        case CL_INVALID_CONTEXT                : errmsg = "Invalid context";                      break;
        case CL_INVALID_QUEUE_PROPERTIES       : errmsg = "Invalid queue properties";             break;
        case CL_INVALID_COMMAND_QUEUE          : errmsg = "Invalid command queue";                break;
        case CL_INVALID_HOST_PTR               : errmsg = "Invalid host pointer";                 break;
        case CL_INVALID_MEM_OBJECT             : errmsg = "Invalid memory object";                break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: errmsg = "Invalid image format descriptor";      break;
        case CL_INVALID_IMAGE_SIZE             : errmsg = "Invalid imgae size";                   break;
        case CL_INVALID_SAMPLER                : errmsg = "Invalid image sampler";                break;
        case CL_INVALID_BINARY                 : errmsg = "Invalid binary";                       break;
        case CL_INVALID_BUILD_OPTIONS          : errmsg = "Invalid build options";                break;
        case CL_INVALID_PROGRAM                : errmsg = "Invalid program";                      break;
        case CL_INVALID_PROGRAM_EXECUTABLE     : errmsg = "Invalid program executable";           break;
        case CL_INVALID_KERNEL_NAME            : errmsg = "Invalid kernel name";                  break;
        case CL_INVALID_KERNEL_DEFINITION      : errmsg = "Invalid kernel definition";            break;
        case CL_INVALID_KERNEL                 : errmsg = "Invalid kernel";                       break;
        case CL_INVALID_ARG_INDEX              : errmsg = "Invalid argument index";               break;
        case CL_INVALID_ARG_VALUE              : errmsg = "Invalid argument value";               break;
        case CL_INVALID_ARG_SIZE               : errmsg = "Invalid argument size";                break;
        case CL_INVALID_KERNEL_ARGS            : errmsg = "Invalid kernel arguments";             break;
        case CL_INVALID_WORK_DIMENSION         : errmsg = "Invalid work dimension";               break;
        case CL_INVALID_WORK_GROUP_SIZE        : errmsg = "Invalid work group size";              break;
        case CL_INVALID_WORK_ITEM_SIZE         : errmsg = "Invalid work item size";               break;
        case CL_INVALID_GLOBAL_OFFSET          : errmsg = "Invalid global offset";                break;
        case CL_INVALID_EVENT_WAIT_LIST        : errmsg = "Invalid event wait list";              break;
        case CL_INVALID_EVENT                  : errmsg = "Invalid event";                        break;
        case CL_INVALID_OPERATION              : errmsg = "Invalid operation";                    break;
        case CL_INVALID_GL_OBJECT              : errmsg = "Invalid GL object";                    break;
        case CL_INVALID_BUFFER_SIZE            : errmsg = "Invalid buffer size";                  break;
        case CL_INVALID_MIP_LEVEL              : errmsg = "Invalid MIP level";                    break;
        default                                : errmsg = "Invalid error code";
        }

        destroy_gpu();
        print_message(file, line, 3, "OpenCL error: %s", errmsg);
}

