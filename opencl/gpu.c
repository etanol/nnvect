#include "gpu.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include "db.h"
#include "util.h"


struct gpu *create_gpu (const char *filename, const char *kernelname)
{
        int fd, e;
        struct stat st;
        size_t filesize, sz;
        char *data, *errors;
        cl_context ctx;
        cl_device_id *devices, dev;
        cl_command_queue queue;
        cl_program prog;
        cl_kernel kernel;
        cl_build_status status;
        cl_int ce;
        struct gpu *gpu;

        /* Create context */
        ctx = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL,
                                               NULL, &ce);  gpu_check(ce);
        /* Get first device in context list */
        ce = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz);  gpu_check(ce);
        devices = xmalloc(sz);
        ce = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sz, devices, NULL);  gpu_check(ce);
        dev = devices[0];
        free(devices);
        /* Create command queue */
        queue = clCreateCommandQueue(ctx, dev, 0, &ce);  gpu_check(ce);

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
        prog = clCreateProgramWithSource(ctx, 1, (const char **) &data,
                                         &filesize, &ce);  gpu_check(ce);
        /* Ignore errors in clBuildProgram() as they will be detected and
         * reported in more detail later */
        clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
        do {
                sleep(1);
                ce = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_STATUS,
                                           sizeof(cl_build_status), &status,
                                           NULL);  gpu_check(ce);
        } while (status == CL_BUILD_IN_PROGRESS);
        if (status != CL_BUILD_SUCCESS)
        {
                ce = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0,
                                           NULL, &sz);  gpu_check(ce);
                errors = xmalloc(sz);
                ce = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                                           errors, NULL);  gpu_check(ce);

                quit("Errors during '%s' compilation:\n%s", filename, errors);
        }

        /* Create the corresponding kernel */
        kernel = clCreateKernel(prog, kernelname, &ce);  gpu_check(ce);

        /* Unload the program source code */
        e = munmap(data, st.st_size);
        if (e == -1)
                error("Unmapping file '%s'", filename);
        e = close(fd);
        if (e == -1)
                error("Closing file '%s'", filename);

        /* Finally pack all the control information together */
        gpu = xmalloc(sizeof(struct gpu));
        gpu->context = ctx;
        gpu->device = dev;
        gpu->queue = queue;
        gpu->program = prog;
        gpu->kernel = kernel;
        return gpu;
}


void send_nn_arguments (struct gpu *gpu, struct db *trdb, struct db *db)
{
        cl_int e;

        gpu->trdata = clCreateBuffer(gpu->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     trdb->count * trdb->dimensions * trdb->typesize,
                                     trdb->data, &e);  gpu_check(e);
        gpu->data = clCreateBuffer(gpu->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   db->count * db->dimensions * db->typesize,
                                   db->data, &e);  gpu_check(e);
        gpu->trklasses = clCreateBuffer(gpu->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        trdb->count * sizeof(int), trdb->klass,
                                        &e);  gpu_check(e);
        gpu->klasses = clCreateBuffer(gpu->context, CL_MEM_WRITE_ONLY,
                                      db->count * sizeof(int), NULL, &e);  gpu_check(e);

        e = clSetKernelArg(gpu->kernel, 0, sizeof(int), &db->dimensions);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 1, sizeof(int), &trdb->count);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 2, sizeof(cl_mem), &gpu->trdata);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 3, sizeof(cl_mem), &gpu->trklasses);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 4, sizeof(int), &db->count);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 5, sizeof(cl_mem), &gpu->data);  gpu_check(e);
        e = clSetKernelArg(gpu->kernel, 6, sizeof(cl_mem), &gpu->klasses);  gpu_check(e);
}


void execute_kernel (struct gpu *gpu)
{
        cl_event ev;
        cl_int e;

        e = clEnqueueTask(gpu->queue, gpu->kernel, 0, NULL, &ev);  gpu_check(e);
        e = clWaitForEvents(1, &ev);  gpu_check(e);
}


void get_nn_result (struct gpu *gpu, int size, void *data)
{
        cl_int e;

        e = clEnqueueReadBuffer(gpu->queue, gpu->klasses, CL_TRUE, 0, size,
                                data, 0, NULL, NULL);  gpu_check(e);
}


void destroy_gpu (struct gpu *gpu)
{
        cl_int e;

        e = clReleaseMemObject(gpu->trdata);     gpu_check(e);
        e = clReleaseMemObject(gpu->trklasses);  gpu_check(e);
        e = clReleaseMemObject(gpu->data);       gpu_check(e);
        e = clReleaseMemObject(gpu->klasses);    gpu_check(e);
        e = clReleaseKernel(gpu->kernel);        gpu_check(e);
        e = clReleaseProgram(gpu->program);      gpu_check(e);
        e = clReleaseCommandQueue(gpu->queue);   gpu_check(e);
        e = clReleaseContext(gpu->context);      gpu_check(e);

        free(gpu);
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

        print_message(file, line, 3, "OpenCL error: %s", errmsg);
}

