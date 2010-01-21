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

struct _GPULocalData
{
        cl_device_id id;
        cl_command_queue queue;
        cl_kernel m_kernel;
        cl_kernel r_kernel;

        cl_mem trdata;
        cl_mem trklasses;
        cl_mem data;
        cl_mem klasses;
        cl_mem distances;
        cl_mem indices;

        cl_event link;
};

struct _GPUGlobalData
{
        int count;
        int trcount;
        int devcount;

        cl_context context;
        cl_program program;

        size_t m_local[2];
        size_t m_global[2];
        size_t r_local[2];
        size_t r_global[2];

        struct _GPULocalData *dev;
        struct db *db;
};

static struct _GPUGlobalData GPU;


int init_gpu (const char *filename)
{
        int fd, i, e;
        struct stat st;
        size_t filesize, sz;
        char *data, *errors;
        cl_build_status status;
        cl_device_id *id;
        cl_int ce;

        memset(&GPU, 0, sizeof(struct _GPUGlobalData));

        /* Create context */
        GPU.context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL,
                                              NULL, &ce);  gpu_check(ce);
        /* Get device list and prepare structures */
        ce = clGetContextInfo(GPU.context , CL_CONTEXT_DEVICES, 0, NULL, &sz);  gpu_check(ce);
        id = xmalloc(sz);
        ce = clGetContextInfo(GPU.context, CL_CONTEXT_DEVICES, sz, id, NULL);  gpu_check(ce);
        GPU.devcount = sz / sizeof(cl_device_id);

        GPU.dev = xmalloc(sizeof(struct _GPULocalData) * GPU.devcount);
        memset(GPU.dev, 0, sizeof(struct _GPULocalData) * GPU.devcount);
        GPU.devcount = sz / sizeof(cl_device_id);

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
        clBuildProgram(GPU.program, 0, NULL, NULL, NULL, NULL);
        do {
                ce = clGetProgramBuildInfo(GPU.program, id[0],
                                           CL_PROGRAM_BUILD_STATUS,
                                           sizeof(cl_build_status), &status,
                                           NULL);  gpu_check(ce);
                if (status != CL_BUILD_IN_PROGRESS)
                        break;
                sleep(1);

        } while (1);
        if (status != CL_BUILD_SUCCESS)
        {
                ce = clGetProgramBuildInfo(GPU.program, id[0], CL_PROGRAM_BUILD_LOG,
                                           0, NULL, &sz);  gpu_check(ce);
                errors = xmalloc(sz);
                ce = clGetProgramBuildInfo(GPU.program, id[0], CL_PROGRAM_BUILD_LOG,
                                           sz, errors, NULL);  gpu_check(ce);
                destroy_gpu();
                quit("Errors during '%s' compilation:\n%s", filename, errors);
        }

        /* Unload the program source code */
        e = munmap(data, st.st_size);
        if (e == -1)
                error("Unmapping file '%s'", filename);
        e = close(fd);
        if (e == -1)
                error("Closing file '%s'", filename);

        /* Now setup control structures for each device */
        for (i = 0;  i < GPU.devcount;  i++)
        {
                GPU.dev[i].id = id[i];
                GPU.dev[i].queue = clCreateCommandQueue(GPU.context, id[i], 0, &ce); gpu_check(ce);
                GPU.dev[i].m_kernel = clCreateKernel(GPU.program, "distance_map", &ce);gpu_check(ce);
                GPU.dev[i].r_kernel = clCreateKernel(GPU.program, "reduction", &ce);  gpu_check(ce);
        }
        free(id);

        return GPU.devcount;
}


void prepare_invocations (struct db *trdb, struct db *db)
{
        int i, bs;
        cl_int e;

        bs = db->block_items * db->dimensions * db->typesize;

        for (i = 0;  i < GPU.devcount;  i++)
        {
                GPU.dev[i].trdata = clCreateBuffer(GPU.context,
                                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   trdb->count * trdb->dimensions * trdb->typesize,
                                                   trdb->data, &e);  gpu_check(e);
                GPU.dev[i].data = clCreateBuffer(GPU.context,
                                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 bs, db->data + i * bs, &e);  gpu_check(e);
                GPU.dev[i].trklasses = clCreateBuffer(GPU.context,
                                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                      trdb->count * sizeof(int), trdb->klass,
                                                      &e);  gpu_check(e);
                GPU.dev[i].klasses = clCreateBuffer(GPU.context, CL_MEM_WRITE_ONLY,
                                                    db->block_items * sizeof(int), NULL, &e);  gpu_check(e);
                GPU.dev[i].distances = clCreateBuffer(GPU.context, CL_MEM_READ_WRITE,
                                                      db->block_items * (trdb->count / 16) * sizeof(float),
                                                      NULL, &e);  gpu_check(e);
                GPU.dev[i].indices = clCreateBuffer(GPU.context, CL_MEM_READ_WRITE,
                                                    db->block_items * (trdb->count / 16) * sizeof(int),
                                                    NULL, &e);  gpu_check(e);

                e = clSetKernelArg(GPU.dev[i].m_kernel, 0, sizeof(int), &db->dimensions);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 1, sizeof(int), &trdb->real_count);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 2, sizeof(int), &trdb->count);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 3, sizeof(cl_mem), &GPU.dev[i].trdata);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 4, sizeof(int), &db->block_items);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 5, sizeof(cl_mem), &GPU.dev[i].data);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 6, sizeof(cl_mem), &GPU.dev[i].distances);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].m_kernel, 7, sizeof(cl_mem), &GPU.dev[i].indices);  gpu_check(e);

                e = clSetKernelArg(GPU.dev[i].r_kernel, 0, sizeof(int), &trdb->count);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].r_kernel, 1, sizeof(int), &db->block_items);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].r_kernel, 2, sizeof(cl_mem), &GPU.dev[i].distances);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].r_kernel, 3, sizeof(cl_mem), &GPU.dev[i].indices);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].r_kernel, 4, sizeof(cl_mem), &GPU.dev[i].trklasses);  gpu_check(e);
                e = clSetKernelArg(GPU.dev[i].r_kernel, 5, sizeof(cl_mem), &GPU.dev[i].klasses);  gpu_check(e);
        }

        GPU.db = db;
        GPU.m_local[0] = 16;
        GPU.m_local[1] = 4;
        GPU.m_global[0] = (db->block_items / 64) * 16;
        GPU.m_global[1] = (trdb->count / 16) * 4;

        GPU.r_local[0] = 64;
        GPU.r_local[1] = 1;
        GPU.r_global[0] = db->block_items;
        GPU.r_global[1] = 1;
}


void invoke_kernels (void)
{
        int i;
        cl_int e;

        for (i = 0;  i < GPU.devcount;  i++)
        {
                e = clEnqueueNDRangeKernel(GPU.dev[i].queue, GPU.dev[i].m_kernel,
                                           2, NULL, GPU.m_global, GPU.m_local, 0,
                                           NULL, &GPU.dev[i].link);  gpu_check(e);
        }
        for (i = 0;  i < GPU.devcount;  i++)
        {
                e = clEnqueueNDRangeKernel(GPU.dev[i].queue, GPU.dev[i].r_kernel,
                                           2, NULL, GPU.r_global, GPU.r_local, 1,
                                           &GPU.dev[i].link, NULL);  gpu_check(e);
        }
        for (i = 0;  i < GPU.devcount;  i++)
        {
                e = clFinish(GPU.dev[i].queue);  gpu_check(e);
        }
}


void get_nn_result (int size, void *data)
{
        int i;
        cl_int e;

        for (i = 0;  i < GPU.devcount;  i++)
        {
                printf ("Get %i\n", i);
                e = clEnqueueReadBuffer(GPU.dev[i].queue, GPU.dev[i].klasses,
                                        CL_TRUE, 0, GPU.db->block_items * sizeof(int), data + i * GPU.db->block_items * sizeof(int),
                                        0, NULL, NULL);  gpu_check(e);
        }
}


void destroy_gpu (void)
{
        int i;

        for (i = 0;  i < GPU.devcount;  i++)
        {
                if (GPU.dev[i].trdata)
                        clReleaseMemObject(GPU.dev[i].trdata);
                if (GPU.dev[i].trklasses)
                        clReleaseMemObject(GPU.dev[i].trklasses);
                if (GPU.dev[i].data)
                        clReleaseMemObject(GPU.dev[i].data);
                if (GPU.dev[i].klasses)
                        clReleaseMemObject(GPU.dev[i].klasses);
                if (GPU.dev[i].distances)
                        clReleaseMemObject(GPU.dev[i].distances);
                if (GPU.dev[i].indices)
                        clReleaseMemObject(GPU.dev[i].indices);
                if (GPU.dev[i].m_kernel)
                        clReleaseKernel(GPU.dev[i].m_kernel);
                if (GPU.dev[i].r_kernel)
                        clReleaseKernel(GPU.dev[i].r_kernel);
                if (GPU.dev[i].queue)
                        clReleaseCommandQueue(GPU.dev[i].queue);
        }
        free(GPU.dev);
        if (GPU.program)
                clReleaseProgram(GPU.program);
        if (GPU.context)
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

