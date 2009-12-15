#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "ocl_util.h"

#define oclCheckError(var) \
        do { \
                if ((var) != CL_SUCCESS) \
                { \
                        errno = 0; \
                        fatal("In OpenCL call, error code %d", (var)); \
                } \
        } while (0)

#define LABEL_PAD -22


static void oclPrintPlatformInfoString (cl_platform_id plat, cl_platform_info param, char *label)
{
        char *s;
        cl_int e;
        size_t b;

        e = clGetPlatformInfo(plat, param, 0, NULL, &b);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying platform string length for \"%s\"", label);
        s = xmalloc(b);
        e = clGetPlatformInfo(plat, param, b, s, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying platform string for \"%s\"", label);
        printf("   %-10s: %s\n", label, s);
        free(s);
}


static void oclPrintDeviceInfoString (cl_device_id dev, cl_device_info param, char *label)
{
        char *s;
        cl_int e;
        size_t b;

        e = clGetDeviceInfo(dev, param, 0, NULL, &b);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying device string length for \"%s\"", label);
        s = xmalloc(b);
        e = clGetDeviceInfo(dev, param, b, s, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying device string for \"%s\"", label);
        printf("      %*s: %s\n", LABEL_PAD, label, s);
        free(s);
}


static void oclPrintDeviceInfoBool (cl_device_id dev, cl_device_info param, char *label)
{
        cl_bool b;
        cl_int e;

        e = clGetDeviceInfo(dev, param, sizeof(cl_bool), &b, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying device boolean for \"%s\"", label);
        printf("      %*s: %s\n", LABEL_PAD, label, (b == CL_TRUE ? "yes" : "no"));
}


static void oclPrintDeviceInfoInt (cl_device_id dev, cl_device_info param, char *label)
{
        cl_uint n;
        cl_int e;

        e = clGetDeviceInfo(dev, param, sizeof(cl_uint), &n, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying device integer for \"%s\"", label);
        printf("      %*s: %u\n", LABEL_PAD, label, n);
}


static void oclPrintDeviceInfoLong (cl_device_id dev, cl_device_info param, char *label)
{
        cl_ulong n;
        cl_int e;

        e = clGetDeviceInfo(dev, param, sizeof(cl_ulong), &n, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Querying device long integer for \"%s\"", label);
        printf("      %*s: %lu\n", LABEL_PAD, label, n);
}


int main ()
{
        cl_int e;
        cl_device_id *devices;
        cl_platform_id *platforms;
        cl_uint device_count, plat_count, widim;
        int p, i, j;
        size_t *dims;

        e = clGetPlatformIDs(0, NULL, &plat_count);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Listing platform IDs");
        platforms = xmalloc(plat_count * sizeof(cl_platform_id));
        e = clGetPlatformIDs(plat_count, platforms, NULL);
        if (e != CL_SUCCESS)
                ocl_fatal(e, "Listing platform IDs");
        printf("%d platforms found:\n\n", plat_count);
        for (p = 0;  p < plat_count;  p++)
        {
                printf("Platform %d:\n\n", p);
                oclPrintPlatformInfoString(platforms[p], CL_PLATFORM_NAME, "Name");
                oclPrintPlatformInfoString(platforms[p], CL_PLATFORM_VENDOR, "Vendor");
                oclPrintPlatformInfoString(platforms[p], CL_PLATFORM_VERSION, "Version");
                oclPrintPlatformInfoString(platforms[p], CL_PLATFORM_EXTENSIONS, "Extensions");
                printf("\n");

                e = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
                if (e != CL_SUCCESS)
                        ocl_fatal(e, "Listing device IDs for platform %d", p);
                devices = xmalloc(device_count * sizeof(cl_device_id));
                e = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices, NULL);
                if (e != CL_SUCCESS)
                        ocl_fatal(e, "Listing device IDs for platform %d", p);
                printf("   %d GPU devices found:\n\n", device_count);

                for (i = 0;  i < device_count;  i++)
                {
                        printf("   Device %d:\n\n", i);

                        /* Name, version and frequency */
                        oclPrintDeviceInfoString(devices[i], CL_DEVICE_NAME, "Name");
                        oclPrintDeviceInfoString(devices[i], CL_DEVICE_VENDOR, "Vendor");
                        oclPrintDeviceInfoInt(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, "Top MHz");
                        oclPrintDeviceInfoString(devices[i], CL_DEVICE_VERSION, "Version");
                        oclPrintDeviceInfoString(devices[i], CL_DRIVER_VERSION, "Driver");

                        /* Booelans */
                        oclPrintDeviceInfoBool(devices[i], CL_DEVICE_AVAILABLE, "Device available");
                        oclPrintDeviceInfoBool(devices[i], CL_DEVICE_COMPILER_AVAILABLE, "Compiler available");
                        oclPrintDeviceInfoBool(devices[i], CL_DEVICE_ENDIAN_LITTLE, "Little endian");
                        oclPrintDeviceInfoBool(devices[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, "Error correction");
                        printf("\n");

                        /* Memory */
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, "Global memory size");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "Global cache size");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Global cacheline size");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, "Local memory size");
                        oclPrintDeviceInfoInt(devices[i], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "Required alignment");
                        oclPrintDeviceInfoInt(devices[i], CL_DEVICE_MEM_BASE_ADDR_ALIGN, "Alignment bits");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Allocation limit");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "Constant buffer limit");
                        printf("\n");

                        /* Work items and executing units */
                        oclPrintDeviceInfoInt(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, "Compute units");
                        oclPrintDeviceInfoLong(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "Work group size");
                        oclPrintDeviceInfoInt(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Work item dimensions");

                        e = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &widim, NULL);
                        if (e != CL_SUCCESS)
                                ocl_fatal(e, "Getting work item dimensions");
                        printf("      %*s: [ ", LABEL_PAD, "Work group volume");
                        dims = xmalloc(sizeof(size_t) * widim);
                        e = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * widim, dims, NULL);
                        if (e != CL_SUCCESS)
                                ocl_fatal(e, "Getting work item sizes");
                        for (j = 0;  j < widim;  j++)
                        {
                                printf("%lu ", dims[j]);

                        }
                        printf("]\n");
                        free(dims);

                        oclPrintDeviceInfoString(devices[i], CL_DEVICE_EXTENSIONS, "Extensions");
                        printf("\n\n");
                }

                printf("\n");
                free(devices);
        }
        free(platforms);
        return 0;
}

