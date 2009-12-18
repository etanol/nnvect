#include "gpu.h"
#include "util.h"


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

        print_message(file, line, 4, "OpenCL error: %s", errmsg);
}

