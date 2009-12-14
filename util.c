#include "util.h"

#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef __APPLE__
#  include <malloc.h>
#endif


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


void *allocate_memory (const char *file, int line, size_t alignment,
                       size_t bytes)
{
        void *mem;

#ifdef __APPLE__
        mem = malloc(bytes);
#else
        if (alignment > 0)
                mem = memalign(alignment, bytes);
        else
                mem = malloc(bytes);
#endif

        if (mem == NULL)
                print_message(file, line, 4, "Unable to allocate %d byte%s",
                              bytes, (alignment > 0 ? "s with %d byte alignment"
                                                    : "s"));
        return mem;
}


char *string_concat (const char *file, int line, const char *a, const char *b)
{
        int len;
        char *ab;

        len = (a != NULL ? strlen(a) : 0) +
              (b != NULL ? strlen(b) : 0);

        if (len == 0)
                return NULL;
        len += 2;

        ab = allocate_memory(file, line, 0, len);
        snprintf(ab, len - 1, "%s%s", (a != NULL ? a : "\0"),
                                      (b != NULL ? b : "\0"));
        return ab;
}

