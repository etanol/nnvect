#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "misc.h"


static inline char *next_line (char *data, int *position, size_t limit)
{
        char *line;
        int i;

        i = *position;
        if (i < limit && (data[i] == '\n' || data[i] == '\0'))
        {
                data[i] = '\0';
                i++;
        }
        if (i >= limit)
        {
                *position = i;
                return NULL;
        }

        line = &data[i];
        while (i < limit && data[i] != '\n' && data[i] != '\0')
                i++;
        *position = i;
        if (i >= limit)
                /* No line jump at the last line of the file; normally caused by
                 * a really broken text editor.  There's not much to do about
                 * it */
                return NULL;

        data[i] = '\0';
        return line;
}


static inline char *skip_target (char *str)
{
        while (isspace(*str))
                str++;
        while (!isspace(*str))
                str++;

        return str;
}


static inline char *skip_element (char *str)
{
        while (*str != ':')
                str++;
        str++;
        while (isspace(*str))
                str++;
        while (*str && !isspace(*str))
                str++;

        return str;
}


void load_db (char *filename, float **features, int **classes,
              int *dimensions, int *item_count)
{
        int fd, e, pos;
        int count, cl, index, dims;
        int rowsize;
        float value;
        char *data, *line;
        struct stat s;

        fd = open(filename, O_RDONLY);
        if (fd == -1)
                fatal("Could not open '%s'", filename);

        e = fstat(fd, &s);
        if (e == -1)
                fatal("Could not stat '%s'", filename);

        data = mmap(NULL, s.st_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
                fatal("Could not map '%s'", filename);

        /* First pass to guess the dimensions of the array : number of lines and
         * maximum index */
        count = 0;
        dims = 0;
        pos = 0;
        line = next_line(data, &pos, s.st_size);
        while (line != NULL)
        {
                count++;
                line = skip_target(line);
                while (sscanf(line, "%d:%*f", &index) == 1)
                {
                        if (index > dims)
                                dims = index;
                        line = skip_element(line);
                }
                line = next_line(data, &pos, s.st_size);
        }

        /* Pad dimensions to ALIGN_BOUNDARY */
        rowsize = (dims * sizeof(float) + LOWER_MASK) & UPPER_MASK;
        dims = rowsize / sizeof(float);
        *dimensions = dims;
        *item_count = count;
        *features = malloc_aligned(count * rowsize);
        *classes = calloc(count, sizeof(int));
        if (*classes == NULL)
                fatal("Could not allocate classes array");

        /* Second pass to load the values into the array */
        count = 0;
        pos = 0;
        line = next_line(data, &pos, s.st_size);
        while (line != NULL)
        {
                sscanf(line, "%d", &cl);
                (*classes)[count] = cl;
                line = skip_target(line);
                while (sscanf(line, "%d:%f", &index, &value) == 2)
                {
                        (*features)[count * dims + index - 1] = value;
                        line = skip_element(line);
                }
                count++;
                line = next_line(data, &pos, s.st_size);
        }

        e = munmap(data, s.st_size);
        if (e == -1)
                error("Unmapping file '%s'", filename);
        e = close(fd);
        if (e == -1)
                error("Closing file '%s'", filename);
}

