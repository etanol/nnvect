#include "db.h"
#include "util.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


struct file
{
        int handle;
        off_t size;
        char *name;
        char *data;
};


static void open_file (const char *name, struct file *f)
{
        int e;
        struct stat s;

        f->name = xmalloc(strlen(name) + 1);
        strcpy(f->name, name);

        f->handle = open(name, O_RDONLY);
        if (f->handle == -1)
                fatal("Could not open '%s'", name);

        e = fstat(f->handle, &s);
        if (e == -1)
                fatal("Could not stat '%s'", name);

        f->size = s.st_size;
        f->data = mmap(NULL, s.st_size, PROT_READ, MAP_PRIVATE, f->handle, 0);
        if (f->data == MAP_FAILED)
                fatal("Could not map '%s'", name);
}


static void close_file (struct file *f)
{
        int e;

        e = munmap(f->data, f->size);
        if (e == -1)
                error("Unmapping file '%s'", f->name);

        e = close(f->handle);
        if (e == -1)
                error("Closing file '%s'", f->name);

        free(f->name);
}


static void read_info (struct file *file, struct db *db)
{
        char *line, *next;

        /* Read the values in order, no need to check for the labels.  If the
         * order ever changes in the Python analyzer, it can also be changed
         * here */
        line = file->data;
        db->count = (int) strtol(&line[9], &next, 10);
        line = next + 1;
        db->real_dimensions = (int) strtol(&line[9], &next, 10);
        line = next + 1;
        db->has_floats = line[9] == 'y';
}


static void read_data (struct file *file, struct db *db)
{
        char *c, *next;
        int dim, i;
        int lc;  /* Line Count so far */

        c = file->data;
        lc = 0;

        /* Each iteration parses a single line */
        while (lc < db->count)
        {
                while (*c == ' ' || *c == '\t')
                        c++;
                db->klass[lc] = (int) strtol(c, &next, 10);
                if (db->klass[lc] == 0 && c == next)
                        fatal("Parsing index %d \"%10s...\"", lc, c);
                c = next;
                while (*c == ' ' || *c == '\t')
                        c++;
                while (*c != '\n')
                {
                        dim = (int) strtol(c, &next, 10);
                        if (dim == 0 && c == next)
                                fatal("Parsing dimension at %d \"%10s...\"",
                                      dim, lc, c);
                        i = lc * db->dimensions + (dim - 1);
                        c = next + 1;
                        switch (db->type)
                        {
                        case BYTE:
                                ((char *) db->data)[i] = (char) strtol(c, &next, 10);
                                if (((char *) db->data)[i] == 0 && c == next)
                                        fatal("Parsing char value at %d, %d \"%10s...\"",
                                              lc, dim, c);
                                break;
                        case SHORT:
                                ((short *) db->data)[i] = (short) strtol(c, &next, 10);
                                if (((short *) db->data)[i] == 0 && c == next)
                                        fatal("Parsing short value at %d, %d \"%10s...\"",
                                              lc, dim, c);
                                break;
                        case INT:
                                ((int *) db->data)[i] = (int) strtol(c, &next, 10);
                                if (((int *) db->data)[i] == 0 && c == next)
                                        fatal("Parsing int value at %d, %d \"%10s...\"",
                                              lc, dim, c);
                                break;
                        case FLOAT:
                                ((float *) db->data)[i] = (float) strtod(c, &next);
                                if (((float *) db->data)[i] == 0.0f && c == next)
                                        fatal("Parsing float value at %d, %d \"%10s...\"",
                                              lc, dim, c);
                                break;
                        case DOUBLE:
                                ((double *) db->data)[i] = strtod(c, &next);
                                if (((double *) db->data)[i] == 0.0 && c == next)
                                        fatal("Parsing double value at %d, %d \"%10s...\"",
                                              lc, dim, c);
                                break;
                        }
                        c = next;
                        while (*c == ' ' || *c == '\t')
                                c++;
                }
                lc++;
                c++;
        }
}


struct db *load_db (const char *filename, enum valuetype type,
                    int max_block_size, int want_padding, int want_distances)
{
        char *infoname;
        int check_float, typesize, rowsize, distsize;
        struct db *db;
        struct file file;

        db = xmalloc(sizeof(struct db));

        infoname = xstrcat(filename, ".info");
        open_file(infoname, &file);
        read_info(&file, db);
        close_file(&file);
        free(infoname);

        check_float = 0;
        typesize = distsize = 0;
        switch (type)
        {
        case BYTE:
                typesize = sizeof(char);
                distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case SHORT:
                typesize = sizeof(short);
                distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case INT:
                typesize = sizeof(int);
                distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case FLOAT:
                typesize = distsize = sizeof(float);
                check_float = 0;
                break;
        case DOUBLE:
                typesize = distsize = sizeof(double);
                check_float = 0;
                break;
        }

        if (check_float && db->has_floats)
                quit("Database has floating point numbers but an integer type was requested");

        if (want_padding)
        {
                rowsize = PADDED(db->real_dimensions * typesize);
                db->dimensions = rowsize / typesize;
                db->data = xmalloc_aligned(db->count * rowsize);
        }
        else
        {
                rowsize = db->real_dimensions * typesize;
                db->dimensions = db->real_dimensions;
                db->data = xmalloc(db->count * rowsize);
        }
        db->type = type;
        db->klass = xmalloc(db->count * sizeof(int));
        memset(db->klass, 0, db->count * sizeof(int));
        memset(db->data, 0, db->count * rowsize);

        if (max_block_size > 0 && db->count * rowsize > max_block_size)
                db->block_items = max_block_size / rowsize;
        else
                db->block_items = 0;

        if (want_distances)
                db->distance = xmalloc(db->count * distsize);
        else
                db->distance = NULL;

        open_file(filename, &file);
        read_data(&file, db);
        close_file(&file);

        return db;
}


void print_db_info (struct db *db)
{
        int padcolumns;
        unsigned int typesize, datasize, padsize, blocksize;

        typesize = 0;
        switch (db->type)
        {
        case BYTE:    typesize = sizeof(char);    break;
        case SHORT:   typesize = sizeof(short);   break;
        case INT:     typesize = sizeof(int);     break;
        case FLOAT:   typesize = sizeof(float);   break;
        case DOUBLE:  typesize = sizeof(double);  break;
        }

        padcolumns = db->dimensions - db->real_dimensions;
        padsize = db->count * padcolumns * typesize;
        datasize = db->count * db->dimensions * typesize;
        blocksize = db->block_items * db->dimensions * typesize;

        printf("    There are %d elements\n", db->count);
        printf("    There are %d dimensions, of which %d are padding\n",
               db->dimensions, padcolumns);
        printf("    Class array is %u bytes in size\n",
               (unsigned int) (db->count * sizeof(int)));
        printf("    Data array is %u bytes, of which %u (%.1f%%) are padding\n",
               datasize, padsize, (padsize * 100.0f) / datasize);
        printf("    Data array would be %u bytes without padding\n",
               db->count * db->real_dimensions * typesize);
        if (db->block_items > 0)
                printf("    Each block has %d elements (%u bytes)\n",
                       db->block_items, blocksize);
        else
                printf("    Blocking not requested, unneeded, or too small block size\n");
}


void free_db (struct db *db)
{
        if (db->distance != NULL)
                free(db->distance);
        free(db->data);
        free(db->klass);
        free(db);
}

