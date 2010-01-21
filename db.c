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
        int i;
        char *line, *next;
        double drop;

        /* Read the values in order, no need to check for the labels.  If the
         * order ever changes in the Python analyzer, it can also be changed
         * here */
        line = file->data;
        db->real_count = (int) strtol(&line[9], &next, 10);
        line = next + 1;
        db->real_dimensions = (int) strtol(&line[9], &next, 10);
        line = next + 1;
        if (line[9] == 'y')
        {
                db->has_floats = 1;
                next = &line[12];
        }
        else
        {
                db->has_floats = 0;
                next = &line[11];
        }
        line = next + 1;
        /* Drop maximum and minimum */
        drop = strtod(&line[9], &next);
        line = next + 1;
        drop = strtod(&line[9], &next);
        line = next + 1;
        db->label_count = (int) strtol(&line[9], &next, 10);
        line = next + 1;
        db->label = xmalloc(db->label_count * sizeof(int));
        for (i = 0;  i < db->label_count;  i++)
        {
                db->label[i] = (int) strtol(&line[9], &next, 10);
                line = next + 1;
        }
}


static void read_data (struct file *file, struct db *db)
{
        char *c, *next;
        int b, blc, bs;  /* Only for transposed loads */
        int dim, i;
        int lc;  /* Line Count so far */

        bs = db->block_items * db->dimensions;
        c = file->data;
        lc = 0;

        /* Each iteration parses a single line */
        while (lc < db->real_count)
        {
                while (*c == ' ' || *c == '\t')
                        c++;
                db->klass[lc] = (int) strtol(c, &next, 10);
                if (db->klass[lc] == 0 && c == next)
                        quit("Parsing index %d \"%10s...\"", lc, c);
                c = next;
                while (*c == ' ' || *c == '\t')
                        c++;
                while (*c != '\n')
                {
                        dim = (int) strtol(c, &next, 10);
                        if (dim == 0 && c == next)
                                quit("Parsing dimension at %d \"%10s...\"",
                                     dim, lc, c);
                        if (db->transposed)
                        {
                                b = lc / db->block_items;
                                blc = lc - b * db->block_items;
                                i = (b * bs) +
                                    ((dim - 1) * db->block_items + blc);
                        }
                        else
                                i = lc * db->dimensions + (dim - 1);
                        c = next + 1;
                        switch (db->type)
                        {
                        case BYTE:
                                BYTE_DATA(db)[i] = (char) strtol(c, &next, 10);
                                if (BYTE_DATA(db)[i] == 0 && c == next)
                                        quit("Parsing char value at %d, %d \"%10s...\"",
                                             lc, dim, c);
                                break;
                        case SHORT:
                                SHORT_DATA(db)[i] = (short) strtol(c, &next, 10);
                                if (SHORT_DATA(db)[i] == 0 && c == next)
                                        quit("Parsing short value at %d, %d \"%10s...\"",
                                             lc, dim, c);
                                break;
                        case INT:
                                INT_DATA(db)[i] = (int) strtol(c, &next, 10);
                                if (INT_DATA(db)[i] == 0 && c == next)
                                        quit("Parsing int value at %d, %d \"%10s...\"",
                                             lc, dim, c);
                                break;
                        case FLOAT:
                                FLOAT_DATA(db)[i] = (float) strtod(c, &next);
                                if (FLOAT_DATA(db)[i] == 0.0f && c == next)
                                        quit("Parsing float value at %d, %d \"%10s...\"",
                                             lc, dim, c);
                                break;
                        case DOUBLE:
                                DOUBLE_DATA(db)[i] = strtod(c, &next);
                                if (DOUBLE_DATA(db)[i] == 0.0 && c == next)
                                        quit("Parsing double value at %d, %d \"%10s...\"",
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


static struct db *prepare_db (const char *filename, enum valuetype type)
{
        char *infoname;
        struct db *db;
        int check_float;
        struct file file;

        db = xmalloc(sizeof(struct db));
        memset(db, 0, sizeof(struct db));
        db->type = type;

        infoname = xstrcat(filename, ".info");
        open_file(infoname, &file);
        read_info(&file, db);
        close_file(&file);
        free(infoname);

        check_float = 0;
        switch (type)
        {
        case BYTE:
                db->typesize = sizeof(char);
                db->distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case SHORT:
                db->typesize = sizeof(short);
                db->distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case INT:
                db->typesize = sizeof(int);
                db->distsize = sizeof(unsigned int);
                check_float = 1;
                break;
        case FLOAT:
                db->typesize = db->distsize = sizeof(float);
                check_float = 0;
                break;
        case DOUBLE:
                db->typesize = db->distsize = sizeof(double);
                check_float = 0;
                break;
        }

        if (check_float && db->has_floats)
                quit("Database has floating point numbers but an integer type was requested");

        return db;
}


struct db *load_db (const char *filename, enum valuetype type,
                    int max_block_size, int row_alignment)
{
        int rowsize;
        struct db *db;
        struct file file;

        db = prepare_db(filename, type);

        db->count = db->real_count;
        rowsize = db->real_dimensions * db->typesize;
        if (row_alignment > 0 && rowsize % row_alignment > 0)
                rowsize +=  row_alignment - (rowsize % row_alignment);
        db->dimensions = rowsize / db->typesize;

        db->data = xmalloc_aligned(db->count * rowsize, row_alignment);
        db->klass = xmalloc(db->count * sizeof(int));
        db->distance = xmalloc(db->count * db->distsize);
        memset(db->data, 0, db->count * rowsize);
        memset(db->klass, 0, db->count * sizeof(int));

        db->wanted_block_size = max_block_size;
        if (max_block_size > 0 && db->count * rowsize > max_block_size)
                db->block_items = max_block_size / rowsize;
        else
                db->block_items = 0;

        db->transposed = 0;
        open_file(filename, &file);
        read_data(&file, db);
        close_file(&file);

        return db;
}


struct db *load_db_transposed (const char *filename, enum valuetype type,
                               int chunks, int row_pad, int column_pad)
{
        struct db *db;
        struct file file;

        db = prepare_db(filename, type);

        db->count = db->real_count;
        if (row_pad > 0 && (db->real_count / chunks) % row_pad > 0)
                db->count += chunks * (row_pad - (db->real_count / chunks) %
                                                 row_pad);
        db->dimensions = db->real_dimensions;
        if (column_pad > 0 && db->real_dimensions % column_pad > 0)
                db->dimensions += column_pad - (db->real_dimensions %
                                                column_pad);

        db->data = xmalloc(db->count * db->dimensions * db->typesize);
        db->klass = xmalloc(db->count * sizeof(int));
        db->distance = NULL;
        memset(db->data, 0, db->count * db->dimensions * db->typesize);
        memset(db->klass, 0, db->count * sizeof(int));

        db->block_items = db->count / chunks;

        db->transposed = 1;
        open_file(filename, &file);
        read_data(&file, db);
        close_file(&file);

        return db;
}


void print_db_info (struct db *db)
{
        int padcolumns;
        unsigned int datasize, padsize, rowsize, blocksize;

        padcolumns = db->dimensions - db->real_dimensions;
        rowsize = db->dimensions * db->typesize;
        padsize = db->count * padcolumns * db->typesize;
        datasize = db->count * rowsize;
        blocksize = db->block_items * rowsize;

        printf("    There are %d elements\n", db->count);
        printf("    There are %d dimensions, of which %d are padding\n",
               db->dimensions, padcolumns);
        printf("    There are %d different classes\n", db->label_count);
        printf("    Class array is %u bytes in size\n",
               (unsigned int) (db->count * sizeof(int)));
        printf("    Data array is %u bytes, of which %u (%.1f%%) are padding\n",
               datasize, padsize, (padsize * 100.0f) / datasize);
        printf("    Data array would be %lu bytes without padding\n",
               db->count * db->real_dimensions * db->typesize);
        /* Detailed blocking diagnostics */
        if (db->wanted_block_size > 0)
        {
                if (db->block_items > 0)
                        printf("    Each block has %d elements (%u bytes)\n",
                               db->block_items, blocksize);
                else
                {
                        if (rowsize > db->wanted_block_size)
                                printf("    Block size is too small (should be"
                                       "at least %d)\n", rowsize);
                        else
                                printf("    Data fits in a single block\n");
                }
        }
        else
                printf("    Blocking not requested\n");
}


void free_db (struct db *db)
{
        if (db->distance != NULL)
                free(db->distance);
        free(db->label);
        free(db->data);
        free(db->klass);
        free(db);
}

