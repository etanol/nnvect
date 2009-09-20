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
                c = next;
                while (*c == ' ' || *c == '\t')
                        c++;
                while (*c != '\n')
                {
                        dim = (int) strtol(c, &next, 10);
                        i = lc * db->dimensions + dim;
                        c = next + 1;
                        switch (db->type)
                        {
                        case BYTE:
                                ((char *) db->data)[i] = (char) strtol(c, &next, 10);
                                break;
                        case SHORT:
                                ((short *) db->data)[i] = (short) strtol(c, &next, 10);
                                break;
                        case INTEGER:
                                ((int *) db->data)[i] = (int) strtol(c, &next, 10);
                                break;
                        case FLOAT:
                                ((float *) db->data)[i] = (float) strtod(c, &next);
                                break;
                        case DOUBLE:
                                ((double *) db->data)[i] = strtod(c, &next);
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


struct db *load_db (const char *filename, enum datatype type, int want_padding)
{
        char *infoname;
        int check_float, typesize, rowsize;
        struct db *db;
        struct file file;

        db = xmalloc(sizeof(struct db));

        infoname = xstrcat(filename, ".info");
        open_file(infoname, &file);
        read_info(&file, db);
        close_file(&file);
        free(infoname);

        check_float = 0;
        typesize = 0;
        switch (type)
        {
        case BYTE:    typesize = sizeof(char);    check_float = 1;  break;
        case SHORT:   typesize = sizeof(short);   check_float = 1;  break;
        case INTEGER: typesize = sizeof(int);     check_float = 1;  break;
        case FLOAT:   typesize = sizeof(float);   check_float = 0;  break;
        case DOUBLE:  typesize = sizeof(double);  check_float = 0;  break;
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

        open_file(filename, &file);
        read_data(&file, db);
        close_file(&file);

        return db;
}


void free_db (struct db *db)
{
        free(db->data);
        free(db->klass);
        free(db);
}

