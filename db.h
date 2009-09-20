#ifndef __nnvect_db
#define __nnvect_db

#include "nn.h"

struct db
{
        enum valuetype type;
        int count;
        int dimensions;
        int real_dimensions;
        int has_floats;
        int *klass;
        void *data;
};

struct db *load_db       (const char *, enum valuetype, int);
void       print_db_info (struct db *);
void       free_db       (struct db *);

#endif /* __nnvect_db */
