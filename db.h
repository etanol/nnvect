#ifndef __nnvect_db
#define __nnvect_db

#include "nn.h"

struct db
{
        enum valuetype type;
        int count;
        int wanted_block_size;
        int block_items;
        int dimensions;
        int real_dimensions;
        int has_floats;
        int klass_count;
        int *klass;
        void *data;
        void *distance;
};

struct db *load_db       (const char *, enum valuetype, int, int, int);
void       print_db_info (struct db *);
void       free_db       (struct db *);

#endif /* __nnvect_db */
