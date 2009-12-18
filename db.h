#ifndef __nnvect_db
#define __nnvect_db

#include "nn.h"
#include <sys/types.h>

struct db
{
        enum valuetype type;
        size_t typesize;
        int count;
        int wanted_block_size;
        int block_items;
        int dimensions;
        int real_dimensions;
        int has_floats;
        int label_count;
        int *label;
        int *klass;
        void *data;
        void *distance;
};

struct db *load_db       (const char *, enum valuetype, int, int, int);
void       print_db_info (struct db *);
void       free_db       (struct db *);


static inline int label_index (struct db *db, int klass)
{
        int i;

        i = 0;
        while (i < db->label_count && db->label[i] != klass)
                i++;
        return i;
}

#endif /* __nnvect_db */
