#ifndef __nnvect_db
#define __nnvect_db

#include "nn.h"
#include <sys/types.h>

#define DOUBLE_DATA(db) ((double *) (db)->data)
#define FLOAT_DATA(db)  ((float *)  (db)->data)
#define INT_DATA(db)    ((int *)    (db)->data)
#define SHORT_DATA(db)  ((short *)  (db)->data)
#define BYTE_DATA(db)   ((char *)   (db)->data)

#define DOUBLE_DISTANCE(db)  ((double *)       (db)->distance)
#define FLOAT_DISTANCE(db)   ((float *)        (db)->distance)
#define UINT_DISTANCE(db)    ((unsigned int *) (db)->distance)

struct db
{
        enum valuetype type;
        size_t typesize;
        size_t distsize;
        int dimensions;
        int count;
        int block_items;
        int has_floats;
        int transposed;
        int wanted_block_size;
        int real_dimensions;
        int real_count;
        int label_count;
        int *label;
        int *klass;
        void *data;
        void *distance;
};

struct db *load_db            (const char *, enum valuetype, int, int);
struct db *load_db_transposed (const char *, enum valuetype, int, int, int); 
void       print_db_info      (struct db *);
void       free_db            (struct db *);


static inline int label_index (struct db *db, int klass)
{
        int i;

        i = 0;
        while (i < db->label_count && db->label[i] != klass)
                i++;
        return i;
}

#endif /* __nnvect_db */
