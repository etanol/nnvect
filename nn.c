#include "nn.h"
#include "db.h"
#include "util.h"

#include <stddef.h>
#include <limits.h>
#include <float.h>

#define DECLARE(f) extern void f (int, int, int, void *, int *, int, int, void *, int *, void *);


/* Sequential versions defined at nnsca-*.c */
DECLARE(nn_byte_sca)
DECLARE(nn_short_sca)
DECLARE(nn_int_sca)
DECLARE(nn_float_sca)
DECLARE(nn_double_sca)

/* Vectorized versions defined at nnvect-*.c */
DECLARE(nn_byte_vec)
DECLARE(nn_short_vec)
#ifdef NO_SSE4
#  define nn_int_vec  NULL
#else
DECLARE(nn_int_vec)
#endif
DECLARE(nn_float_vec)
DECLARE(nn_double_vec)


void clear_distances (struct db *db)
{
        int i;

        switch (db->type)
        {
        case BYTE:
        case SHORT:
        case INT:
                for (i = 0;  i < db->count;  i++)
                        UINT_DISTANCE(db)[i] = UINT_MAX;
                break;
        case FLOAT:
                for (i = 0;  i < db->count;  i++)
                        FLOAT_DISTANCE(db)[i] = FLT_MAX;
                break;
        case DOUBLE:
                for (i = 0;  i < db->count;  i++)
                        DOUBLE_DISTANCE(db)[i] = DBL_MAX;
                break;
        }
}


void nn (enum valuetype type, int scalar, struct db *trdb, struct db *db)
{
        void (*func) (int, int, int, void *, int *, int, int, void *, int *, void *);
        int blockcount, trblockcount;

        func = NULL;
        switch (type)
        {
        case BYTE:
                func = (scalar ? nn_byte_sca : nn_byte_vec);
                break;
        case SHORT:
                func = (scalar ? nn_short_sca : nn_short_vec);
                break;
        case INT:
                func = (scalar ? nn_int_sca : nn_int_vec);
                break;
        case FLOAT:
                func = (scalar ? nn_float_sca : nn_float_vec);
                break;
        case DOUBLE:
                func = (scalar ? nn_double_sca : nn_double_vec);
                break;
        }
        if (func == NULL)
                quit("Invalid combination of implementation and value type");

        if (trdb->block_items > 0)
                trblockcount = trdb->block_items;
        else
                trblockcount = trdb->count;
        if (db->block_items > 0)
                blockcount = db->block_items;
        else
                blockcount = db->count;

        func(db->dimensions, trdb->count, trblockcount, trdb->data, trdb->klass,
             db->count, blockcount, db->data, db->klass, db->distance);
}

