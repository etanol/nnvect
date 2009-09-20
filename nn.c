#include "nn.h"
#include "db.h"
#include "util.h"

#include <stddef.h>
#include <limits.h>
#include <float.h>

#define DECLARE(f)  extern void f##_U (int, int, void *, int *, int, void *, int *); \
                    extern void f##_B (int, int, int, void *, int *, int, void *, int *, void *);


/* Sequential versions defined at nnsca-*.c */
DECLARE(nn_byte_sca)
DECLARE(nn_short_sca)
DECLARE(nn_int_sca)
DECLARE(nn_float_sca)
DECLARE(nn_double_sca)

/* Vectorized versions defined at nnvect-*.c */
DECLARE(nn_byte_vec)
DECLARE(nn_short_vec)
DECLARE(nn_int_vec)
DECLARE(nn_float_vec)
DECLARE(nn_double_vec)


static void clear_tmp_distances (struct db *db)
{
        int i;

        if (db->distance == NULL)
                return;

        switch (db->type)
        {
        case BYTE:
        case SHORT:
        case INT:
                for (i = 0;  i < db->count;  i++)
                        ((unsigned int *) db->distance)[i] = UINT_MAX;
                break;
        case FLOAT:
                for (i = 0;  i < db->count;  i++)
                        ((float *) db->distance)[i] = FLT_MAX;
                break;
        case DOUBLE:
                for (i = 0;  i < db->count;  i++)
                        ((double *) db->distance)[i] = DBL_MAX;
                break;
        }

}


void nn (enum valuetype type, int scalar, struct db *trdb, struct db *db)
{
        if (trdb->block_items > 0)
        {
                /* Select blocked versions */
                void (*func) (int, int, int, void *, int *, int, void *, int *, void *);

                func = NULL;
                switch (type)
                {
                case BYTE:
                        func = (scalar ? nn_byte_sca_B : nn_byte_vec_B);
                        break;
                case SHORT:
                        func = (scalar ? nn_short_sca_B : nn_short_vec_B);
                        break;
                case INT:
                        func = (scalar ? nn_int_sca_B : nn_int_vec_B);
                        break;
                case FLOAT:
                        func = (scalar ? nn_float_sca_B : nn_float_vec_B);
                        break;
                case DOUBLE:
                        func = (scalar ? nn_double_sca_B : nn_double_vec_B);
                        break;
                }
                if (func == NULL)
                        quit("Invalid combination of implementation and value type, with blocking");

                clear_tmp_distances(db);
                func(db->dimensions, trdb->count, trdb->block_items, trdb->data,
                     trdb->klass, db->count, db->data, db->klass, db->distance);
        }
        else
        {
                /* Select unblocked versions */
                void (*func) (int, int, void *, int *, int, void *, int *);

                func = NULL;
                switch (type)
                {
                case BYTE:
                        func = (scalar ? nn_byte_sca_U : nn_byte_vec_U);
                        break;
                case SHORT:
                        func = (scalar ? nn_short_sca_U : nn_short_vec_U);
                        break;
                case INT:
                        func = (scalar ? nn_int_sca_U : nn_int_vec_U);
                        break;
                case FLOAT:
                        func = (scalar ? nn_float_sca_U : nn_float_vec_U);
                        break;
                case DOUBLE:
                        func = (scalar ? nn_double_sca_U : nn_double_vec_U);
                        break;
                }
                if (func == NULL)
                        quit("Invalid combination of implementation and value type, without blocking");

                func(db->dimensions, trdb->count, trdb->data, trdb->klass,
                     db->count, db->data, db->klass);
        }
}

