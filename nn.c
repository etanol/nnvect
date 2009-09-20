#include "nn.h"
#include "db.h"
#include "util.h"

#include <stddef.h>

#define DECLARE(f)  extern void f (int, int, void *, int *, int, void *, int *)


/* Sequential versions defined at nnseq.c */
DECLARE(nn_byte_sca_U);
DECLARE(nn_short_sca_U);
DECLARE(nn_int_sca_U);
DECLARE(nn_float_sca_U);
DECLARE(nn_double_sca_U);

/* Vectorized versions defined at nnvect.c */
DECLARE(nn_byte_vec_U);
DECLARE(nn_short_vec_U);
DECLARE(nn_int_vec_U);
DECLARE(nn_float_vec_U);
DECLARE(nn_double_vec_U);


void nn (enum valuetype type, int scalar, struct db *trdb, struct db *db)
{
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
                quit("Invalid combination of implementation, value type and distance kind");

        func(db->dimensions, trdb->count, trdb->data, trdb->klass, db->count,
             db->data, db->klass);
}

