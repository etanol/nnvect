#include "nn.h"
#include "db.h"
#include "util.h"

#include <stddef.h>

#define DECLARE(f)  extern void f (int, int, void *, int *, int, void *, int *)


/* Sequential versions defined at nnseq.c */
DECLARE(nn_byte_sca_E);
DECLARE(nn_byte_sca_M);
DECLARE(nn_short_sca_E);
DECLARE(nn_short_sca_M);
DECLARE(nn_int_sca_E);
DECLARE(nn_int_sca_M);
DECLARE(nn_float_sca_E);
DECLARE(nn_float_sca_M);
DECLARE(nn_double_sca_E);
DECLARE(nn_double_sca_M);

/* Vectorized versions defined at nnvect.c */
//DECLARE(nn_byte_vec_E);
DECLARE(nn_byte_vec_M);
DECLARE(nn_short_vec_E);
DECLARE(nn_short_vec_M);
DECLARE(nn_int_vec_E);
DECLARE(nn_int_vec_M);
DECLARE(nn_float_vec_E);
DECLARE(nn_float_vec_M);
DECLARE(nn_double_vec_E);
DECLARE(nn_double_vec_M);


void nn (enum valuetype type, enum distancekind kind, int scalar,
         struct db *trdb, struct db *db)
{
        void (*func) (int, int, void *, int *, int, void *, int *);

        func = NULL;
        switch (kind)
        {
        case EUCLIDEAN:
                switch (type)
                {
                case BYTE:
                        func = (scalar ? nn_byte_sca_E : NULL);
                        break;
                case SHORT:
                        func = (scalar ? nn_short_sca_E : nn_short_vec_E);
                        break;
                case INT:
                        func = (scalar ? nn_int_sca_E : nn_int_vec_E);
                        break;
                case FLOAT:
                        func = (scalar ? nn_float_sca_E : nn_float_vec_E);
                        break;
                case DOUBLE:
                        func = (scalar ? nn_double_sca_E : nn_double_vec_E);
                        break;
                }
                break;
        case MANHATTAN:
                switch (type)
                {
                case BYTE:
                        func = (scalar ? nn_byte_sca_M : nn_byte_vec_M);
                        break;
                case SHORT:
                        func = (scalar ? nn_short_sca_M : nn_short_vec_M);
                        break;
                case INT:
                        func = (scalar ? nn_int_sca_M : nn_int_vec_M);
                        break;
                case FLOAT:
                        func = (scalar ? nn_float_sca_M : nn_float_vec_M);
                        break;
                case DOUBLE:
                        func = (scalar ? nn_double_sca_M : nn_double_vec_M);
                        break;
                }
                break;
        }

        if (func == NULL)
                quit("Invalid combination of implementation, value type and distance kind");

        func(db->dimensions, trdb->count, trdb->data, trdb->klass, db->count,
             db->data, db->klass);
}

