#include "nn.h"
#include "db.h"
#include "util.h"

#include <stddef.h>

#define DECLARE(f)  extern void f (int, int, void *, int *, int, void *, int *)


/* Sequential versions defined at nnseq.c */
DECLARE(nn_byte_seq_E);
DECLARE(nn_byte_seq_M);
DECLARE(nn_short_seq_E);
DECLARE(nn_short_seq_M);
DECLARE(nn_int_seq_E);
DECLARE(nn_int_seq_M);
DECLARE(nn_float_seq_E);
DECLARE(nn_float_seq_M);
DECLARE(nn_double_seq_E);
DECLARE(nn_double_seq_M);

/* Vectorized versions defined at nnvect.c */
DECLARE(nn_byte_vect_M);
//DECLARE(nn_short_vect_E);
DECLARE(nn_short_vect_M);
DECLARE(nn_int_vect_E);
DECLARE(nn_int_vect_M);
DECLARE(nn_float_vect_E);
DECLARE(nn_float_vect_M);
DECLARE(nn_double_vect_E);
DECLARE(nn_double_vect_M);


void nn (enum valuetype type, enum distancekind kind, int sequential,
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
                        func = (sequential ? nn_byte_seq_E : NULL);
                        break;
                case SHORT:
                        func = (sequential ? nn_short_seq_E : NULL);
                        break;
                case INT:
                        func = (sequential ? nn_int_seq_E : nn_int_vect_E);
                        break;
                case FLOAT:
                        func = (sequential ? nn_float_seq_E : nn_float_vect_E);
                        break;
                case DOUBLE:
                        func = (sequential ? nn_double_seq_E : nn_double_vect_E);
                        break;
                }
                break;
        case MANHATTAN:
                switch (type)
                {
                case BYTE:
                        func = (sequential ? nn_byte_seq_M : nn_byte_vect_M);
                        break;
                case SHORT:
                        func = (sequential ? nn_short_seq_M : nn_short_vect_M);
                        break;
                case INT:
                        func = (sequential ? nn_int_seq_M : nn_int_vect_M);
                        break;
                case FLOAT:
                        func = (sequential ? nn_float_seq_M : nn_float_vect_M);
                        break;
                case DOUBLE:
                        func = (sequential ? nn_double_seq_M : nn_double_vect_M);
                        break;
                }
                break;
        }

        if (func == NULL)
                quit("Invalid combination of implementation, value type and distance kind");

        func(db->dimensions, trdb->count, trdb->data, trdb->klass, db->count,
             db->data, db->klass);
}

