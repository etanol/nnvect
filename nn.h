#ifndef __nnvect_nn
#define __nnvect_nn

/* Supported data types */
enum datatype
{
        BYTE = 1,
        SHORT,
        INTEGER,
        FLOAT,
        DOUBLE
};

/* Sequential versions located in nnseq.c */
void nn_byte_seq    (int, int, char *,   int *, int, char *,   int *);
void nn_short_seq   (int, int, short *,  int *, int, short *,  int *);
void nn_int_seq     (int, int, int *,    int *, int, int *,    int *);
void nn_float_seq   (int, int ,float *,  int *, int, float *,  int *);
void nn_double_seq  (int, int, double *, int *, int, double *, int *);

/* Vectorized versions located in nnvect.c */
void nn_byte_vect   (int, int, char *,   int *, int, char *,   int *);
void nn_short_vect  (int, int, short *,  int *, int, short *,  int *);
void nn_int_vect    (int, int, int *,    int *, int, int *,    int *);
void nn_float_vect  (int, int ,float *,  int *, int, float *,  int *);
void nn_double_vect (int, int, double *, int *, int, double *, int *);


/*
 * Sequential proxy function
 */
static inline void nn_seq (enum datatype type, int dims, int tcnt, void *tdata,
                           int *tcl, int cnt, void *data, int *cl)
{
        switch (type)
        {
        case BYTE:
                nn_byte_seq(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case SHORT:
                nn_short_seq(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case INTEGER:
                nn_int_seq(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case FLOAT:
                nn_float_seq(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case DOUBLE:
                nn_double_seq(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        }
}


/*
 * Vectorized proxy function
 */
static inline void nn_vect (enum datatype type, int dims, int tcnt, void *tdata,
                            int *tcl, int cnt, void *data, int *cl)
{
        switch (type)
        {
        case BYTE:
                nn_byte_vect(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case SHORT:
                nn_short_vect(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case INTEGER:
                nn_int_vect(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case FLOAT:
                nn_float_vect(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case DOUBLE:
                nn_double_vect(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        }
}

#endif /* __nnvect_nn */
