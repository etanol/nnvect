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
void nn_byte_seq_E    (int, int, char *,   int *, int, char *,   int *);
void nn_short_seq_E   (int, int, short *,  int *, int, short *,  int *);
void nn_int_seq_E     (int, int, int *,    int *, int, int *,    int *);
void nn_float_seq_E   (int, int ,float *,  int *, int, float *,  int *);
void nn_double_seq_E  (int, int, double *, int *, int, double *, int *);
void nn_byte_seq_M    (int, int, char *,   int *, int, char *,   int *);
void nn_short_seq_M   (int, int, short *,  int *, int, short *,  int *);
void nn_int_seq_M     (int, int, int *,    int *, int, int *,    int *);

/* Vectorized versions located in nnvect.c */
void nn_byte_vect_M   (int, int, char *,   int *, int, char *,   int *);
void nn_short_vect_M  (int, int, short *,  int *, int, short *,  int *);
void nn_int_vect_M    (int, int, int *,    int *, int, int *,    int *);
void nn_float_vect_E  (int, int ,float *,  int *, int, float *,  int *);
void nn_double_vect_E (int, int, double *, int *, int, double *, int *);


/*
 * Sequential proxy function
 */
static inline void nn_seq (enum datatype type, int dims, int tcnt, void *tdata,
                           int *tcl, int cnt, void *data, int *cl)
{
        switch (type)
        {
        case BYTE:
                nn_byte_seq_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case SHORT:
                nn_short_seq_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case INTEGER:
                nn_int_seq_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case FLOAT:
                nn_float_seq_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case DOUBLE:
                nn_double_seq_E(dims, tcnt, tdata, tcl, cnt, data, cl);
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
                nn_byte_vect_M(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case SHORT:
                nn_short_vect_M(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case INTEGER:
                nn_int_vect_M(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case FLOAT:
                nn_float_vect_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        case DOUBLE:
                nn_double_vect_E(dims, tcnt, tdata, tcl, cnt, data, cl);
                break;
        }
}

#endif /* __nnvect_nn */
