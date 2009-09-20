#ifndef __nnvect_nn
#define __nnvect_nn

/*
 * Sequential versions located in nnseq.c
 */
void nn_byte_seq   (int, int, char *,      int *, int, char *,      int *);
void nn_short_seq  (int, int, short *,     int *, int, short *,     int *);
void nn_int_seq    (int, int, int *,       int *, int, int *,       int *);
void nn_float_seq  (int, int ,float *,     int *, int, float *,     int *);
void nn_double_seq (int, int, double *,    int *, int, double *,    int *);


/*
 * Vectorized versions located in nnvect.c
 */
void nn_byte_vect   (int, int, char *,      int *, int, char *,      int *);
void nn_short_vect  (int, int, short *,     int *, int, short *,     int *);
void nn_int_vect    (int, int, int *,       int *, int, int *,       int *);
void nn_float_vect  (int, int ,float *,     int *, int, float *,     int *);
void nn_double_vect (int, int, double *,    int *, int, double *,    int *);

#endif /* __nnvect_nn */
