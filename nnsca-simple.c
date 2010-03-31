#include "util.h"

#include <limits.h>
#include <float.h>
#include <math.h>
#include <omp.h>

/* Block adjustment */
int adjusted_block_count (int bc)
{
        return bc;
}


/******************************  INTEGER VALUES  ******************************/

void nn_byte_sca (int dimensions, int trcount, int trblockcount, char *trdata,
                  int *trklass, int count, int blockcount, char *data,
                  int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d;
        unsigned int min_distance, dist;
        char tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, tmp)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < min_distance)
                                        {
                                                min_distance = dist;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


void nn_short_sca (int dimensions, int trcount, int trblockcount, short *trdata,
                   int *trklass, int count, int blockcount, short *data,
                   int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d;
        unsigned int min_distance, dist;
        short tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, tmp)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < min_distance)
                                        {
                                                min_distance = dist;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


void nn_int_sca (int dimensions, int trcount, int trblockcount, int *trdata,
                 int *trklass, int count, int blockcount, int *data,
                 int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d;
        unsigned int min_distance, dist;
        int tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, tmp)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < min_distance)
                                        {
                                                min_distance = dist;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_sca (int dimensions, int trcount, int trblockcount, float *trdata,
                   int *trklass, int count, int blockcount, float *data,
                   int *klass, float *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d;
        float min_distance, dist;
        float tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, tmp)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0.0f;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < min_distance)
                                        {
                                                min_distance = dist;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


void nn_double_sca (int dimensions, int trcount, int trblockcount, double *trdata,
                    int *trklass, int count, int blockcount, double *data,
                    int *klass, double *distance)
{
        int bn, bc, tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d;
        double min_distance, dist;
        double tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, tmp)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0.0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < min_distance)
                                        {
                                                min_distance = dist;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}

