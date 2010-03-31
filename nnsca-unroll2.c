#include "util.h"

#include <limits.h>
#include <float.h>
#include <math.h>
#include <omp.h>

/* Block adjustment */
int adjusted_block_count (int bc)
{
        return bc & ~0x01;
}


/******************************  INTEGER VALUES  ******************************/

void nn_byte_sca (int dimensions, int trcount, int trblockcount, char *trdata,
                  int *trklass, int count, int blockcount, char *data,
                  int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        int cl, d;
        unsigned int min_distance;
        unsigned int dist1, dist2;
        char datum, tmp1, tmp2;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        tbcU = tbc & ~0x01;
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti1, ti2, dist1, dist2, \
                                d, datum, tmp1, tmp2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                datum = data[i + d];
                                                tmp1 = datum - trdata[ti1 + d];
                                                tmp2 = datum - trdata[ti2 + d];
                                                dist1 += tmp1 * tmp1;
                                                dist2 += tmp2 * tmp2;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                        if (dist2 < min_distance)
                                        {
                                                min_distance = dist2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp1 = data[i + d] - trdata[ti1 + d];
                                                dist1 += tmp1 * tmp1;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        int cl, d;
        unsigned int min_distance;
        unsigned int dist1, dist2;
        short datum, tmp1, tmp2;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        tbcU = tbc & ~0x01;
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti1, ti2, dist1, dist2, \
                                d, datum, tmp1, tmp2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                datum = data[i + d];
                                                tmp1 = datum - trdata[ti1 + d];
                                                tmp2 = datum - trdata[ti2 + d];
                                                dist1 += tmp1 * tmp1;
                                                dist2 += tmp2 * tmp2;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                        if (dist2 < min_distance)
                                        {
                                                min_distance = dist2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp1 = data[i + d] - trdata[ti1 + d];
                                                dist1 += tmp1 * tmp1;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n]= cl;
                        }
                }
        }
}


void nn_int_sca (int dimensions, int trcount, int trblockcount, int *trdata,
                 int *trklass, int count, int blockcount, int *data,
                 int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        int cl, d;
        unsigned int min_distance;
        unsigned int dist1, dist2;
        int datum, tmp1, tmp2;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        tbcU = tbc & ~0x01;
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti1, ti2, dist1, dist2, \
                                d, datum, tmp1, tmp2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                datum = data[i + d];
                                                tmp1 = datum - trdata[ti1 + d];
                                                tmp2 = datum - trdata[ti2 + d];
                                                dist1 += tmp1 * tmp1;
                                                dist2 += tmp2 * tmp2;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                        if (dist2 < min_distance)
                                        {
                                                min_distance = dist2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = 0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp1 = data[i + d] - trdata[ti1 + d];
                                                dist1 += tmp1 * tmp1;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        int cl, d;
        float min_distance, datum;
        float dist1, dist2;
        float tmp1, tmp2;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        tbcU = tbc & ~0x01;
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti1, ti2, dist1, dist2, \
                                d, datum, tmp1, tmp2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = 0.0f;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                datum = data[i + d];
                                                tmp1 = datum - trdata[ti1 + d];
                                                tmp2 = datum - trdata[ti2 + d];
                                                dist1 += tmp1 * tmp1;
                                                dist2 += tmp2 * tmp2;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                        if (dist2 < min_distance)
                                        {
                                                min_distance = dist2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = 0.0f;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp1 = data[i + d] - trdata[ti1 + d];
                                                dist1 += tmp1 * tmp1;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        int cl, d;
        double min_distance, datum;
        double dist1, dist2;
        double tmp1, tmp2;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        tbcU = tbc & ~0x01;
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti1, ti2, dist1, dist2, \
                                d, datum, tmp1, tmp2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = 0.0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                datum = data[i + d];
                                                tmp1 = datum - trdata[ti1 + d];
                                                tmp2 = datum - trdata[ti2 + d];
                                                dist1 += tmp1 * tmp1;
                                                dist2 += tmp2 * tmp2;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                        if (dist2 < min_distance)
                                        {
                                                min_distance = dist2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = 0.0;
                                        for (d = 0;  d < dimensions;  d++)
                                        {
                                                tmp1 = data[i + d] - trdata[ti1 + d];
                                                dist1 += tmp1 * tmp1;
                                        }
                                        if (dist1 < min_distance)
                                        {
                                                min_distance = dist1;
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}

