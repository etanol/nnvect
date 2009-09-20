#include "util.h"

#include <limits.h>
#include <float.h>

void nn_float_seq (int dimensions, int trcount, float *trdata, int *trklass,
                   int count, float *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        float min_distance, dist;
        float tmp;

        debug("Class\tEuclid\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = FLT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_seq (int dimensions, int trcount, double *trdata, int *trklass,
                    int count, double *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        double min_distance, dist;
        double tmp;

        debug("Class\tEuclid\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = DBL_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_byte_seq (int dimensions, int trcount, char *trdata, int *trklass,
                  int count, char *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        char min_distance, dist;
        char tmp, mask;

        debug("Class\tEuclid\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = CHAR_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                tmp = data[i + d] - trdata[i + d];
                                mask = tmp >> (sizeof(char) * CHAR_BIT - 1);
                                dist += (tmp ^ mask) - mask;
                        }
                        if (dist < min_distance)
                        {
                                min_distance = dist;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%d\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_short_seq (int dimensions, int trcount, short *trdata, int *trklass,
                   int count, short *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        short min_distance, dist;
        short tmp, mask;

        debug("Class\tEuclid\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = CHAR_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                tmp = data[i + d] - trdata[i + d];
                                mask = tmp >> (sizeof(short) * CHAR_BIT - 1);
                                dist += (tmp ^ mask) - mask;
                        }
                        if (dist < min_distance)
                        {
                                min_distance = dist;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%d\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_int_seq (int dimensions, int trcount, int *trdata, int *trklass,
                 int count, int *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        int min_distance, dist;
        int tmp, mask;

        debug("Class\tEuclid\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = CHAR_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                tmp = data[i + d] - trdata[i + d];
                                mask = tmp >> (sizeof(int) * CHAR_BIT - 1);
                                dist += (tmp ^ mask) - mask;
                        }
                        if (dist < min_distance)
                        {
                                min_distance = dist;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%d\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}

