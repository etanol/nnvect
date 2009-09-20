#include "util.h"

#include <limits.h>
#include <float.h>
#include <math.h>


/******************************************************************************/
/*                                                                            */
/*                             UNBLOCKED VERSIONS                             */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_sca_U (int dimensions, int trcount, char *trdata, int *trklass,
                    int count, char *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance;
        unsigned int dist1, dist2, dist3, dist4;
        char datum, tmp1, tmp2, tmp3, tmp4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 4)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                datum = data[i + d];
                                tmp1 = datum - trdata[ti1 + d];
                                tmp2 = datum - trdata[ti2 + d];
                                tmp3 = datum - trdata[ti3 + d];
                                tmp4 = datum - trdata[ti4 + d];
                                dist1 += tmp1 * tmp1;
                                dist2 += tmp2 * tmp2;
                                dist3 += tmp3 * tmp3;
                                dist4 += tmp4 * tmp4;
                        }
                        if (dist1 < min_distance)
                        {
                                min_distance = dist1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (dist2 < min_distance)
                        {
                                min_distance = dist2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (dist3 < min_distance)
                        {
                                min_distance = dist3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (dist4 < min_distance)
                        {
                                min_distance = dist4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_short_sca_U (int dimensions, int trcount, short *trdata, int *trklass,
                     int count, short *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance;
        unsigned int dist1, dist2, dist3, dist4;
        short datum, tmp1, tmp2, tmp3, tmp4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 4)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                datum = data[i + d];
                                tmp1 = datum - trdata[ti1 + d];
                                tmp2 = datum - trdata[ti2 + d];
                                tmp3 = datum - trdata[ti3 + d];
                                tmp4 = datum - trdata[ti4 + d];
                                dist1 += tmp1 * tmp1;
                                dist2 += tmp2 * tmp2;
                                dist3 += tmp3 * tmp3;
                                dist4 += tmp4 * tmp4;
                        }
                        if (dist1 < min_distance)
                        {
                                min_distance = dist1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (dist2 < min_distance)
                        {
                                min_distance = dist2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (dist3 < min_distance)
                        {
                                min_distance = dist3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (dist4 < min_distance)
                        {
                                min_distance = dist4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n]= cl;
        }
}


void nn_int_sca_U (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance;
        unsigned int dist1, dist2, dist3, dist4;
        int datum, tmp1, tmp2, tmp3, tmp4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 4)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = 0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                datum = data[i + d];
                                tmp1 = datum - trdata[ti1 + d];
                                tmp2 = datum - trdata[ti2 + d];
                                tmp3 = datum - trdata[ti3 + d];
                                tmp4 = datum - trdata[ti4 + d];
                                dist1 += tmp1 * tmp1;
                                dist2 += tmp2 * tmp2;
                                dist3 += tmp3 * tmp3;
                                dist4 += tmp4 * tmp4;
                        }
                        if (dist1 < min_distance)
                        {
                                min_distance = dist1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (dist2 < min_distance)
                        {
                                min_distance = dist2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (dist3 < min_distance)
                        {
                                min_distance = dist3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (dist4 < min_distance)
                        {
                                min_distance = dist4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_sca_U (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        float min_distance, datum;
        float dist1, dist2, dist3, dist4;
        float tmp1, tmp2, tmp3, tmp4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = FLT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 4)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = 0.0f;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                datum = data[i + d];
                                tmp1 = datum - trdata[ti1 + d];
                                tmp2 = datum - trdata[ti2 + d];
                                tmp3 = datum - trdata[ti3 + d];
                                tmp4 = datum - trdata[ti4 + d];
                                dist1 += tmp1 * tmp1;
                                dist2 += tmp2 * tmp2;
                                dist3 += tmp3 * tmp3;
                                dist4 += tmp4 * tmp4;
                        }
                        if (dist1 < min_distance)
                        {
                                min_distance = dist1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (dist2 < min_distance)
                        {
                                min_distance = dist2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (dist3 < min_distance)
                        {
                                min_distance = dist3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (dist4 < min_distance)
                        {
                                min_distance = dist4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_sca_U (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        double min_distance, datum;
        double dist1, dist2, dist3, dist4;
        double tmp1, tmp2, tmp3, tmp4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = DBL_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 4)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = 0.0;
                        for (d = 0;  d < dimensions;  d++)
                        {
                                datum = data[i + d];
                                tmp1 = datum - trdata[ti1 + d];
                                tmp2 = datum - trdata[ti2 + d];
                                tmp3 = datum - trdata[ti3 + d];
                                tmp4 = datum - trdata[ti4 + d];
                                dist1 += tmp1 * tmp1;
                                dist2 += tmp2 * tmp2;
                                dist3 += tmp3 * tmp3;
                                dist4 += tmp4 * tmp4;
                        }
                        if (dist1 < min_distance)
                        {
                                min_distance = dist1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (dist2 < min_distance)
                        {
                                min_distance = dist2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (dist3 < min_distance)
                        {
                                min_distance = dist3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (dist4 < min_distance)
                        {
                                min_distance = dist4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}

