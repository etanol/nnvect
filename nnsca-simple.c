#include "util.h"

#include <limits.h>
#include <float.h>
#include <math.h>

/* Block adjustment */
int adjusted_block_count (int bc)
{
        return bc;
}



/******************************************************************************/
/*                                                                            */
/*                             UNBLOCKED VERSIONS                             */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_sca_U (int dimensions, int trcount, char *trdata, int *trklass,
                    int count, char *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        char tmp;

        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        short tmp;

        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_int_sca_U (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        int tmp;

        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcount;  tn++)
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        float min_distance, dist;
        float tmp;

        debug("Class\tDist\tIndex ");
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


void nn_double_sca_U (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, idx;
        double min_distance, dist;
        double tmp;

        debug("Class\tDist\tIndex ");
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



/******************************************************************************/
/*                                                                            */
/*                              BLOCKED VERSIONS                              */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_sca_B (int dimensions, int trcount, int trblockcount, char *trdata,
                    int *trklass, int count, char *data, int *klass,
                    unsigned int *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        char tmp;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                        }
                        debug("%d\t%u\t%d ", cl, min_distance, idx);
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}


void nn_short_sca_B (int dimensions, int trcount, int trblockcount, short *trdata,
                     int *trklass, int count, short *data, int *klass,
                     unsigned int *distance)

{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        short tmp;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                        }
                        debug("%d\t%u\t%d ", cl, min_distance, idx);
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}


void nn_int_sca_B (int dimensions, int trcount, int trblockcount, int *trdata,
                   int *trklass, int count, int *data, int *klass,
                   unsigned int *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, dist;
        int tmp;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                        }
                        debug("%d\t%u\t%d ", cl, min_distance, idx);
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_sca_B (int dimensions, int trcount, int trblockcount, float *trdata,
                     int *trklass, int count, float *data, int *klass,
                     float *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        int cl, d, idx;
        float min_distance, dist;
        float tmp;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                        }
                        debug("%d\t%f\t%d ", cl, min_distance, idx);
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}


void nn_double_sca_B (int dimensions, int trcount, int trblockcount, double *trdata,
                      int *trklass, int count, double *data, int *klass,
                      double *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int cl, d, idx;
        double min_distance, dist;
        double tmp;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
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
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}

