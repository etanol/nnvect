#include "util.h"

#include <smmintrin.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>


/******************************************************************************/
/*                                                                            */
/*                             UNBLOCKED VERSIONS                             */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_vec_U (int dimensions, int trcount, char *trdata, int *trklass,
                    int count, char *data, int *klass)
{
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        short sdist[8] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2, mask;
        __m128i dist;

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
                        dist = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 16)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                tmp1 = _mm_sub_epi8(vec, tvec);
                                mask = _mm_cmplt_epi8(tmp1, _mm_setzero_si128());
                                tmp1 = _mm_sub_epi8(_mm_xor_si128(tmp1, mask), mask);
                                tmp2 = _mm_maddubs_epi16(tmp1, tmp1);
                                dist = _mm_adds_epu16(dist, tmp2);
                        }
                        tmp1 = _mm_hadd_epi16(dist, dist);
                        tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                        dist = _mm_hadd_epi16(tmp2, tmp2);
                        _mm_store_si128((__m128i *) sdist, dist);
                        if (sdist[0] < min_distance)
                        {
                                min_distance = sdist[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_short_vec_U (int dimensions, int trcount, short *trdata, int *trklass,
                     int count, short *data, int *klass)
{
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

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
                        dist = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 8)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                tmp1 = _mm_sub_epi16(vec, tvec);
                                tmp2 = _mm_madd_epi16(tmp1, tmp1);
                                dist = _mm_add_epi32(dist, tmp2);
                        }
                        tmp1 = _mm_hadd_epi32(dist, dist);
                        dist = _mm_hadd_epi32(tmp1, tmp1);
                        _mm_store_si128((__m128i *) sdist, dist);
                        if (sdist[0] < min_distance)
                        {
                                min_distance = sdist[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_int_vec_U (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

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
                        dist = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                tmp1 = _mm_sub_epi32(vec, tvec);
                                tmp2 = _mm_mullo_epi32(tmp1, tmp1);
                                dist = _mm_add_epi32(dist, tmp2);
                        }
                        tmp1 = _mm_hadd_epi32(dist, dist);
                        dist = _mm_hadd_epi32(tmp1, tmp1);
                        _mm_store_si128((__m128i *) sdist, dist);
                        if (sdist[0] < min_distance)
                        {
                                min_distance = sdist[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_vec_U (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn;
        int i, ti;
        float min_distance;
        float sdist __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec;
        __m128 tmp1, tmp2;
        __m128 dist;

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
                        dist = _mm_setzero_ps();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_ps(&data[i + d]);
                                tvec = _mm_load_ps(&trdata[ti + d]);
                                tmp1 = _mm_sub_ps(vec, tvec);
                                tmp2 = _mm_mul_ps(tmp1, tmp1);
                                dist = _mm_add_ps(dist, tmp2);
                        }
                        tmp1 = _mm_hadd_ps(dist, dist);
                        dist = _mm_hadd_ps(tmp1, tmp1);
                        _mm_store_ss(&sdist, dist);
                        if (sdist < min_distance)
                        {
                                min_distance = sdist;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_vec_U (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn;
        int i, ti;
        double min_distance;
        double sdist[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec;
        __m128d tmp1, tmp2;
        __m128d dist;

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
                        dist = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec = _mm_load_pd(&trdata[ti + d]);
                                tmp1 = _mm_sub_pd(vec, tvec);
                                tmp2 = _mm_mul_pd(tmp1, tmp1);
                                dist = _mm_add_pd(dist, tmp2);
                        }
                        _mm_store_pd(sdist, dist);
                        if (sdist[0] + sdist[1] < min_distance)
                        {
                                min_distance = sdist[0] + sdist[1];
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

void nn_byte_vec_B (int dimensions, int trcount, int trblockcount, char *trdata,
                    int *trklass, int count, char *data, int *klass,
                    unsigned int *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        short sdist[8] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2, mask;
        __m128i dist;

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
                                dist = _mm_setzero_si128();
                                for (d = 0;  d < dimensions;  d += 16)
                                {
                                        vec = _mm_load_si128((__m128i *) &data[i + d]);
                                        tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                        tmp1 = _mm_sub_epi8(vec, tvec);
                                        mask = _mm_cmplt_epi8(tmp1, _mm_setzero_si128());
                                        tmp1 = _mm_sub_epi8(_mm_xor_si128(tmp1, mask), mask);
                                        tmp2 = _mm_maddubs_epi16(tmp1, tmp1);
                                        dist = _mm_adds_epu16(dist, tmp2);
                                }
                                tmp1 = _mm_hadd_epi16(dist, dist);
                                tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                                dist = _mm_hadd_epi16(tmp2, tmp2);
                                _mm_store_si128((__m128i *) sdist, dist);
                                if (sdist[0] < min_distance)
                                {
                                        min_distance = sdist[0];
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


void nn_short_vec_B (int dimensions, int trcount, int trblockcount, short *trdata,
                     int *trklass, int count, short *data, int *klass,
                     unsigned int *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

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
                                dist = _mm_setzero_si128();
                                for (d = 0;  d < dimensions;  d += 8)
                                {
                                        vec = _mm_load_si128((__m128i *) &data[i + d]);
                                        tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                        tmp1 = _mm_sub_epi16(vec, tvec);
                                        tmp2 = _mm_madd_epi16(tmp1, tmp1);
                                        dist = _mm_add_epi32(dist, tmp2);
                                }
                                tmp1 = _mm_hadd_epi32(dist, dist);
                                dist = _mm_hadd_epi32(tmp1, tmp1);
                                _mm_store_si128((__m128i *) sdist, dist);
                                if (sdist[0] < min_distance)
                                {
                                        min_distance = sdist[0];
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


void nn_int_vec_B (int dimensions, int trcount, int trblockcount, int *trdata,
                   int *trklass, int count, int *data, int *klass,
                   unsigned int *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

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
                                dist = _mm_setzero_si128();
                                for (d = 0;  d < dimensions;  d += 4)
                                {
                                        vec = _mm_load_si128((__m128i *) &data[i + d]);
                                        tvec = _mm_load_si128((__m128i *) &trdata[ti + d]);
                                        tmp1 = _mm_sub_epi32(vec, tvec);
                                        tmp2 = _mm_mullo_epi32(tmp1, tmp1);
                                        dist = _mm_add_epi32(dist, tmp2);
                                }
                                tmp1 = _mm_hadd_epi32(dist, dist);
                                dist = _mm_hadd_epi32(tmp1, tmp1);
                                _mm_store_si128((__m128i *) sdist, dist);
                                if (sdist[0] < min_distance)
                                {
                                        min_distance = sdist[0];
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

void nn_float_vec_B (int dimensions, int trcount, int trblockcount, float *trdata,
                     int *trklass, int count, float *data, int *klass,
                     float *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        float min_distance;
        float sdist __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec;
        __m128 tmp1, tmp2;
        __m128 dist;

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
                                dist = _mm_setzero_ps();
                                for (d = 0;  d < dimensions;  d += 4)
                                {
                                        vec = _mm_load_ps(&data[i + d]);
                                        tvec = _mm_load_ps(&trdata[ti + d]);
                                        tmp1 = _mm_sub_ps(vec, tvec);
                                        tmp2 = _mm_mul_ps(tmp1, tmp1);
                                        dist = _mm_add_ps(dist, tmp2);
                                }
                                tmp1 = _mm_hadd_ps(dist, dist);
                                dist = _mm_hadd_ps(tmp1, tmp1);
                                _mm_store_ss(&sdist, dist);
                                if (sdist < min_distance)
                                {
                                        min_distance = sdist;
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


void nn_double_vec_B (int dimensions, int trcount, int trblockcount, double *trdata,
                      int *trklass, int count, double *data, int *klass,
                      double *distance)
{
        int tbn, tbc;
        int n, tn;
        int i, ti;
        double min_distance;
        double sdist[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec;
        __m128d tmp1, tmp2;
        __m128d dist;

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
                                dist = _mm_setzero_pd();
                                for (d = 0;  d < dimensions;  d += 2)
                                {
                                        vec = _mm_load_pd(&data[i + d]);
                                        tvec = _mm_load_pd(&trdata[ti + d]);
                                        tmp1 = _mm_sub_pd(vec, tvec);
                                        tmp2 = _mm_mul_pd(tmp1, tmp1);
                                        dist = _mm_add_pd(dist, tmp2);
                                }
                                _mm_store_pd(sdist, dist);
                                if (sdist[0] + sdist[1] < min_distance)
                                {
                                        min_distance = sdist[0] + sdist[1];
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

