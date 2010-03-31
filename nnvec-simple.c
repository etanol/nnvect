#include "util.h"

#ifdef NO_SSE4
#  include <tmmintrin.h>
#else
#  include <smmintrin.h>
#endif
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <omp.h>


/******************************  INTEGER VALUES  ******************************/

void nn_byte_vec (int dimensions, int trcount, int trblockcount, char *trdata,
                  int *trklass, int count, int blockcount, char *data,
                  int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        short sdist[8] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec;
        __m128i tmp1, tmp2, mask;
        __m128i dist;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, vec, tvec, \
                                tmp1, mask, tmp2, sdist)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
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
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


void nn_short_vec (int dimensions, int trcount, int trblockcount, short *trdata,
                   int *trklass, int count, int blockcount, short *data,
                   int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, vec, tvec, \
                                tmp1, tmp2, sdist)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
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
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


#ifndef NO_SSE4
void nn_int_vec (int dimensions, int trcount, int trblockcount, int *trdata,
                 int *trklass, int count, int blockcount, int *data,
                 int *klass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        unsigned int min_distance;
        int sdist[4] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, vec, tvec, \
                                tmp1, tmp2, sdist)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
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
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}
#endif


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_vec (int dimensions, int trcount, int trblockcount, float *trdata,
                   int *trklass, int count, int blockcount, float *data,
                   int *klass, float *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        float min_distance;
        float sdist __attribute__((aligned(16)));
        int cl, d;
        __m128 vec, tvec;
        __m128 tmp1, tmp2;
        __m128 dist;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, vec, tvec, \
                                tmp1, tmp2, sdist)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
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
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}


void nn_double_vec (int dimensions, int trcount, int trblockcount, double *trdata,
                    int *trklass, int count, int blockcount, double *data,
                    int *klass, double *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        double min_distance;
        double sdist[2] __attribute__((aligned(16)));
        int cl, d;
        __m128d vec, tvec;
        __m128d tmp1, tmp2;
        __m128d dist;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule(static) private(i, \
                                min_distance, cl, tn, ti, dist, d, vec, tvec, \
                                tmp1, tmp2, sdist)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
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
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}

