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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i mask1, mask2;
        __m128i dist1, dist2;

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
                                d, vec, tvec1, tvec2, tmp1, tmp2, mask1, \
                                mask2, sdist1, sdist2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 16)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                                tmp1 = _mm_sub_epi8(vec, tvec1);
                                                tmp2 = _mm_sub_epi8(vec, tvec2);
                                                mask1 = _mm_cmplt_epi8(tmp1, _mm_setzero_si128());
                                                mask2 = _mm_cmplt_epi8(tmp2, _mm_setzero_si128());
                                                tmp1 = _mm_sub_epi8(_mm_xor_si128(tmp1, mask1), mask1);
                                                tmp2 = _mm_sub_epi8(_mm_xor_si128(tmp2, mask2), mask2);
                                                tmp1 = _mm_maddubs_epi16(tmp1, tmp1);
                                                tmp2 = _mm_maddubs_epi16(tmp2, tmp2);
                                                dist1 = _mm_adds_epu16(dist1, tmp1);
                                                dist2 = _mm_adds_epu16(dist2, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi16(dist1, dist1);
                                        tmp2 = _mm_hadd_epi16(dist2, dist2);
                                        tmp1 = _mm_hadd_epi16(tmp1, tmp1);
                                        tmp2 = _mm_hadd_epi16(tmp2, tmp2);
                                        dist1 = _mm_hadd_epi16(tmp1, tmp1);
                                        dist2 = _mm_hadd_epi16(tmp2, tmp2);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        _mm_store_si128((__m128i *) sdist2, dist2);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
                                                cl = trklass[tn];
                                        }
                                        if (sdist2[0] < min_distance)
                                        {
                                                min_distance = sdist2[0];
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 16)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tmp1 = _mm_sub_epi8(vec, tvec1);
                                                mask1 = _mm_cmplt_epi8(tmp1, _mm_setzero_si128());
                                                tmp1 = _mm_sub_epi8(_mm_xor_si128(tmp1, mask1), mask1);
                                                tmp2 = _mm_maddubs_epi16(tmp1, tmp1);
                                                dist1 = _mm_adds_epu16(dist1, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi16(dist1, dist1);
                                        tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                                        dist1 = _mm_hadd_epi16(tmp2, tmp2);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
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

        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

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
                                d, vec, tvec1, tvec2, tmp1, tmp2, sdist1, \
                                sdist2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 8)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                                tmp1 = _mm_sub_epi16(vec, tvec1);
                                                tmp2 = _mm_sub_epi16(vec, tvec2);
                                                tmp1 = _mm_madd_epi16(tmp1, tmp1);
                                                tmp2 = _mm_madd_epi16(tmp2, tmp2);
                                                dist1 = _mm_add_epi32(dist1, tmp1);
                                                dist2 = _mm_add_epi32(dist2, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                                        tmp2 = _mm_hadd_epi32(dist2, dist2);
                                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                                        dist2 = _mm_hadd_epi32(tmp2, tmp2);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        _mm_store_si128((__m128i *) sdist2, dist2);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
                                                cl = trklass[tn];
                                        }
                                        if (sdist2[0] < min_distance)
                                        {
                                                min_distance = sdist2[0];
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 8)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tmp1 = _mm_sub_epi16(vec, tvec1);
                                                tmp2 = _mm_madd_epi16(tmp1, tmp1);
                                                dist1 = _mm_add_epi32(dist1, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

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
                                d, vec, tvec1, tvec2, tmp1, tmp2, sdist1, \
                                sdist2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 4)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                                tmp1 = _mm_sub_epi32(vec, tvec1);
                                                tmp2 = _mm_sub_epi32(vec, tvec2);
                                                tmp1 = _mm_mullo_epi32(tmp1, tmp1);
                                                tmp2 = _mm_mullo_epi32(tmp2, tmp2);
                                                dist1 = _mm_add_epi32(dist1, tmp1);
                                                dist2 = _mm_add_epi32(dist2, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                                        tmp2 = _mm_hadd_epi32(dist2, dist2);
                                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                                        dist2 = _mm_hadd_epi32(tmp2, tmp2);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        _mm_store_si128((__m128i *) sdist2, dist2);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
                                                cl = trklass[tn];
                                        }
                                        if (sdist2[0] < min_distance)
                                        {
                                                min_distance = sdist2[0];
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = _mm_setzero_si128();
                                        for (d = 0;  d < dimensions;  d += 4)
                                        {
                                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                                tmp1 = _mm_sub_epi32(vec, tvec1);
                                                tmp2 = _mm_mullo_epi32(tmp1, tmp1);
                                                dist1 = _mm_add_epi32(dist1, tmp2);
                                        }
                                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                                        _mm_store_si128((__m128i *) sdist1, dist1);
                                        if (sdist1[0] < min_distance)
                                        {
                                                min_distance = sdist1[0];
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        float min_distance;
        float distance1 __attribute__((aligned(16)));
        float distance2 __attribute__((aligned(16)));
        int cl, d;
        __m128 vec, tvec1, tvec2;
        __m128 tmp1, tmp2;
        __m128 dist1, dist2;

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
                                d, vec, tvec1, tvec2, tmp1, tmp2, distance1, \
                                distance2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = _mm_setzero_ps();
                                        for (d = 0;  d < dimensions;  d += 4)
                                        {
                                                vec = _mm_load_ps(&data[i + d]);
                                                tvec1 = _mm_load_ps(&trdata[ti1 + d]);
                                                tvec2 = _mm_load_ps(&trdata[ti2 + d]);
                                                tmp1 = _mm_sub_ps(vec, tvec1);
                                                tmp2 = _mm_sub_ps(vec, tvec2);
                                                tmp1 = _mm_mul_ps(tmp1, tmp1);
                                                tmp2 = _mm_mul_ps(tmp2, tmp2);
                                                dist1 = _mm_add_ps(dist1, tmp1);
                                                dist2 = _mm_add_ps(dist2, tmp2);
                                        }
                                        tmp1 = _mm_hadd_ps(dist1, dist1);
                                        tmp2 = _mm_hadd_ps(dist2, dist2);
                                        dist1 = _mm_hadd_ps(tmp1, tmp1);
                                        dist2 = _mm_hadd_ps(tmp2, tmp2);
                                        _mm_store_ss(&distance1, dist1);
                                        _mm_store_ss(&distance2, dist2);
                                        if (distance1 < min_distance)
                                        {
                                                min_distance = distance1;
                                                cl = trklass[tn];
                                        }
                                        if (distance2 < min_distance)
                                        {
                                                min_distance = distance2;
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = _mm_setzero_ps();
                                        for (d = 0;  d < dimensions;  d += 4)
                                        {
                                                vec = _mm_load_ps(&data[i + d]);
                                                tvec1 = _mm_load_ps(&trdata[ti1 + d]);
                                                tmp1 = _mm_sub_ps(vec, tvec1);
                                                tmp2 = _mm_mul_ps(tmp1, tmp1);
                                                dist1 = _mm_add_ps(dist1, tmp2);
                                        }
                                        tmp1 = _mm_hadd_ps(dist1, dist1);
                                        dist1 = _mm_hadd_ps(tmp1, tmp1);
                                        _mm_store_ss(&distance1, dist1);
                                        if (distance1 < min_distance)
                                        {
                                                min_distance = distance1;
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
        int bc, bn, tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        double min_distance;
        double sdist1[2] __attribute__((aligned(16)));
        double sdist2[2] __attribute__((aligned(16)));
        int cl, d;
        __m128d vec, tvec1, tvec2;
        __m128d tmp1, tmp2;
        __m128d dist1, dist2;

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
                                d, vec, tvec1, tvec2, tmp1, tmp2, sdist1, \
                                sdist2)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                min_distance = distance[n];
                                cl = klass[n];
                                for (tn = tbn;  tn < tbcU;  tn += 2)
                                {
                                        ti1 = tn * dimensions;
                                        ti2 = (tn + 1) * dimensions;
                                        dist1 = dist2 = _mm_setzero_pd();
                                        for (d = 0;  d < dimensions;  d += 2)
                                        {
                                                vec = _mm_load_pd(&data[i + d]);
                                                tvec1 = _mm_load_pd(&trdata[ti1 + d]);
                                                tvec2 = _mm_load_pd(&trdata[ti2 + d]);
                                                tmp1 = _mm_sub_pd(vec, tvec1);
                                                tmp2 = _mm_sub_pd(vec, tvec2);
                                                tmp1 = _mm_mul_pd(tmp1, tmp1);
                                                tmp2 = _mm_mul_pd(tmp2, tmp2);
                                                dist1 = _mm_add_pd(dist1, tmp1);
                                                dist2 = _mm_add_pd(dist2, tmp2);
                                        }
                                        _mm_store_pd(sdist1, dist1);
                                        _mm_store_pd(sdist2, dist2);
                                        if (sdist1[0] + sdist1[1] < min_distance)
                                        {
                                                min_distance = sdist1[0] + sdist1[1];
                                                cl = trklass[tn];
                                        }
                                        if (sdist2[0] + sdist2[1] < min_distance)
                                        {
                                                min_distance = sdist2[0] + sdist2[1];
                                                cl = trklass[tn + 1];
                                        }
                                }
                                for (;  tn < tbc;  tn++)
                                {
                                        ti1 = tn * dimensions;
                                        dist1 = _mm_setzero_pd();
                                        for (d = 0;  d < dimensions;  d += 2)
                                        {
                                                vec = _mm_load_pd(&data[i + d]);
                                                tvec1 = _mm_load_pd(&trdata[ti1 + d]);
                                                tmp1 = _mm_sub_pd(vec, tvec1);
                                                tmp2 = _mm_mul_pd(tmp1, tmp1);
                                                dist1 = _mm_add_pd(dist1, tmp2);
                                        }
                                        _mm_store_pd(sdist1, dist1);
                                        if (sdist1[0] + sdist1[1] < min_distance)
                                        {
                                                min_distance = sdist1[0] + sdist1[1];
                                                cl = trklass[tn];
                                        }
                                }
                                distance[n] = min_distance;
                                klass[n] = cl;
                        }
                }
        }
}

