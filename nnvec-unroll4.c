#include "util.h"

#include <smmintrin.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>


/*****************************************************************************/
/*                                                                           */
/*                        EUCLIDEAN DISTANCE VERSIONS                        */
/*                                                                           */
/*****************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_vec_E (int dimensions, int trcount, char *trdata, int *trklass,
                    int count, char *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        unsigned int min_distance;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        short sdist3[8] __attribute__((aligned(16)));
        short sdist4[8] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i mask1, mask2, mask3, mask4;
        __m128i dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 16)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sub_epi8(vec, tvec1);
                                tmp2 = _mm_sub_epi8(vec, tvec2);
                                tmp3 = _mm_sub_epi8(vec, tvec3);
                                tmp4 = _mm_sub_epi8(vec, tvec4);
                                mask1 = _mm_cmplt_epi8(tmp1, _mm_setzero_si128());
                                mask2 = _mm_cmplt_epi8(tmp2, _mm_setzero_si128());
                                mask3 = _mm_cmplt_epi8(tmp3, _mm_setzero_si128());
                                mask4 = _mm_cmplt_epi8(tmp4, _mm_setzero_si128());
                                tmp1 = _mm_sub_epi8(_mm_xor_si128(tmp1, mask1), mask1);
                                tmp2 = _mm_sub_epi8(_mm_xor_si128(tmp2, mask2), mask2);
                                tmp3 = _mm_sub_epi8(_mm_xor_si128(tmp3, mask3), mask3);
                                tmp4 = _mm_sub_epi8(_mm_xor_si128(tmp4, mask4), mask4);
                                tmp1 = _mm_maddubs_epi16(tmp1, tmp1);
                                tmp2 = _mm_maddubs_epi16(tmp2, tmp2);
                                tmp3 = _mm_maddubs_epi16(tmp3, tmp3);
                                tmp4 = _mm_maddubs_epi16(tmp4, tmp4);
                                dist1 = _mm_adds_epu16(dist1, tmp1);
                                dist2 = _mm_adds_epu16(dist2, tmp2);
                                dist3 = _mm_adds_epu16(dist3, tmp3);
                                dist4 = _mm_adds_epu16(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_epi16(dist1, dist1);
                        tmp2 = _mm_hadd_epi16(dist2, dist2);
                        tmp3 = _mm_hadd_epi16(dist3, dist3);
                        tmp4 = _mm_hadd_epi16(dist4, dist4);
                        tmp1 = _mm_hadd_epi16(tmp1, tmp1);
                        tmp2 = _mm_hadd_epi16(tmp2, tmp2);
                        tmp3 = _mm_hadd_epi16(tmp3, tmp3);
                        tmp4 = _mm_hadd_epi16(tmp4, tmp4);
                        dist1 = _mm_hadd_epi16(tmp1, tmp1);
                        dist2 = _mm_hadd_epi16(tmp2, tmp2);
                        dist3 = _mm_hadd_epi16(tmp3, tmp3);
                        dist4 = _mm_hadd_epi16(tmp4, tmp4);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] < min_distance)
                        {
                                min_distance = sdist3[0];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] < min_distance)
                        {
                                min_distance = sdist4[0];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_short_vec_E (int dimensions, int trcount, short *trdata, int *trklass,
                     int count, short *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int sdist3[4] __attribute__((aligned(16)));
        int sdist4[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 8)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sub_epi16(vec, tvec1);
                                tmp2 = _mm_sub_epi16(vec, tvec2);
                                tmp3 = _mm_sub_epi16(vec, tvec3);
                                tmp4 = _mm_sub_epi16(vec, tvec4);
                                tmp1 = _mm_madd_epi16(tmp1, tmp1);
                                tmp2 = _mm_madd_epi16(tmp2, tmp2);
                                tmp3 = _mm_madd_epi16(tmp3, tmp3);
                                tmp4 = _mm_madd_epi16(tmp4, tmp4);
                                dist1 = _mm_add_epi32(dist1, tmp1);
                                dist2 = _mm_add_epi32(dist2, tmp2);
                                dist3 = _mm_add_epi32(dist3, tmp3);
                                dist4 = _mm_add_epi32(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                        tmp2 = _mm_hadd_epi32(dist2, dist2);
                        tmp3 = _mm_hadd_epi32(dist3, dist3);
                        tmp4 = _mm_hadd_epi32(dist4, dist4);
                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                        dist2 = _mm_hadd_epi32(tmp2, tmp2);
                        dist3 = _mm_hadd_epi32(tmp3, tmp3);
                        dist4 = _mm_hadd_epi32(tmp4, tmp4);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] < min_distance)
                        {
                                min_distance = sdist3[0];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] < min_distance)
                        {
                                min_distance = sdist4[0];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_int_vec_E (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int sdist3[4] __attribute__((aligned(16)));
        int sdist4[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sub_epi32(vec, tvec1);
                                tmp2 = _mm_sub_epi32(vec, tvec2);
                                tmp3 = _mm_sub_epi32(vec, tvec3);
                                tmp4 = _mm_sub_epi32(vec, tvec4);
                                tmp1 = _mm_mullo_epi32(tmp1, tmp1);
                                tmp2 = _mm_mullo_epi32(tmp2, tmp2);
                                tmp3 = _mm_mullo_epi32(tmp3, tmp3);
                                tmp4 = _mm_mullo_epi32(tmp4, tmp4);
                                dist1 = _mm_add_epi32(dist1, tmp1);
                                dist2 = _mm_add_epi32(dist2, tmp2);
                                dist3 = _mm_add_epi32(dist3, tmp3);
                                dist4 = _mm_add_epi32(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                        tmp2 = _mm_hadd_epi32(dist2, dist2);
                        tmp3 = _mm_hadd_epi32(dist3, dist3);
                        tmp4 = _mm_hadd_epi32(dist4, dist4);
                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                        dist2 = _mm_hadd_epi32(tmp2, tmp2);
                        dist3 = _mm_hadd_epi32(tmp3, tmp3);
                        dist4 = _mm_hadd_epi32(tmp4, tmp4);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] < min_distance)
                        {
                                min_distance = sdist3[0];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] < min_distance)
                        {
                                min_distance = sdist4[0];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_vec_E (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        float min_distance;
        float distance1 __attribute__((aligned(16)));
        float distance2 __attribute__((aligned(16)));
        float distance3 __attribute__((aligned(16)));
        float distance4 __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec1, tvec2, tvec3, tvec4;
        __m128 tmp1, tmp2, tmp3, tmp4;
        __m128 dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_ps();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_ps(&data[i + d]);
                                tvec1 = _mm_load_ps(&trdata[ti1 + d]);
                                tvec2 = _mm_load_ps(&trdata[ti2 + d]);
                                tvec3 = _mm_load_ps(&trdata[ti3 + d]);
                                tvec4 = _mm_load_ps(&trdata[ti4 + d]);
                                tmp1 = _mm_sub_ps(vec, tvec1);
                                tmp2 = _mm_sub_ps(vec, tvec2);
                                tmp3 = _mm_sub_ps(vec, tvec3);
                                tmp4 = _mm_sub_ps(vec, tvec4);
                                tmp1 = _mm_mul_ps(tmp1, tmp1);
                                tmp2 = _mm_mul_ps(tmp2, tmp2);
                                tmp3 = _mm_mul_ps(tmp3, tmp3);
                                tmp4 = _mm_mul_ps(tmp4, tmp4);
                                dist1 = _mm_add_ps(dist1, tmp1);
                                dist2 = _mm_add_ps(dist2, tmp2);
                                dist3 = _mm_add_ps(dist3, tmp3);
                                dist4 = _mm_add_ps(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_ps(dist1, dist1);
                        tmp2 = _mm_hadd_ps(dist2, dist2);
                        tmp3 = _mm_hadd_ps(dist3, dist3);
                        tmp4 = _mm_hadd_ps(dist4, dist4);
                        dist1 = _mm_hadd_ps(tmp1, tmp1);
                        dist2 = _mm_hadd_ps(tmp2, tmp2);
                        dist3 = _mm_hadd_ps(tmp3, tmp3);
                        dist4 = _mm_hadd_ps(tmp4, tmp4);
                        _mm_store_ss(&distance1, dist1);
                        _mm_store_ss(&distance2, dist2);
                        _mm_store_ss(&distance3, dist3);
                        _mm_store_ss(&distance4, dist4);
                        if (distance1 < min_distance)
                        {
                                min_distance = distance1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (distance2 < min_distance)
                        {
                                min_distance = distance2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (distance3 < min_distance)
                        {
                                min_distance = distance3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (distance4 < min_distance)
                        {
                                min_distance = distance4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_vec_E (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        double min_distance;
        double sdist1[2] __attribute__((aligned(16)));
        double sdist2[2] __attribute__((aligned(16)));
        double sdist3[2] __attribute__((aligned(16)));
        double sdist4[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec1, tvec2, tvec3, tvec4;
        __m128d tmp1, tmp2, tmp3, tmp4;
        __m128d dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec1 = _mm_load_pd(&trdata[ti1 + d]);
                                tvec2 = _mm_load_pd(&trdata[ti2 + d]);
                                tvec3 = _mm_load_pd(&trdata[ti3 + d]);
                                tvec4 = _mm_load_pd(&trdata[ti4 + d]);
                                tmp1 = _mm_sub_pd(vec, tvec1);
                                tmp2 = _mm_sub_pd(vec, tvec2);
                                tmp3 = _mm_sub_pd(vec, tvec3);
                                tmp4 = _mm_sub_pd(vec, tvec4);
                                tmp1 = _mm_mul_pd(tmp1, tmp1);
                                tmp2 = _mm_mul_pd(tmp2, tmp2);
                                tmp3 = _mm_mul_pd(tmp3, tmp3);
                                tmp4 = _mm_mul_pd(tmp4, tmp4);
                                dist1 = _mm_add_pd(dist1, tmp1);
                                dist2 = _mm_add_pd(dist2, tmp2);
                                dist3 = _mm_add_pd(dist3, tmp3);
                                dist4 = _mm_add_pd(dist4, tmp4);
                        }
                        _mm_store_pd(sdist1, dist1);
                        _mm_store_pd(sdist2, dist2);
                        _mm_store_pd(sdist3, dist3);
                        _mm_store_pd(sdist4, dist4);
                        if (sdist1[0] + sdist1[1] < min_distance)
                        {
                                min_distance = sdist1[0] + sdist1[1];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] + sdist2[1] < min_distance)
                        {
                                min_distance = sdist2[0] + sdist2[1];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] + sdist3[1] < min_distance)
                        {
                                min_distance = sdist3[0] + sdist3[1];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] + sdist4[1] < min_distance)
                        {
                                min_distance = sdist4[0] + sdist4[1];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
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
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}



/*****************************************************************************/
/*                                                                           */
/*                        MANHATTAN DISTANCE VERSIONS                        */
/*                                                                           */
/*****************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void nn_byte_vec_M (int dimensions, int trcount, char *trdata, int *trklass,
                    int count, char *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance ;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        short sdist3[8] __attribute__((aligned(16)));
        short sdist4[8] __attribute__((aligned(16)));
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 16)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sad_epu8(vec, tvec1);
                                tmp2 = _mm_sad_epu8(vec, tvec2);
                                tmp3 = _mm_sad_epu8(vec, tvec3);
                                tmp4 = _mm_sad_epu8(vec, tvec4);
                                dist1 = _mm_adds_epi16(dist1, tmp1);
                                dist2 = _mm_adds_epi16(dist2, tmp2);
                                dist3 = _mm_adds_epi16(dist3, tmp3);
                                dist4 = _mm_adds_epi16(dist4, tmp4);
                        }
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] + sdist1[4] < min_distance)
                        {
                                min_distance = sdist1[0] + sdist1[4];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] + sdist2[4] < min_distance)
                        {
                                min_distance = sdist2[0] + sdist2[4];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] + sdist3[4] < min_distance)
                        {
                                min_distance = sdist3[0] + sdist3[4];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] + sdist4[4] < min_distance)
                        {
                                min_distance = sdist4[0] + sdist4[4];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
                {
                        ti1 = tn * dimensions;
                        dist1 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 16)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tmp1 = _mm_sad_epu8(vec, tvec1);
                                dist1 = _mm_adds_epi16(dist1, tmp1);
                        }
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        if (sdist1[0] + sdist1[4] < min_distance)
                        {
                                min_distance = sdist1[0] + sdist1[4];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_short_vec_M (int dimensions, int trcount, short *trdata, int *trklass,
                     int count, short *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        short sdist3[8] __attribute__((aligned(16)));
        short sdist4[8] __attribute__((aligned(16)));
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i mask1, mask2, mask3, mask4;
        __m128i dist1, dist2, dist3, dist4;

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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 8)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sub_epi16(vec, tvec1);
                                tmp2 = _mm_sub_epi16(vec, tvec2);
                                tmp3 = _mm_sub_epi16(vec, tvec3);
                                tmp4 = _mm_sub_epi16(vec, tvec4);
                                mask1 = _mm_srai_epi16(tmp1, 15);
                                mask2 = _mm_srai_epi16(tmp2, 15);
                                mask3 = _mm_srai_epi16(tmp3, 15);
                                mask4 = _mm_srai_epi16(tmp4, 15);
                                tmp1 = _mm_sub_epi16(_mm_xor_si128(tmp1, mask1), mask1);
                                tmp2 = _mm_sub_epi16(_mm_xor_si128(tmp2, mask2), mask2);
                                tmp3 = _mm_sub_epi16(_mm_xor_si128(tmp3, mask3), mask3);
                                tmp4 = _mm_sub_epi16(_mm_xor_si128(tmp4, mask4), mask4);
                                dist1 = _mm_adds_epu16(dist1, tmp1);
                                dist2 = _mm_adds_epu16(dist2, tmp2);
                                dist3 = _mm_adds_epu16(dist3, tmp3);
                                dist4 = _mm_adds_epu16(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_epi16(dist1, dist1);
                        tmp2 = _mm_hadd_epi16(dist2, dist2);
                        tmp3 = _mm_hadd_epi16(dist3, dist3);
                        tmp4 = _mm_hadd_epi16(dist4, dist4);
                        tmp1 = _mm_hadd_epi16(tmp1, tmp1);
                        tmp2 = _mm_hadd_epi16(tmp2, tmp2);
                        tmp3 = _mm_hadd_epi16(tmp3, tmp3);
                        tmp4 = _mm_hadd_epi16(tmp4, tmp4);
                        dist1 = _mm_hadd_epi16(tmp1, tmp1);
                        dist2 = _mm_hadd_epi16(tmp2, tmp2);
                        dist3 = _mm_hadd_epi16(tmp3, tmp3);
                        dist4 = _mm_hadd_epi16(tmp4, tmp4);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] < min_distance)
                        {
                                min_distance = sdist3[0];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] < min_distance)
                        {
                                min_distance = sdist4[0];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
                {
                        ti1 = tn * dimensions;
                        dist1 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 8)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tmp1 = _mm_sub_epi16(vec, tvec1);
                                mask1 = _mm_srai_epi16(tmp1, 15);
                                tmp1 = _mm_sub_epi16(_mm_xor_si128(tmp1, mask1), mask1);
                                dist1 = _mm_adds_epi16(dist1, tmp1);
                        }
                        tmp1 = _mm_hadd_epi16(dist1, dist1);
                        tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                        dist1 = _mm_hadd_epi16(tmp2, tmp2);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_int_vec_M (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        unsigned int min_distance;
        unsigned int sdist1[4] __attribute__((aligned(16)));
        unsigned int sdist2[4] __attribute__((aligned(16)));
        unsigned int sdist3[4] __attribute__((aligned(16)));
        unsigned int sdist4[4] __attribute__((aligned(16)));
        __m128i vec, tvec1, tvec2, tvec3, tvec4;
        __m128i tmp1, tmp2, tmp3, tmp4;
        __m128i mask1, mask2, mask3, mask4;
        __m128i dist1, dist2, dist3, dist4;

        trcountU = trcount & ~0x03;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn++)
                {
                        ti1 = tn * dimensions;
                        ti2 = (tn + 1) * dimensions;
                        ti3 = (tn + 2) * dimensions;
                        ti4 = (tn + 3) * dimensions;
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tvec2 = _mm_load_si128((__m128i *) &trdata[ti2 + d]);
                                tvec3 = _mm_load_si128((__m128i *) &trdata[ti3 + d]);
                                tvec4 = _mm_load_si128((__m128i *) &trdata[ti4 + d]);
                                tmp1 = _mm_sub_epi32(vec, tvec1);
                                tmp2 = _mm_sub_epi32(vec, tvec2);
                                tmp3 = _mm_sub_epi32(vec, tvec3);
                                tmp4 = _mm_sub_epi32(vec, tvec4);
                                mask1 = _mm_srai_epi32(tmp1, 31);
                                mask2 = _mm_srai_epi32(tmp2, 31);
                                mask3 = _mm_srai_epi32(tmp3, 31);
                                mask4 = _mm_srai_epi32(tmp4, 31);
                                tmp1 = _mm_sub_epi32(_mm_xor_si128(tmp1, mask1), mask1);
                                tmp2 = _mm_sub_epi32(_mm_xor_si128(tmp2, mask2), mask2);
                                tmp3 = _mm_sub_epi32(_mm_xor_si128(tmp3, mask3), mask3);
                                tmp4 = _mm_sub_epi32(_mm_xor_si128(tmp4, mask4), mask4);
                                dist1 = _mm_add_epi32(dist1, tmp1);
                                dist2 = _mm_add_epi32(dist2, tmp2);
                                dist3 = _mm_add_epi32(dist3, tmp3);
                                dist4 = _mm_add_epi32(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                        tmp2 = _mm_hadd_epi32(dist2, dist2);
                        tmp3 = _mm_hadd_epi32(dist3, dist3);
                        tmp4 = _mm_hadd_epi32(dist4, dist4);
                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                        dist2 = _mm_hadd_epi32(tmp2, tmp2);
                        dist3 = _mm_hadd_epi32(tmp3, tmp3);
                        dist4 = _mm_hadd_epi32(tmp4, tmp4);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        _mm_store_si128((__m128i *) sdist2, dist2);
                        _mm_store_si128((__m128i *) sdist3, dist3);
                        _mm_store_si128((__m128i *) sdist4, dist4);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] < min_distance)
                        {
                                min_distance = sdist3[0];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] < min_distance)
                        {
                                min_distance = sdist4[0];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
                {
                        ti1 = tn * dimensions;
                        dist1 = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_si128((__m128i *) &data[i + d]);
                                tvec1 = _mm_load_si128((__m128i *) &trdata[ti1 + d]);
                                tmp1 = _mm_sub_epi32(vec, tvec1);
                                mask1 = _mm_srai_epi32(tmp1, 31);
                                tmp2 = _mm_sub_epi32(_mm_xor_si128(tmp1, mask1), mask1);
                                dist1 = _mm_add_epi32(dist1, tmp2);
                        }
                        tmp1 = _mm_hadd_epi32(dist1, dist1);
                        dist1 = _mm_hadd_epi32(tmp1, tmp1);
                        _mm_store_si128((__m128i *) sdist1, dist1);
                        if (sdist1[0] < min_distance)
                        {
                                min_distance = sdist1[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%u\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_vec_M (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        float min_distance;
        float distance1 __attribute__((aligned(16)));
        float distance2 __attribute__((aligned(16)));
        float distance3 __attribute__((aligned(16)));
        float distance4 __attribute__((aligned(16)));
        __m128 vec, tvec1, tvec2, tvec3, tvec4;
        __m128 mask, tmp1, tmp2, tmp3, tmp4;
        __m128 dist1, dist2, dist3, dist4;

        trcountU = trcount & ~0x03;
        mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_ps();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_ps(&data[i + d]);
                                tvec1 = _mm_load_ps(&trdata[ti1 + d]);
                                tvec2 = _mm_load_ps(&trdata[ti2 + d]);
                                tvec3 = _mm_load_ps(&trdata[ti3 + d]);
                                tvec4 = _mm_load_ps(&trdata[ti4 + d]);
                                tmp1 = _mm_sub_ps(vec, tvec1);
                                tmp2 = _mm_sub_ps(vec, tvec2);
                                tmp3 = _mm_sub_ps(vec, tvec3);
                                tmp4 = _mm_sub_ps(vec, tvec4);
                                tmp1 = _mm_and_ps(tmp1, mask);
                                tmp2 = _mm_and_ps(tmp2, mask);
                                tmp3 = _mm_and_ps(tmp3, mask);
                                tmp4 = _mm_and_ps(tmp4, mask);
                                dist1 = _mm_add_ps(dist1, tmp1);
                                dist2 = _mm_add_ps(dist2, tmp2);
                                dist3 = _mm_add_ps(dist3, tmp3);
                                dist4 = _mm_add_ps(dist4, tmp4);
                        }
                        tmp1 = _mm_hadd_ps(dist1, dist1);
                        tmp2 = _mm_hadd_ps(dist2, dist2);
                        tmp3 = _mm_hadd_ps(dist3, dist3);
                        tmp4 = _mm_hadd_ps(dist4, dist4);
                        dist1 = _mm_hadd_ps(tmp1, tmp1);
                        dist2 = _mm_hadd_ps(tmp2, tmp2);
                        dist3 = _mm_hadd_ps(tmp3, tmp3);
                        dist4 = _mm_hadd_ps(tmp4, tmp4);
                        _mm_store_ss(&distance1, dist1);
                        _mm_store_ss(&distance2, dist2);
                        _mm_store_ss(&distance3, dist3);
                        _mm_store_ss(&distance4, dist4);
                        if (distance1 < min_distance)
                        {
                                min_distance = distance1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (distance2 < min_distance)
                        {
                                min_distance = distance2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (distance3 < min_distance)
                        {
                                min_distance = distance3;
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (distance4 < min_distance)
                        {
                                min_distance = distance4;
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
                {
                        ti1 = tn * dimensions;
                        dist1 = _mm_setzero_ps();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_ps(&data[i + d]);
                                tvec1 = _mm_load_ps(&trdata[ti1 + d]);
                                tmp1 = _mm_and_ps(_mm_sub_ps(vec, tvec1), mask);
                                dist1 = _mm_add_ps(dist1, tmp1);
                        }
                        tmp1 = _mm_hadd_ps(dist1, dist1);
                        dist1 = _mm_hadd_ps(tmp1, tmp1);
                        _mm_store_ss(&distance1, dist1);
                        if (distance1 < min_distance)
                        {
                                min_distance = distance1;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_vec_M (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2, ti3, ti4;
        int cl, d, idx;
        double min_distance;
        double sdist1[2] __attribute__((aligned(16)));
        double sdist2[2] __attribute__((aligned(16)));
        double sdist3[2] __attribute__((aligned(16)));
        double sdist4[2] __attribute__((aligned(16)));
        __m128d vec, tvec1, tvec2, tvec3, tvec4;
        __m128d dist1, dist2, dist3, dist4;
        __m128d mask, tmp1, tmp2, tmp3, tmp4;
        __m128i imask;

        trcountU = trcount & ~0x03;
        imask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
        imask = _mm_srli_epi64(imask, 1);
        mask = _mm_castsi128_pd(imask);
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
                        dist1 = dist2 = dist3 = dist4 = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec1 = _mm_load_pd(&trdata[ti1 + d]);
                                tvec2 = _mm_load_pd(&trdata[ti2 + d]);
                                tvec3 = _mm_load_pd(&trdata[ti3 + d]);
                                tvec4 = _mm_load_pd(&trdata[ti4 + d]);
                                tmp1 = _mm_sub_pd(vec, tvec1);
                                tmp2 = _mm_sub_pd(vec, tvec2);
                                tmp3 = _mm_sub_pd(vec, tvec3);
                                tmp4 = _mm_sub_pd(vec, tvec4);
                                tmp1 = _mm_and_pd(tmp1, mask);
                                tmp2 = _mm_and_pd(tmp2, mask);
                                tmp3 = _mm_and_pd(tmp3, mask);
                                tmp4 = _mm_and_pd(tmp4, mask);
                                dist1 = _mm_add_pd(dist1, tmp1);
                                dist2 = _mm_add_pd(dist2, tmp2);
                                dist3 = _mm_add_pd(dist3, tmp3);
                                dist4 = _mm_add_pd(dist4, tmp4);
                        }
                        _mm_store_pd(sdist1, dist1);
                        _mm_store_pd(sdist2, dist2);
                        _mm_store_pd(sdist3, dist3);
                        _mm_store_pd(sdist4, dist4);
                        if (sdist1[0] + sdist1[1] < min_distance)
                        {
                                min_distance = sdist1[0] + sdist1[1];
                                cl = trklass[tn];
                                idx = tn;
                        }
                        if (sdist2[0] + sdist2[1] < min_distance)
                        {
                                min_distance = sdist2[0] + sdist2[1];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
                        }
                        if (sdist3[0] + sdist3[1] < min_distance)
                        {
                                min_distance = sdist3[0] + sdist3[1];
                                cl = trklass[tn + 2];
                                idx = tn + 2;
                        }
                        if (sdist4[0] + sdist4[1] < min_distance)
                        {
                                min_distance = sdist4[0] + sdist4[1];
                                cl = trklass[tn + 3];
                                idx = tn + 3;
                        }
                }
                for (;  tn < trcount;  tn++)
                {
                        ti1 = tn * dimensions;
                        dist1 = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec1 = _mm_load_pd(&trdata[ti1 + d]);
                                tmp1 = _mm_and_pd(_mm_sub_pd(vec, tvec1), mask);
                                dist1 = _mm_add_pd(dist1, tmp1);
                        }
                        _mm_store_pd(sdist1, dist1);
                        if (sdist1[0] + sdist1[1] < min_distance)
                        {
                                min_distance = sdist1[0] + sdist1[1];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}
