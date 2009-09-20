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
        int n, tn, trcountU;
        int i, ti1, ti2;
        unsigned int min_distance;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i mask1, mask2;
        __m128i dist1, dist2;

        trcountU = trcount & ~0x01;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 2)
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
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
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


void nn_short_vec_U (int dimensions, int trcount, short *trdata, int *trklass,
                     int count, short *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

        trcountU = trcount & ~0x01;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 2)
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
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
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


void nn_int_vec_U (int dimensions, int trcount, int *trdata, int *trklass,
                   int count, int *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

        trcountU = trcount & ~0x01;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = UINT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 2)
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
                                idx = tn;
                        }
                        if (sdist2[0] < min_distance)
                        {
                                min_distance = sdist2[0];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
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

void nn_float_vec_U (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2;
        float min_distance;
        float distance1 __attribute__((aligned(16)));
        float distance2 __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec1, tvec2;
        __m128 tmp1, tmp2;
        __m128 dist1, dist2;

        trcountU = trcount & ~0x01;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = FLT_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 2)
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
                                idx = tn;
                        }
                        if (distance2 < min_distance)
                        {
                                min_distance = distance2;
                                cl = trklass[tn + 1];
                                idx = tn + 1;
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


void nn_double_vec_U (int dimensions, int trcount, double *trdata, int *trklass,
                      int count, double *data, int *klass)
{
        int n, tn, trcountU;
        int i, ti1, ti2;
        double min_distance;
        double sdist1[2] __attribute__((aligned(16)));
        double sdist2[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec1, tvec2;
        __m128d tmp1, tmp2;
        __m128d dist1, dist2;

        trcountU = trcount & ~0x01;
        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = DBL_MAX;
                cl = -1;
                idx = -1;
                for (tn = 0;  tn < trcountU;  tn += 2)
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
                                idx = tn;
                        }
                        if (sdist2[0] + sdist2[1] < min_distance)
                        {
                                min_distance = sdist2[0] + sdist2[1];
                                cl = trklass[tn + 1];
                                idx = tn + 1;
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
        int tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        short sdist1[8] __attribute__((aligned(16)));
        short sdist2[8] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i mask1, mask2;
        __m128i dist1, dist2;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                tbcU = tbc & ~0x01;
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                                if (sdist2[0] < min_distance)
                                {
                                        min_distance = sdist2[0];
                                        cl = trklass[tn + 1];
                                        idx = tn + 1;
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

        int tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                tbcU = tbc & ~0x01;
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                                if (sdist2[0] < min_distance)
                                {
                                        min_distance = sdist2[0];
                                        cl = trklass[tn + 1];
                                        idx = tn + 1;
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
        int tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        unsigned int min_distance;
        int sdist1[4] __attribute__((aligned(16)));
        int sdist2[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec1, tvec2;
        __m128i tmp1, tmp2;
        __m128i dist1, dist2;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                tbcU = tbc & ~0x01;
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                                if (sdist2[0] < min_distance)
                                {
                                        min_distance = sdist2[0];
                                        cl = trklass[tn + 1];
                                        idx = tn + 1;
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
        int tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        float min_distance;
        float distance1 __attribute__((aligned(16)));
        float distance2 __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec1, tvec2;
        __m128 tmp1, tmp2;
        __m128 dist1, dist2;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                tbcU = tbc & ~0x01;
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                                if (distance2 < min_distance)
                                {
                                        min_distance = distance2;
                                        cl = trklass[tn + 1];
                                        idx = tn + 1;
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
        int tbc, tbcU, tbn;
        int n, tn;
        int i, ti1, ti2;
        double min_distance;
        double sdist1[2] __attribute__((aligned(16)));
        double sdist2[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec1, tvec2;
        __m128d tmp1, tmp2;
        __m128d dist1, dist2;

        debug("Class\tDist\tIndex ");
        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                tbcU = tbc & ~0x01;
                for (n = 0;  n < count;  n++)
                {
                        i = n * dimensions;
                        min_distance = distance[n];
                        cl = klass[n];
                        idx = -1;
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
                                        idx = tn;
                                }
                                if (sdist2[0] + sdist2[1] < min_distance)
                                {
                                        min_distance = sdist2[0] + sdist2[1];
                                        cl = trklass[tn + 1];
                                        idx = tn + 1;
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
                                        idx = tn;
                                }
                        }
                        debug("%d\t%lf\t%d ", cl, min_distance, idx);
                        distance[n] = min_distance;
                        klass[n] = cl;
                }
        }
}

