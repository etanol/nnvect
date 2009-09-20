#include "util.h"

#include <tmmintrin.h>
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
        int n, tn;
        int i, ti;
        unsigned int min_distance, distance;
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
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3] +
                                   sdist[4] + sdist[5] + sdist[6] + sdist[7];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        unsigned int min_distance, distance;
        short sdist[8] __attribute__((aligned(16)));
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
                                tmp2 = _mm_mullo_epi16(tmp1, tmp1);
                                dist = _mm_adds_epu16(dist, tmp2);
                        }
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3] +
                                   sdist[4] + sdist[5] + sdist[6] + sdist[7];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        uint64_t min_distance, distance;
        uint64_t sdist[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128i vec, tvec;
        __m128i tmp1, tmp2, mask;
        __m128i dist;

        debug("Class\tDist\tIndex ");
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = ~0ULL;
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
                                mask = _mm_srai_epi32(tmp1, 31);
                                tmp1 = _mm_sub_epi32(_mm_xor_si128(tmp1, mask), mask);
                                tmp2 = _mm_mul_epu32(tmp1, tmp1);
                                dist = _mm_add_epi64(dist, tmp2);
                                tmp1 = _mm_slli_si128(tmp1, 4);
                                tmp2 = _mm_mul_epu32(tmp1, tmp1);
                                dist = _mm_add_epi64(dist, tmp2);
                        }
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[1];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%llu\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void nn_float_vec_E (int dimensions, int trcount, float *trdata, int *trklass,
                     int count, float *data, int *klass)
{
        int n, tn;
        int i, ti;
        float min_distance, distance;
        float sdist[4] __attribute__((aligned(16)));
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
                        /*
                        tmp1 = _mm_hadd_ps(dist, _mm_setzero_ps());
                        dist = _mm_hadd_ps(tmp1, _mm_setzero_ps());
                        _mm_store_ss(&distance, dist);
                        */
                        _mm_store_ps(sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        double min_distance, distance;
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
                        distance = sdist[0] + sdist[1];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, distance;
        short sdist[8] __attribute__((aligned(16)));
        __m128i vec, tvec;
        __m128i dist, tmp;

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
                                tmp = _mm_sad_epu8(vec, tvec);
                                dist = _mm_adds_epi16(dist, tmp);
                        }
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[4];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, distance;
        short sdist[8] __attribute__((aligned(16)));
        __m128i vec, tvec;
        __m128i dist, tmp, mask;

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
                                tmp = _mm_sub_epi16(vec, tvec);
                                mask = _mm_srai_epi16(tmp, 15);
                                tmp = _mm_sub_epi16(_mm_xor_si128(tmp, mask), mask);
                                dist = _mm_adds_epi16(dist, tmp);
                        }
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3] +
                                   sdist[4] + sdist[5] + sdist[6] + sdist[7];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        unsigned int min_distance, distance;
        unsigned int sdist[4] __attribute__((aligned(16)));
        __m128i vec, tvec;
        __m128i dist, tmp, mask;

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
                                tmp = _mm_sub_epi32(vec, tvec);
                                mask = _mm_srai_epi32(tmp, 31);
                                tmp = _mm_sub_epi32(_mm_xor_si128(tmp, mask), mask);
                                dist = _mm_add_epi32(dist, tmp);
                        }
                        _mm_store_si128((__m128i *) sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        float min_distance, distance = 0.0f;
        float sdist[4] __attribute__((aligned(16)));
        __m128 vec, tvec;
        __m128 dist, tmp, mask;

        mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
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
                                tmp = _mm_and_ps(_mm_sub_ps(vec, tvec), mask);
                                dist = _mm_add_ps(dist, tmp);
                        }
                        _mm_store_ps(sdist, dist);
                        distance = sdist[0] + sdist[1] + sdist[2] + sdist[3];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
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
        int n, tn;
        int i, ti;
        int cl, d, idx;
        double min_distance, distance = 0.0;
        double sdist[2] __attribute__((aligned(16)));
        __m128d vec, tvec;
        __m128d dist, tmp, mask;
        __m128i imask;

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
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec = _mm_load_pd(&trdata[ti + d]);
                                tmp = _mm_and_pd(_mm_sub_pd(vec, tvec), mask);
                                dist = _mm_add_pd(dist, tmp);
                        }
                        _mm_store_pd(sdist, dist);
                        distance = sdist[0] + sdist[1];
                        if (distance < min_distance)
                        {
                                min_distance = distance;
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}

