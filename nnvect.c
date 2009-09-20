#include "util.h"

#include <emmintrin.h>
#include <limits.h>
#include <float.h>


void nn_byte_vect (int dimensions, int trcount, char *trdata, int *trklass,
                   int count, char *data, int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d;
        int min_distance, distance;
        short sdist[8] __attribute__((aligned(16)));
        __m128i vec, tvec;
        __m128i dist, tmp;

        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = INT_MAX;
                cl = -1;

                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 16)
                        {
                                vec = _mm_castps_si128(_mm_load_ps((float *) &data[i + d]));
                                tvec = _mm_castps_si128(_mm_load_ps((float *) &trdata[ti + d]));
#if 0
                                vec = _mm_setr_epi8(features[i + d + 0], features[i + d + 1], features[i + d + 2], features[i + d + 3],
                                                    features[i + d + 4], features[i + d + 5], features[i + d + 6], features[i + d + 7],
                                                    features[i + d + 8], features[i + d + 9], features[i + d + 10], features[i + d + 11],
                                                    features[i + d + 12], features[i + d + 13], features[i + d + 14], features[i + d + 15]);
                                tvec = _mm_setr_epi8(trfeatures[ti + d + 0], trfeatures[ti + d + 1], trfeatures[ti + d + 2], trfeatures[ti + d + 3],
                                                     trfeatures[ti + d + 4], trfeatures[ti + d + 5], trfeatures[ti + d + 6], trfeatures[ti + d + 7],
                                                     trfeatures[ti + d + 8], trfeatures[ti + d + 9], trfeatures[ti + d + 10], trfeatures[ti + d + 11],
                                                     trfeatures[ti + d + 12], trfeatures[ti + d + 13], trfeatures[ti + d + 14], trfeatures[ti + d + 15]);
                                tmp = _mm_sad_epi8(vec, tvec);
#endif
                                tmp = _mm_setzero_si128();
                                dist = _mm_adds_epi16(dist, tmp);
                        }

                        _mm_store_ps((float *) sdist, _mm_castsi128_ps(dist));
                        distance = sdist[0] + sdist[4];

                        if (distance < min_distance)
                        {
                                min_distance = distance;
                                cl= trklass[tn];
                        }
                }
                klass[n] = cl;
        }
}


void nn_short_vect (int dimensions, int trcount, short *trdata, int *trklass,
                    int count, short *data, int *klass)
{
}


void nn_int_vect (int dimensions, int trcount, int *trdata, int *trklass,
                  int count, int *data, int *klass)
{
#if 0
        int n, tn;
        int i, ti;
        int min_distance, distance;
        int sdist[4] __attribute__((aligned(16)));
        __m128i vect, tvec;
        __m128i tmp1, tmp2;
        __m128i dist;

        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = INT_MAX;
                class = -1;

                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = _mm_setzero_si128();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_castps_si128(_mm_load_ps((float *) &features[i + d]));
                                tvec = _mm_castps_si128(_mm_load_ps((float *) &trfeatures[ti + d]));
                                tmp1 = _mm_sub_epi32(vec, tvec);

                        }
                }
        }
#endif
}


void nn_float_vect (int dimensions, int trcount, float *trdata, int *trklass,
                    int count, float *data, int *klass)
{
        int n, tn;
        int i, ti;
        float min_distance;
        float distance[4] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128 vec, tvec;
        __m128 tmp1, tmp2;
        __m128 dist;

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
                        _mm_store_ps(distance, dist);
                        distance[0] += distance[1];
                        distance[2] += distance[3];
                        distance[0] += distance[2];
                        if (distance[0] < min_distance)
                        {
                                min_distance = distance[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%f\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}


void nn_double_vect (int dimensions, int trcount, double *trdata, int *trklass,
                     int count, double *data, int *klass)
{
        int n, tn;
        int i, ti;
        double min_distance;
        double distance[2] __attribute__((aligned(16)));
        int cl, d, idx;
        __m128d vec, tvec;
        __m128d tmp1, tmp2;
        __m128d dist;

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
                        dist = _mm_setzero_pd();
                        for (d = 0;  d < dimensions;  d += 2)
                        {
                                vec = _mm_load_pd(&data[i + d]);
                                tvec = _mm_load_pd(&trdata[ti + d]);
                                tmp1 = _mm_sub_pd(vec, tvec);
                                tmp2 = _mm_mul_pd(tmp1, tmp1);
                                dist = _mm_add_pd(dist, tmp2);
                        }
                        _mm_store_pd(distance, dist);
                        distance[0] += distance[1];
                        if (distance[0] < min_distance)
                        {
                                min_distance = distance[0];
                                cl = trklass[tn];
                                idx = tn;
                        }
                }
                debug("%d\t%lf\t%d ", cl, min_distance, idx);
                klass[n] = cl;
        }
}

