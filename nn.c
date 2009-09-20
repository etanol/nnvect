#include <xmmintrin.h>

void nn (int dimensions, int trcount, float *trfeatures, int *trclasses,
         int count, float *features, int **classes)
{
        int n, tn;
        int i, ti;
        float min_distance;
        float distance[4] __attribute__((aligned(16)));
        int class, d;
        __m128 vec, tvec;
        __m128 tmp1, tmp2;
        __m128 dist;

        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = (float) (1 << 31);
                class = -1;

                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = _mm_setzero_ps();
                        for (d = 0;  d < dimensions;  d += 4)
                        {
                                vec = _mm_load_ps(&features[i + d]);
                                tvec = _mm_load_ps(&trfeatures[ti + d]);
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
                                class = trclasses[tn];
                        }
                }
                (*classes)[n] = class;
        }
}

