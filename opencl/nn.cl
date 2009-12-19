__kernel
void nn (int dimensions, int trcount, __global float *trdata,
         __global int *trklass, int count, __global float *data,
         __global int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d;
        float min_distance, dist, tmp;

        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                min_distance = FLT_MAX;
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
                        }
                }
                klass[n] = cl;
        }
}

// vim:syntax=c
