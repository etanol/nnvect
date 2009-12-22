__kernel
void nn (int dimensions, int trcount, __global float *trdata,
         __global int *trklass, int count, __global float *data,
         __global int *klass)
{
        int n, tn;
        int i, ti;
        int cl, d, stride;
        float min_distance;
        __local float dist[get_local_size(0)];
        __local float tmp[get_local_size(0)];

        for (n = 0;  n < count;  n += get_global_size(1))
        {
                i = n * dimensions;
                if (get_local_id(0) == 0)
                        min_distance = FLT_MAX;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist[get_local_id(0)] = 0.0;
                        for (d = 0;  d < dimensions;  d += get_local_size(0))
                        {
                                tmp[get_local_id(0)] = data[i + d + get_local_id(0)] -
                                                       trdata[ti + d + get_local_id(0)];
                                dist[get_local_id(0)] += tmp[get_local_id(0)] * tmp[get_local_id(0)];
                        }
                        /* Distance reduction */
                        for (stride = get_local_size(0) / 2;  stride > 0;  stride /= 2)
                        {
                                barrier(CLK_LOCAL_MEM_FENCE);
                                if (get_local_id(0) < stride)
                                        dist[get_local_id(0)] += dist[get_local_id(0) + stride];
                        }
                        if (get_local_id(0) == 0)
                                if (dist[0] < min_distance)
                                {
                                        min_distance = dist[0];
                                        cl = trklass[tn];
                                }
                        barrier(CLK_LOCAL_MEM_FENCE);
                }
                klass[n] = cl;
        }
}

// vim:syntax=c
