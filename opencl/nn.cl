__kernel
void nn (int dimensions, int trcount, __global float *trdata,
         __global int *trklass, int count, __global float *data,
         __global int *klass)
{
        int n, tn;
        int d, stride;
        float tmp;
        __local float distance[BLOCKDIM_Y][BLOCKDIM_X];
        __local float min_distance[BLOCKDIM_Y];
        __local int cl[BLOCKDIM_Y];

        for (n = get_global_id(1);  n < count;  n += get_global_size(1))
        {
                if (get_local_id(0) == 0)
                        min_distance[get_local_id(1)] = FLT_MAX;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        distance[get_local_id(1)][get_local_id(0)] = 0.0f;
                        for (d = 0;  d < dimensions;  d += get_local_size(0))
                        {
                                tmp = data[n * dimensions + d + get_local_id(0)] -
                                     trdata[tn * dimensions + d + get_local_id(0)];
                                distance[get_local_id(1)][get_local_id(0)] += tmp * tmp;
                        }
                        /* Distance reduction */
                        barrier(CLK_LOCAL_MEM_FENCE);
                        for (stride = get_local_size(0) / 2;  stride > 0;  stride /= 2)
                                if (get_local_id(0) < stride)
                                        distance[get_local_id(1)][get_local_id(0)] += distance[get_local_id(1)][get_local_id(0) + stride];
                        if (get_local_id(0) == 0)
                                if (distance[get_local_id(1)][0] < min_distance[get_local_id(1)])
                                {
                                        min_distance[get_local_id(1)] = distance[get_local_id(1)][0];
                                        cl[get_local_id(1)] = trklass[tn];
                                }
                }
                klass[n] = cl[get_local_id(1)];
        }
}

// vim:syntax=c
