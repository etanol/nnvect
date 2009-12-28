#define LOCAL_X   get_local_id(0)
#define LOCAL_Y   get_local_id(1)
#define GLOBAL_X  get_global_id(0)
#define GLOBAL_Y  get_global_id(1)
#define DIM_X     get_global_size(0)
#define DIM_Y     get_global_size(1)

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

        for (n = GLOBAL_Y;  n < count;  n += DIM_Y)
        {
                if (LOCAL_X == 0)
                        min_distance[LOCAL_Y] = FLT_MAX;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        distance[LOCAL_Y][LOCAL_X] = 0.0f;
                        for (d = 0;  d < dimensions;  d += BLOCKDIM_X)
                        {
                                tmp = data[n * dimensions + d + LOCAL_X] -
                                     trdata[tn * dimensions + d + LOCAL_X];
                                distance[LOCAL_Y][LOCAL_X] += tmp * tmp;
                        }
                        /* Distance reduction */
                        barrier(CLK_LOCAL_MEM_FENCE);
                        for (stride = BLOCKDIM_X / 2;  stride > 0;  stride /= 2)
                                if (LOCAL_X < stride)
                                        distance[LOCAL_Y][LOCAL_X] += distance[LOCAL_Y][LOCAL_X + stride];
                        if (LOCAL_X == 0)
                                if (distance[LOCAL_Y][0] < min_distance[LOCAL_Y])
                                {
                                        min_distance[LOCAL_Y] = distance[LOCAL_Y][0];
                                        cl[LOCAL_Y] = trklass[tn];
                                }
                }
                klass[n] = cl[LOCAL_Y];
        }
}

// vim:syntax=c
