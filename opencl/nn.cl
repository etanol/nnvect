/*
 * BS_X and BS_Y are provided through the kernel compilation arguments.
 */
#define X     get_local_id(0)
#define Y     get_local_id(1)
#define GX    get_global_id(0)
#define GY    get_global_id(1)
#define GS_X  get_global_size(0)
#define GS_Y  get_global_size(1)

__kernel
void nn (int dimensions, int trcount, __global float *trdata,
         __global int *trklass, int count, __global float *data,
         __global int *klass)
{
        int n, tn;
        int d, stride;
        float tmp;
        __local float distance[BS_Y][BS_X];
        __local float min_distance[BS_Y];
        __local int cl[BS_Y];

        for (n = GY;  n < count;  n += GS_Y)
        {
                if (X == 0)
                        min_distance[Y] = FLT_MAX;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        distance[Y][X] = 0.0f;
                        for (d = 0;  d < dimensions;  d += BS_X)
                        {
                                tmp = data[n * dimensions + d + X] -
                                      trdata[tn * dimensions + d + X];
                                distance[Y][X] += tmp * tmp;
                        }
                        /* Distance reduction */
                        barrier(CLK_LOCAL_MEM_FENCE);
                        for (stride = BS_X / 2;  stride > 0;  stride /= 2)
                                if (X < stride)
                                        distance[Y][X] += distance[Y][X + stride];
                        if (X == 0)
                                if (distance[Y][0] < min_distance[Y])
                                {
                                        min_distance[Y] = distance[Y][0];
                                        cl[Y] = trklass[tn];
                                }
                }
                klass[n] = cl[Y];
        }
}

// vim:syntax=c
