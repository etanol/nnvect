#define X     get_local_id(0)
#define Y     get_local_id(1)
#define GX    get_global_id(0)
#define GY    get_global_id(1)

void accum (float tst, __local float *trn, float *dist)
{
        dist[0]  += (tst - trn[0])  * (tst - trn[0]);
        dist[1]  += (tst - trn[1])  * (tst - trn[1]);
        dist[2]  += (tst - trn[2])  * (tst - trn[2]);
        dist[3]  += (tst - trn[3])  * (tst - trn[3]);
        dist[4]  += (tst - trn[4])  * (tst - trn[4]);
        dist[5]  += (tst - trn[5])  * (tst - trn[5]);
        dist[6]  += (tst - trn[6])  * (tst - trn[6]);
        dist[7]  += (tst - trn[7])  * (tst - trn[7]);
        dist[8]  += (tst - trn[8])  * (tst - trn[8]);
        dist[9]  += (tst - trn[9])  * (tst - trn[9]);
        dist[10] += (tst - trn[10]) * (tst - trn[10]);
        dist[11] += (tst - trn[11]) * (tst - trn[11]);
        dist[12] += (tst - trn[12]) * (tst - trn[12]);
        dist[13] += (tst - trn[13]) * (tst - trn[13]);
        dist[14] += (tst - trn[14]) * (tst - trn[14]);
        dist[15] += (tst - trn[15]) * (tst - trn[15]);
}


__kernel
void distance_map (int dimensions, int trcount, __global float *trdata,
                   int count, __global float *data,
                    __global float *distances, __global int *indices)
{
        int bx = (GX / 16) * 64;
        int by = (GY / 4) * 16;
        int tid = Y * 16 + X;
        int i, idx;
        float min, tst[4];
        float dist[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        __local float trn[4][16];

        data += bx + tid;
        trdata += by + X + Y * trcount;
        distances += bx + tid + (GY / 4) * count;
        indices += bx + tid + (GY / 4) * count;

        for (i = 0;  i < dimensions;  i += 4)
        {
                tst[0] = data[0 * count];
                tst[2] = data[1 * count];
                tst[3] = data[2 * count];
                tst[4] = data[3 * count];
                trn[Y][X] = trdata[0];
                barrier(CLK_LOCAL_MEM_FENCE);

                accum(tst[0], trn[0], dist);
                accum(tst[1], trn[1], dist);
                accum(tst[2], trn[2], dist);
                accum(tst[3], trn[3], dist);

                data += 4 * count;
                trdata += 4 * trcount;
                barrier(CLK_LOCAL_MEM_FENCE);
        }

        min = MAXFLOAT;
        idx = -1;
        for (i = 0;  i < 16;  i++)
                if (dist[i] < min)
                {
                        min = dist[i];
                        idx = i;
                }
        distances[0] = min;
        indices[0] = by + idx;
}


__kernel
void reduction (int trcount, int count, __global float *distances,
                __global int *indices, __global int *trklass,
                __global int *klass)
{
        float min = MAXFLOAT;
        int i, idx, cols = trcount / 16;

        distances += GX;
        indices += GX;

        for (i = 0;  i < cols;  i++)
        {
                if (distances[0] < min)
                {
                        min = distances[0];
                        idx = indices[0];
                }
                distances += count;
                indices += count;
        }

        klass[GX] = trklass[idx];
}

// vim:syntax=c
