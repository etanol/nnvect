#include "knn.h"
#include "db.h"
#include "util.h"

#include <limits.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

typedef void (*knn_func) (int, int, int, void *, int *, int, int, void *, int,
                          int *, int *, void *);

void knn_byte   (int, int, int, char *,   int *, int, int, char *,   int, int *, int *, unsigned int *);
void knn_short  (int, int, int, short *,  int *, int, int, short *,  int, int *, int *, unsigned int *);
void knn_int    (int, int, int, int *,    int *, int, int, int *,    int, int *, int *, unsigned int *);
void knn_float  (int, int, int, float *,  int *, int, int, float *,  int, int *, int *, float *);
void knn_double (int, int, int, double *, int *, int, int, double *, int, int *, int *, double *);


struct nbhood *create_neighbourhood (int k, struct db *db)
{
        struct nbhood *nbh;

        nbh = xmalloc(sizeof(struct nbhood));
        nbh->imax = xmalloc(db->count * sizeof(int));
        nbh->klass = xmalloc(db->count * k * sizeof(int));
        switch (db->type)
        {
        case BYTE:
        case SHORT:
        case INT:
                nbh->distance = xmalloc(db->count * k * sizeof(unsigned int));
                break;
        case FLOAT:
                nbh->distance = xmalloc(db->count * k * sizeof(float));
                break;
        case DOUBLE:
                nbh->distance = xmalloc(db->count * k * sizeof(double));
                break;
        }

        return nbh;
}


void clear_neighbourhood (int k, struct db *db, struct nbhood *nbh)
{
        int i, all;

        all = db->count * k;

        memset(nbh->imax, 0, db->count * sizeof(int));
        memset(nbh->klass, 0, db->count * k * sizeof(int));

        switch (db->type)
        {
        case BYTE:
        case SHORT:
        case INT:
                for (i = 0;  i < all;  i++)
                        ((unsigned int *) nbh->distance)[i] = UINT_MAX;
                break;
        case FLOAT:
                for (i = 0;  i < all;  i++)
                        ((float *) nbh->distance)[i] = FLT_MAX;
                break;
        case DOUBLE:
                for (i = 0;  i < all;  i++)
                        ((double *) nbh->distance)[i] = DBL_MAX;
                break;
        }

}


void free_neighbourhood (struct nbhood *nbh)
{
        free(nbh->distance);
        free(nbh->klass);
        free(nbh->imax);
        free(nbh);
}


void knn (int k, enum valuetype type, struct db *trdb, struct db *db,
          struct nbhood * nbh)
{
        int blockcount, trblockcount;
        knn_func func;

        func = NULL;
        switch (type)
        {
                case BYTE  :  func = (knn_func) knn_byte;    break;
                case SHORT :  func = (knn_func) knn_short;   break;
                case INT   :  func = (knn_func) knn_int;     break;
                case FLOAT :  func = (knn_func) knn_float;   break;
                case DOUBLE:  func = (knn_func) knn_double;  break;
        }
        if (func == NULL)
                quit("Invalid combination of implementation, k-neighbours and value type, with blocking");

        if (trdb->block_items > 0)
                trblockcount = trdb->block_items;
        else
                trblockcount = trdb->count;
        if (db->block_items > 0)
                blockcount = db->block_items;
        else
                blockcount = db->count;

        func(db->dimensions, trdb->count, trblockcount, trdb->data, trdb->klass,
             db->count, blockcount, db->data, k, nbh->imax, nbh->klass,
             nbh->distance);
}



/*************************                            *************************/
/*************************  CLASSIFICATION FUNCTIONS  *************************/
/*************************                            *************************/

static void classify_int (int k, struct db *db, struct nbhood *nbh)
{
        int b, i, n, idx;
        int freq[db->label_count];
        unsigned int dist[db->label_count], d;

        for (n = 0;  n < db->count;  n++)
        {
                for (i = 0;  i < db->label_count;  i++)
                {
                        freq[i] = 0;
                        dist[i] = UINT_MAX;
                }
                b = n * k;
                for (i = 0;  i < k;  i++)
                {
                        idx = label_index(db, nbh->klass[b + i]);
                        d = ((unsigned int *) nbh->distance)[b + i];
                        freq[idx]++;
                        if (d < dist[idx])
                                dist[idx] = d;
                }
                idx = 0;
                for (i = 1;  i < db->label_count;  i++)
                        if (freq[i] > freq[idx] ||
                            (freq[i] == freq[idx] && dist[i] < dist[idx]))
                                idx = i;
                db->klass[n] = db->label[idx];
        }
}


static void classify_float (int k, struct db *db, struct nbhood *nbh)
{
        int b, i, n, idx;
        int freq[db->label_count];
        float dist[db->label_count], d;

        for (n = 0;  n < db->count;  n++)
        {
                for (i = 0;  i < db->label_count;  i++)
                {
                        freq[i] = 0;
                        dist[i] = FLT_MAX;
                }
                b = n * k;
                for (i = 0;  i < k;  i++)
                {
                        idx = label_index(db, nbh->klass[b + i]);
                        d = ((float *) nbh->distance)[b + i];
                        freq[idx]++;
                        if (d < dist[idx])
                                dist[idx] = d;
                }
                idx = 0;
                for (i = 1;  i < db->label_count;  i++)
                        if (freq[i] > freq[idx] ||
                            (freq[i] == freq[idx] && dist[i] < dist[idx]))
                                idx = i;
                db->klass[n] = db->label[idx];
        }
}


static void classify_double (int k, struct db *db, struct nbhood *nbh)
{
        int b, i, n, idx;
        int freq[db->label_count];
        double dist[db->label_count], d;

        for (n = 0;  n < db->count;  n++)
        {
                for (i = 0;  i < db->label_count;  i++)
                {
                        freq[i] = 0;
                        dist[i] = DBL_MAX;
                }
                b = n * k;
                for (i = 0;  i < k;  i++)
                {
                        idx = label_index(db, nbh->klass[b + i]);
                        d = ((double *) nbh->distance)[b + i];
                        freq[idx]++;
                        if (d < dist[idx])
                                dist[idx] = d;
                }
                idx = 0;
                for (i = 1;  i < db->label_count;  i++)
                        if (freq[i] > freq[idx] ||
                            (freq[i] == freq[idx] && dist[i] < dist[idx]))
                                idx = i;
                db->klass[n] = db->label[idx];
        }
}


void classify (int k, struct db *db, struct nbhood *nbh)
{
        switch (db->type)
        {
        case BYTE:
        case SHORT:
        case INT:
                classify_int(k, db, nbh);
                break;
        case FLOAT:
                classify_float(k, db, nbh);
                break;
        case DOUBLE:
                classify_double(k, db, nbh);
                break;
        }
}



/******************************  INTEGER VALUES  ******************************/

void knn_byte (int dimensions, int trcount, int trblockcount, char *trdata,
               int *trklass, int count, int blockcount, char *data, int k,
               int *imax, int *nklass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        char tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule (static) private(i, \
                                b, tn, ti, dist, d, tmp, j)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                b = n * k;
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions; d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < distance[b + imax[n]])
                                        {
                                                distance[b + imax[n]] = dist;
                                                nklass[b + imax[n]] = trklass[tn];
                                                imax[n] = 0;
                                                for (j = 1;  j < k;  j++)
                                                        if (distance[b + j] > distance[b + imax[n]])
                                                                imax[n] = j;
                                        }
                                }
                        }
                }
        }
}


void knn_short (int dimensions, int trcount, int trblockcount, short *trdata,
                int *trklass, int count, int blockcount, short *data, int k,
                int *imax, int *nklass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        short tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule (static) private(i, \
                                b, tn, ti, dist, d, tmp, j)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                b = n * k;
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions; d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < distance[b + imax[n]])
                                        {
                                                distance[b + imax[n]] = dist;
                                                nklass[b + imax[n]] = trklass[tn];
                                                imax[n] = 0;
                                                for (j = 1;  j < k;  j++)
                                                        if (distance[b + j] > distance[b + imax[n]])
                                                                imax[n] = j;
                                        }
                                }
                        }
                }
        }
}


void knn_int (int dimensions, int trcount, int trblockcount, int *trdata,
              int *trklass, int count, int blockcount, int *data, int k,
              int *imax, int *nklass, unsigned int *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        int tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule (static) private(i, \
                                b, tn, ti, dist, d, tmp, j)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                b = n * k;
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0;
                                        for (d = 0;  d < dimensions; d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < distance[b + imax[n]])
                                        {
                                                distance[b + imax[n]] = dist;
                                                nklass[b + imax[n]] = trklass[tn];
                                                imax[n] = 0;
                                                for (j = 1;  j < k;  j++)
                                                        if (distance[b + j] > distance[b + imax[n]])
                                                                imax[n] = j;
                                        }
                                }
                        }
                }
        }
}


/**************************  FLOATING POINT VALUES  **************************/

void knn_float (int dimensions, int trcount, int trblockcount, float *trdata,
                int *trklass, int count, int blockcount, float *data, int k,
                int *imax, int *nklass, float *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        float dist, tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule (static) private(i, \
                                b, tn, ti, dist, d, tmp, j)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                b = n * k;
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0.0f;
                                        for (d = 0;  d < dimensions; d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < distance[b + imax[n]])
                                        {
                                                distance[b + imax[n]] = dist;
                                                nklass[b + imax[n]] = trklass[tn];
                                                imax[n] = 0;
                                                for (j = 1;  j < k;  j++)
                                                        if (distance[b + j] > distance[b + imax[n]])
                                                                imax[n] = j;
                                        }
                                }
                        }
                }
        }
}


void knn_double (int dimensions, int trcount, int trblockcount,
                 double *trdata, int *trklass, int count, int blockcount,
                 double *data, int k, int *imax, int *nklass, double *distance)
{
        int bc, bn, tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        double dist, tmp;

        for (bn = 0;  bn < count;  bn += blockcount)
        {
                bc = (bn + blockcount < count ?
                      bn + blockcount : count);
                for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
                {
                        tbc = (tbn + trblockcount < trcount ?
                               tbn + trblockcount : trcount);
                        #pragma omp parallel for schedule (static) private(i, \
                                b, tn, ti, dist, d, tmp, j)
                        for (n = bn;  n < bc;  n++)
                        {
                                i = n * dimensions;
                                b = n * k;
                                for (tn = tbn;  tn < tbc;  tn++)
                                {
                                        ti = tn * dimensions;
                                        dist = 0.0;
                                        for (d = 0;  d < dimensions; d++)
                                        {
                                                tmp = data[i + d] - trdata[ti + d];
                                                dist += tmp * tmp;
                                        }
                                        if (dist < distance[b + imax[n]])
                                        {
                                                distance[b + imax[n]] = dist;
                                                nklass[b + imax[n]] = trklass[tn];
                                                imax[n] = 0;
                                                for (j = 1;  j < k;  j++)
                                                        if (distance[b + j] > distance[b + imax[n]])
                                                                imax[n] = j;
                                        }
                                }
                        }
                }
        }
}

