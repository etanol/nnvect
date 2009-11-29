#include "knn.h"
#include "db.h"
#include "util.h"

#include <limits.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

typedef void (*U_func) (int, int, void *, int *, int, void *, int, int *, int *, void *);
typedef void (*B_func) (int, int, int, void *, int *, int, void *, int, int *, int *, void *);


void knn_byte_U   (int, int, char *,   int *, int, char *,   int, int *, int *, unsigned int *);
void knn_short_U  (int, int, short *,  int *, int, short *,  int, int *, int *, unsigned int *);
void knn_int_U    (int, int, int *,    int *, int, int *,    int, int *, int *, unsigned int *);
void knn_float_U  (int, int, float *,  int *, int, float *,  int, int *, int *, float *);
void knn_double_U (int, int, double *, int *, int, double *, int, int *, int *, double *);

void knn_byte_B   (int, int, int, char *,   int *, int, char *,   int, int *, int *, unsigned int *);
void knn_short_B  (int, int, int, short *,  int *, int, short *,  int, int *, int *, unsigned int *);
void knn_int_B    (int, int, int, int *,    int *, int, int *,    int, int *, int *, unsigned int *);
void knn_float_B  (int, int, int, float *,  int *, int, float *,  int, int *, int *, float *);
void knn_double_B (int, int, int, double *, int *, int, double *, int, int *, int *, double *);


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
        if (trdb->block_items > 0)
        {
                /* Select blocked versions */
                B_func func = NULL;

                switch (type)
                {
                        case BYTE  :  func = (B_func) knn_byte_B;    break;
                        case SHORT :  func = (B_func) knn_short_B;   break;
                        case INT   :  func = (B_func) knn_int_B;     break;
                        case FLOAT :  func = (B_func) knn_float_B;   break;
                        case DOUBLE:  func = (B_func) knn_double_B;  break;
                }
                if (func == NULL)
                        quit("Invalid combination of implementation, k-neighbours and value type, with blocking");

                func(db->dimensions, trdb->count, trdb->block_items, trdb->data,
                     trdb->klass, db->count, db->data, k, nbh->imax, nbh->klass,
                     nbh->distance);
        }
        else
        {
                /* Select unblocked versions */
                U_func func = NULL;

                switch (type)
                {
                        case BYTE  :  func = (U_func) knn_byte_U;    break;
                        case SHORT :  func = (U_func) knn_short_U;   break;
                        case INT   :  func = (U_func) knn_int_U;     break;
                        case FLOAT :  func = (U_func) knn_float_U;   break;
                        case DOUBLE:  func = (U_func) knn_double_U;  break;
                }
                if (func == NULL)
                        quit("Invalid combination of implementation, k-neighbours and value type, without blocking");

                func(db->dimensions, trdb->count, trdb->data, trdb->klass,
                     db->count, db->data, k, nbh->imax, nbh->klass,
                     nbh->distance);
        }
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



/******************************************************************************/
/*                                                                            */
/*                             UNBLOCKED VERSIONS                             */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void knn_byte_U (int dimensions, int trcount, char *trdata, int *trklass,
                 int count, char *data, int k, int *imax, int *nklass,
                 unsigned int *distance)
{
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        char tmp;

        #pragma omp parallel for schedule(static) private(i, b, tn, ti, dist, \
                d, tmp, j)
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                b = n * k;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
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


void knn_short_U (int dimensions, int trcount, short *trdata, int *trklass,
                  int count, short *data, int k, int *imax, int *nklass,
                  unsigned int *distance)
{
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        short tmp;

        #pragma omp parallel for schedule(static) private(i, b, tn, ti, dist, \
                d, tmp, j)
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                b = n * k;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
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


void knn_int_U (int dimensions, int trcount, int *trdata, int *trklass,
                int count, int *data, int k, int *imax, int *nklass,
                unsigned int *distance)
{
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        int tmp;

        #pragma omp parallel for schedule(static) private(i, b, tn, ti, dist, \
                d, tmp, j)
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                b = n * k;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0;
                        for (d = 0;  d < dimensions;  d++)
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


/**************************  FLOATING POINT VALUES  **************************/

void knn_float_U (int dimensions, int trcount, float *trdata, int *trklass,
                  int count, float *data, int k, int *imax, int *nklass,
                  float *distance)
{
        int n, tn;
        int i, ti;
        int d, j, b;
        float dist, tmp;

        #pragma omp parallel for schedule(static) private(i, b, tn, ti, dist, \
                d, tmp, j)
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                b = n * k;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0.0f;
                        for (d = 0;  d < dimensions;  d++)
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


void knn_double_U (int dimensions, int trcount, double *trdata, int *trklass,
                   int count, double *data, int k, int *imax, int *nklass,
                   double *distance)
{
        int n, tn;
        int i, ti;
        int d, j, b;
        double dist, tmp;

        #pragma omp parallel for schedule(static) private(i, b, tn, ti, dist, \
                d, tmp, j)
        for (n = 0;  n < count;  n++)
        {
                i = n * dimensions;
                b = n * k;
                for (tn = 0;  tn < trcount;  tn++)
                {
                        ti = tn * dimensions;
                        dist = 0.0;
                        for (d = 0;  d < dimensions;  d++)
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



/******************************************************************************/
/*                                                                            */
/*                              BLOCKED VERSIONS                              */
/*                                                                            */
/******************************************************************************/


/******************************  INTEGER VALUES  ******************************/

void knn_byte_B (int dimensions, int trcount, int trblockcount, char *trdata,
                 int *trklass, int count, char *data, int k, int *imax,
                 int *nklass, unsigned int *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        char tmp;

        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                #pragma omp parallel for schedule (static) private(i, b, tn, \
                        ti, dist, d, tmp, j)
                for (n = 0;  n < count;  n++)
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


void knn_short_B (int dimensions, int trcount, int trblockcount, short *trdata,
                  int *trklass, int count, short *data, int k, int *imax,
                  int *nklass, unsigned int *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        short tmp;

        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                #pragma omp parallel for schedule (static) private(i, b, tn, \
                        ti, dist, d, tmp, j)
                for (n = 0;  n < count;  n++)
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


void knn_int_B (int dimensions, int trcount, int trblockcount, int *trdata,
                 int *trklass, int count, int *data, int k, int *imax,
                 int *nklass, unsigned int *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        unsigned int dist;
        int tmp;

        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                #pragma omp parallel for schedule (static) private(i, b, tn, \
                        ti, dist, d, tmp, j)
                for (n = 0;  n < count;  n++)
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


/**************************  FLOATING POINT VALUES  **************************/

void knn_float_B (int dimensions, int trcount, int trblockcount, float *trdata,
                  int *trklass, int count, float *data, int k, int *imax,
                  int *nklass, float *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        float dist, tmp;

        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                #pragma omp parallel for schedule (static) private(i, b, tn, \
                        ti, dist, d, tmp, j)
                for (n = 0;  n < count;  n++)
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


void knn_double_B (int dimensions, int trcount, int trblockcount,
                   double *trdata, int *trklass, int count, double *data,
                   int k, int *imax, int *nklass, double *distance)
{
        int tbc, tbn;
        int n, tn;
        int i, ti;
        int d, j, b;
        double dist, tmp;

        for (tbn = 0;  tbn < trcount;  tbn += trblockcount)
        {
                tbc = (tbn + trblockcount < trcount ?
                       tbn + trblockcount : trcount);
                #pragma omp parallel for schedule (static) private(i, b, tn, \
                        ti, dist, d, tmp, j)
                for (n = 0;  n < count;  n++)
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

