#ifndef __nnvect_knn
#define __nnvect_knn

#include "nn.h"

struct nbhood
{
        int *imax;
        int *klass;
        void *distance;
};

struct db;

struct nbhood *create_neighbourhood (int, struct db *);
void clear_neighbourhood (int, struct db *, struct nbhood *);
void free_neighbourhood (struct nbhood *);
void knn (int, enum valuetype, struct db *, struct db *, struct nbhood *);
void classify (int, struct db *, struct nbhood *);

#endif /* __nnvect_knn */
