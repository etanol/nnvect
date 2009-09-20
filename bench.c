#include "util.h"
#include "db.h"
#include "nn.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main (int argc, char **argv)
{
        struct timeval start, finish;
        struct db *db, *train_db;

        if (argc < 3)
                quit("Not enough arguments");

        printf("Loading %s\n", argv[1]);
        train_db = load_db(argv[1], FLOAT);
        printf("Loading %s\n", argv[2]);
        db = load_db(argv[2], FLOAT);

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);

        /* Ignore class data from the file to evaluate */
        memset(db->klass, 0, db->count * sizeof(int));

        printf("NN serial\n");
        gettimeofday(&start, NULL);
        nn_float(db->dimensions, train_db->count, train_db->data,
                 train_db->klass, db->count, db->data, db->klass);
        gettimeofday(&finish, NULL);
        printf("NN took %fs\n", elapsed_time(&start, &finish));

        printf("NN vector\n");
        gettimeofday(&start, NULL);
        nn_floats(db->dimensions, train_db->count, train_db->data,
                  train_db->klass, db->count, db->data, db->klass);
        gettimeofday(&finish, NULL);
        printf("NN took %fs\n", elapsed_time(&start, &finish));


        free_db(train_db);
        free_db(db);
        /*
        printf("Verifying\n");
        for (i = 0;  i < count;  i++)
                if (classes[i] != computed_classes[i])
                        printf("classes[%d] = %d,  computed_classes[%d] = %d\n",
                               i, classes[i], i, computed_classes[i]);

        */
        return EXIT_SUCCESS;
}

