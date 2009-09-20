#include "util.h"
#include "db.h"
#include "nn.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main (int argc, char **argv)
{
        int alen;
        char *trfilename;
        struct db *db, *train_db;
        struct timeval start, finish;

        if (argc < 2)
                quit("No matrix file specified");

        alen = strlen(argv[1]);
        trfilename = xmalloc(alen + 4);
        snprintf(trfilename, alen + 3, "%s.t", argv[1]);
        printf("Loading %s\n", trfilename);
        train_db = load_db(trfilename, FLOAT);
        free(trfilename);

        printf("Loading %s\n", argv[1]);
        db = load_db(argv[1], FLOAT);

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);

        /* Ignore class data from the file to evaluate */
        memset(db->klass, 0, db->count * sizeof(int));

        printf("NN sequential\n");
        gettimeofday(&start, NULL);
        nn_float_seq(db->dimensions, train_db->count, train_db->data,
                     train_db->klass, db->count, db->data, db->klass);
        gettimeofday(&finish, NULL);
        printf("NN took %fs\n", elapsed_time(&start, &finish));

        printf("NN vector\n");
        gettimeofday(&start, NULL);
        nn_float_vect(db->dimensions, train_db->count, train_db->data,
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

