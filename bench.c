#include <stdio.h>
#include <stdlib.h>
#include "misc.h"

extern void load_db (char *, float **, int **, int *, int *);
extern void nn (int, int, float *, int *, int, float *, int **);

int main (int argc, char **argv)
{
        float start, finish;
        float *features, *train_features;
        int *classes, *train_classes, *computed_classes;
        int dimensions, train_dimensions;
        int count, train_count;
        int i;

        if (argc < 3)
                fatal("Not enough arguments");

        printf("Loading %s\n", argv[1]);
        load_db(argv[1], &train_features, &train_classes, &train_dimensions,
                &train_count);
        printf("Loading %s\n", argv[2]);
        load_db(argv[2], &features, &classes, &dimensions, &count);

        printf("Dimensions %d, train dimensions %d\n", dimensions, train_dimensions);
        if (dimensions != train_dimensions)
                fatal("Dimensions do not match (%d != %d)", dimensions,
                      train_dimensions);

        computed_classes = calloc(count, sizeof(int));
        if (computed_classes == NULL)
                fatal("Not enough memory for computed classes");

        printf("NN\n");
        start = system_time();
        nn(dimensions, train_count, train_features, train_classes, count,
           features, &computed_classes);
        finish = system_time();
        printf("NN took %fms\n", finish - start);

        printf("Verifying\n");
        for (i = 0;  i < count;  i++)
                if (classes[i] != computed_classes[i])
                        printf("i = %d,  classes[i] = %d,  computed_classes[i] = %d",
                               i, classes[i], computed_classes[i]);

        return EXIT_SUCCESS;
}

