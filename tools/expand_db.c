#include "db.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s source dest\n\n"
"NOTE: Just the common prefix of all the files needs to be specified as input.\n"
"      The \".trn\", \".tst\" suffixes are added automatically and the\n"
"      corresponding files are assumed to reside under the same path.\n\n", progname);

        exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
        struct db *db;
        char *fullname;
        FILE *f;
        int n, d, i, e;

        if (argc < 3)
                usage(argv[0]);

        fullname = xstrcat(argv[1], ".trn");
        printf("Loading training data (%s)\n", fullname);
        db = load_db(fullname, FLOAT, 0, 0);
        free(fullname);

        fullname = xstrcat(argv[2], ".ascii");
        f = fopen(fullname, "wt");
        if (f == NULL)
                fatal("Could not create %s", fullname);
        fprintf(f, "%d %d 2\n", db->dimensions, db->count);

        for (n = 0;  n < db->count;  n++)
        {
                i = n * db->dimensions;
                fprintf(f, "%f", ((float *) db->data)[i]);
                for (d = 1;  d < db->dimensions;  d++)
                        fprintf(f, " %f", ((float *) db->data)[i + d]);
                fprintf(f, "\n");
        }

        e = fclose(f);
        if (e == EOF)
                error("Closing %s", fullname);
        free(fullname);
        free_db(db);

        fullname = xstrcat(argv[1], ".tst");
        printf("Loading test data (%s)\n", fullname);
        db = load_db(fullname, FLOAT, 0, 0);
        free(fullname);

        fullname = xstrcat(argv[2], ".tst");
        f = fopen(fullname, "wt");
        if (f == NULL)
                fatal("Could not create %s", fullname);

        for (n = 0;  n < db->count;  n++)
        {
                i = n * db->dimensions;
                fprintf(f, "-1");
                for (d = 0;  d < db->dimensions;  d++)
                        fprintf(f, ",%f", ((float *) db->data)[i + d]);
                fprintf(f, "\n");
        }
        fprintf(f, "-0\n");

        e = fclose(f);
        if (e == EOF)
                error("Closing %s", fullname);
        free(fullname);
        free_db(db);

        return 0;
}

