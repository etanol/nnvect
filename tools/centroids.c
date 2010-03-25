#include "db.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s source dest\n\n"
"Find the centroids of each class in \"source.trn\" and place them in\n"
"\"dest.trn\"\n\n"
"NOTE: Just the common prefix of all the files needs to be specified as input.\n"
"      The \".trn\" and \".trn.info\" and \".tst.info\" suffixes are added\n"
"      automatically and the corresponding files are assumed to reside under\n"
"      the same path.\n\n", progname);

        exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
        int li, n, d;
        float sum, value;
        int count;
        char *fullname;
        FILE *f;
        struct db *db;

        if (argc < 3)
                usage(argv[0]);

        fullname = xstrcat(argv[1], ".trn");
        printf("Loading training data (%s)\n", fullname);
        db = load_db(fullname, FLOAT, 0, 0);
        free(fullname);

        fullname = xstrcat(argv[2], ".trn");
        f = fopen(fullname, "wt");
        if (f == NULL)
                fatal("Cannot open %s for writing", fullname);
        free(fullname);

        for (li = 0;  li < db->label_count;  li++)
        {
                fprintf(f, "%d", db->label[li]);
                for (d = 0;  d < db->dimensions;  d++)
                {
                        sum = 0.0f;
                        count = 0;
                        for (n = 0;  n < db->count;  n++)
                                if (db->klass[n] == db->label[li])
                                {
                                        sum += ((float *) db->data)[n * db->dimensions + d];
                                        count++;
                                }
                        value = sum / count;
                        if (value != 0.0f)
                                fprintf(f, " %d:%f", d, value);
                }
                fprintf(f, "\n");
        }

        fclose(f);
        return 0;
}

