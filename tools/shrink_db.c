#include "db.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s source dest\n\n"
"NOTE: Just the common prefix of all the files needs to be specified as input.\n"
"      The \".trn\", \".tst\", \".trn.info\" and \".tst.info\" suffixes are\n"
"      added automatically and the corresponding files are assumed to reside\n"
"      under the same path.\n\n", progname);

        exit(EXIT_FAILURE);
}


static int *create_map (struct db *db, int *valid)
{
        int *map;
        int i, lvi;  /* Last Valid Index */

        map = xmalloc(sizeof(int) * db->dimensions);
        lvi = 0;
        for (i = 0;  i < db->dimensions;  i++)
                if (valid[i])
                {
                        map[i] = lvi;
                        lvi++;
                }
                else
                        map[i] = -1;

        return map;
}


static void dump_valid_db_data (struct db *db, int *map, const char *filename)
{
        FILE *f;
        int n, d, i, e;

        f = fopen(filename, "wt");
        if (f == NULL)
                fatal("Could not create %s", filename);

        if (db->has_floats)
        {
                for (n = 0;  n < db->count;  n++)
                {
                        fprintf(f, "%d", db->klass[n]);
                        i = n * db->dimensions;
                        for (d = 0;  d < db->dimensions;  d++)
                                if (map[d] >= 0 && ((float *) db->data)[i + d] != 0.0f)
                                        fprintf(f, " %d:%f", map[d],
                                                ((float *) db->data)[i + d]);
                        fprintf(f, "\n");
                }
        }
        else
        {
                for (n = 0;  n < db->count;  n++)
                {
                        fprintf(f, "%d", db->klass[n]);
                        i = n * db->dimensions;
                        for (d = 0;  d < db->dimensions;  d++)
                                if (map[d] >= 0 && ((float *) db->data)[i + d] != 0.0f)
                                        fprintf(f, " %d:%d", map[d], (int)
                                                ((float *) db->data)[i + d]);
                        fprintf(f, "\n");
                }
        }

        e = fclose(f);
        if (e == EOF)
                error("Closing %s", filename);
}


int main (int argc, char **argv)
{
        struct db *db;
        float *value;
        int *valid, *map;
        char *fullname;
        int n, d, i;

        if (argc < 3)
                usage(argv[0]);

        fullname = xstrcat(argv[1], ".trn");
        printf("Loading training data (%s)\n", fullname);
        db = load_db(fullname, FLOAT, 0, 0);
        free(fullname);

        value = xmalloc(sizeof(float) * db->dimensions);
        valid = xmalloc(sizeof(int) * db->dimensions);
        memset(value, 0, sizeof(float) * db->dimensions);
        memset(valid, 0, sizeof(int) * db->dimensions);

        for (d = 0;  d < db->dimensions;  d++)
                value[d] = ((float *) db->data)[d];

        for (n = 1;  n < db->count;  n++)
        {
                i = n * db->dimensions;
                for (d = 0;  d < db->dimensions;  d++)
                        valid[d] = valid[d] ||
                                   value[d] != ((float *) db->data)[i + d];
        }

        map = create_map(db, valid);
        free(value);
        free(valid);

        fullname = xstrcat(argv[2], ".trn");
        printf("Dumping filtered training data (%s)\n", fullname);
        dump_valid_db_data(db, map, fullname);
        free(fullname);
        free_db(db);

        fullname = xstrcat(argv[1], ".tst");
        printf("Loading test data (%s)\n", fullname);
        db = load_db(fullname, FLOAT, 0, 0);
        free(fullname);

        fullname = xstrcat(argv[2], ".tst");
        printf("Dumping filtered test data (%s)\n", fullname);
        dump_valid_db_data(db, map, fullname);
        free(fullname);
        free(map);
        free_db(db);

        return 0;
}

