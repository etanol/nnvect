#include "util.h"
#include "db.h"
#include "nn.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

static char OptString[] = "d:hr:st:";

static struct option LongOpts[] = {
        { "dump",   required_argument, NULL, 'd' },
        { "help",   no_argument,       NULL, 'h' },
        { "runs",   required_argument, NULL, 'r' },
        { "scalar", no_argument,       NULL, 's' },
        { "type",   required_argument, NULL, 't' },
        { NULL,     0,                 NULL, 0   }
};


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s [options] dbfile\n\n"
"Where possible options are:\n\n"
"    -d, --dump=FILE  Save the calculated results to FILE in order to compare\n"
"                     them against valid solutions.\n\n"
"    -h, --help       This help.\n\n"
"    -r, --runs=N     Execute N runs for statistical purposes.  In the absence\n"
"                     of this option, three runs are performed.  This parameter\n"
"                     is ignored when combined with \"-d\" or \"--dump\".\n\n"
"    -s, --scalar     Run a non-vectorized version of the algorithm.  Vectorized\n"
"                     versions are used by default.\n\n"
"    -t, --type=TYPE  Load data as the given TYPE.  Possible types are:\n"
"                         %d  BYTE    (%d bytes)\n"
"                         %d  SHORT   (%d bytes)\n"
"                         %d  INTEGER (%d bytes)\n"
"                         %d  FLOAT   (%d bytes)\n"
"                         %d  DOUBLE  (%d bytes)\n"
"                     Only numeric values are accepted.  By default, INTEGER and\n"
"                     FLOAT are used where appropriate.\n\n"
"NOTE: Just a single file needs to be specified as input.  However, the files\n"
"      \"dbfile.info\", \"dbfile.t\" and \"dbfile.t.info\" are assumed to reside\n"
"      under the same path as \"dbfile\".\n\n", progname,
               BYTE, (int) sizeof(char), SHORT, (int) sizeof(short),
               INTEGER, (int) sizeof(int), FLOAT, (int) sizeof(float),
               DOUBLE, (int) sizeof(double));

        exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
        char *progname = argv[0], *trfilename;
        int alen, cmdopt, has_opts;
        struct db *db, *train_db;
        struct timeval start, finish;

        has_opts = 1;
        while (has_opts)
        {
                cmdopt = getopt_long(argc, argv, OptString, LongOpts, NULL);
                switch (cmdopt)
                {
                case -1:
                        has_opts = 0;
                        break;
                case 'd':
                case 'h':
                case 'r':
                case 's':
                case 't':
                        fprintf(stderr, "Option not yet supported -- %c\n", cmdopt);
                        return EXIT_FAILURE;
                case '?':
                        fputs("\n", stderr);
                        usage(progname);
                        break;
                default:
                        fprintf(stderr, "Unrecognized option -- %c\n\n", cmdopt);
                        usage(progname);
                        break;
                }
        }
        argc -= optind;
        argv += optind;
        if (argc < 1)
        {
                fputs("Please specify a file to load.\n\n", stderr);
                usage(progname);
        }

        alen = strlen(argv[0]);
        trfilename = xmalloc(alen + 4);
        snprintf(trfilename, alen + 3, "%s.t", argv[0]);
        printf("Loading %s\n", trfilename);
        train_db = load_db(trfilename, FLOAT);
        free(trfilename);

        printf("Loading %s\n", argv[0]);
        db = load_db(argv[0], FLOAT);

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);

        /* Ignore class data from the file to evaluate */
        memset(db->klass, 0, db->count * sizeof(int));

        printf("NN sequential\n");
        gettimeofday(&start, NULL);
        nn_seq(FLOAT, db->dimensions, train_db->count, train_db->data,
               train_db->klass, db->count, db->data, db->klass);
        gettimeofday(&finish, NULL);
        printf("NN took %fs\n", elapsed_time(&start, &finish));

        printf("NN vector\n");
        gettimeofday(&start, NULL);
        nn_vect(FLOAT, db->dimensions, train_db->count, train_db->data,
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

