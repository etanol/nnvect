#include "util.h"
#include "db.h"
#include "nn.h"
#include "stats.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#define DEFAULT_RUNS  3

static char OptString[] = "b:ho:r:st:";

static struct option LongOpts[] = {
        { "blocksize", required_argument, NULL, 'b' },
        { "help",      no_argument,       NULL, 'h' },
        { "output",    required_argument, NULL, 'o' },
        { "runs",      required_argument, NULL, 'r' },
        { "scalar",    no_argument,       NULL, 's' },
        { "type",      required_argument, NULL, 't' },
        { NULL,        0,                 NULL, 0   }
};


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s [options] dbfile\n\n"
"Where possible options are:\n\n"
"    -b, --blocksize=SIZE  The maximum amount of bytes to read from the\n"
"                          training array at once.  Setting this value to\n"
"                          zero (the default) selects a non-blocking\n"
"                          implementation.\n\n"
"    -h, --help            This help.\n\n"
"    -o, --output=FILE     Save the calculated results to FILE in order to\n"
"                          compare them against valid solutions.\n\n"
"    -r, --runs=N          Execute N runs for statistical purposes.  In the\n"
"                          absence of this option, three runs are performed.\n"
"                          This parameter is ignored when combined with\n"
"                          \"-d\" or \"--dump\".\n\n"
"    -s, --scalar          Run a non-vectorized version of the algorithm\n"
"                          instead of the vectorized one.\n\n"
"    -t, --type=TYPE       Load data as the given TYPE.  Possible types are:\n"
"                              byte    (%d bytes)\n"
"                              short   (%d bytes)\n"
"                              int     (%d bytes)\n"
"                              float   (%d bytes)\n"
"                              double  (%d bytes)\n\n"
"NOTE: Just a single file needs to be specified as input.  However, the files\n"
"      \"dbfile.info\", \"dbfile.t\" and \"dbfile.t.info\" are assumed to reside\n"
"      under the same path as \"dbfile\".\n\n", progname,
                (int) sizeof(char), (int) sizeof(short), (int) sizeof(int),
                (int) sizeof(float), (int) sizeof(double));

        exit(EXIT_FAILURE);
}


static void strtolower (char *s)
{
        while (*s != '\0')
        {
                *s = tolower(*s);
                s++;
        }
}


int main (int argc, char **argv)
{
        char *progname, *trfilename, *dumpfile, *typelabel;
        int cmdopt, has_opts, want_scalar, r, runs;
        int block_limit;
        enum valuetype type;
        struct db *db, *train_db;
        struct timestats *ts;

        progname = argv[0];
        dumpfile = NULL;
        block_limit = 0;
        runs = 3;
        type = FLOAT;
        typelabel = NULL;
        want_scalar = 0;
        has_opts = 1;

        while (has_opts)
        {
                cmdopt = getopt_long(argc, argv, OptString, LongOpts, NULL);
                switch (cmdopt)
                {
                case -1:
                        has_opts = 0;
                        break;
                case 'b':
                        block_limit = atoi(optarg);
                        if (block_limit < 0)
                        {
                                block_limit = 0;
                                warning("Invalid blocksize: %s", optarg);
                        }
                        break;
                case 'h':
                        usage(progname);
                        break;
                case 'o':
                        if (dumpfile != NULL)
                                free(dumpfile);
                        dumpfile = xstrcat(optarg, NULL);
                        runs = 1;
                        break;
                case 'r':
                        if (dumpfile == NULL)
                        {
                                runs = atoi(optarg);
                                if (runs <= 0)
                                {
                                        runs = DEFAULT_RUNS;
                                        warning("Invalid value for runs: %s",
                                                optarg);
                                }
                        }
                        break;
                case 's':
                        want_scalar = 1;
                        break;
                case 't':
                        strtolower(optarg);
                        if (strcmp(optarg, "byte") == 0)
                                type = BYTE;
                        else if (strcmp(optarg, "short") == 0)
                                type = SHORT;
                        else if (strcmp(optarg, "int") == 0)
                                type = INT;
                        else if (strcmp(optarg, "float") == 0)
                                type = FLOAT;
                        else if (strcmp(optarg, "double") == 0)
                                type = DOUBLE;
                        else
                                quit("Invalid type: %s", optarg);

                        if (typelabel != NULL)
                                free(typelabel);
                        typelabel = xstrcat(optarg, NULL);
                        break;
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

        printf("Using %s NN.\n\n", (want_scalar ? "scalar" : "vectorized"));

        trfilename = xstrcat(argv[0], ".t");
        printf("Loading %s as %ss\n\n", trfilename,
               (typelabel == NULL ? "float" : typelabel));
        train_db = load_db(trfilename, type, block_limit, !want_scalar, 0);
        free(trfilename);
        print_db_info(train_db);

        printf("\nLoading %s as %ss\n\n", argv[0],
               (typelabel == NULL ? "float" : typelabel));
        db = load_db(argv[0], type, 0, !want_scalar, train_db->block_items > 0);
        memset(db->klass, 0, db->count * sizeof(int));
        print_db_info(db);
        printf("\n");

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);

        ts = prepare_stats(runs);
        for (r = 0;  r < runs;  r++)
        {
                printf("Run %d of %d ... ", r + 1, runs);
                fflush(stdout);
                start_run(ts);
                nn(type, want_scalar, train_db, db);
                stop_run(ts);
                printf("%lf s\n", get_last_run_time(ts));
        }

        if (dumpfile == NULL)
        {
                struct stats sts;

                calculate_stats(ts, &sts);
                printf("\nStatistics\n\n");
                printf("- Minimum time: %lf s\n", sts.minimum);
                printf("- Maximum time: %lf s\n", sts.maximum);
                printf("- Average time: %lf s\n", sts.mean);
                printf("- Standard deviation: %lf s\n\n", sts.deviation);
        }
        else
        {
                int i;
                FILE *f;

                f = fopen(dumpfile, "w");
                if (f == NULL)
                        fatal("Could not open %s", dumpfile);
                for (i = 0;  i < db->count;  i++)
                        fprintf(f, "%d\n", db->klass[i]);
                fclose(f);
        }

        free_db(train_db);
        free_db(db);
        return EXIT_SUCCESS;
}

