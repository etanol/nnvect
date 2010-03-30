#include "util.h"
#include "db.h"
#include "nn.h"
#include "knn.h"
#include "stats.h"
#include "machine.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#define DEFAULT_RUNS  3

static char OptString[] = "B:b:hk:o:r:st:";

static struct option LongOpts[] = {
        { "superblock", required_argument, NULL, 'B' },
        { "blocksize",  required_argument, NULL, 'b' },
        { "help",       no_argument,       NULL, 'h' },
        { "neighbours", required_argument, NULL, 'k' },
        { "output",     required_argument, NULL, 'o' },
        { "runs",       required_argument, NULL, 'r' },
        { "scalar",     no_argument,       NULL, 's' },
        { "type",       required_argument, NULL, 't' },
        { NULL,         0,                 NULL, 0   }
};


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s [options] dbfile\n\n"
"Where possible options are:\n\n"
"    -B, --superblock=SIZE The maximum amount of bytes to read from the\n"
"                          test array at once.  Setting this value to zero\n"
"                          disables blocking at this level.\n\n"
"    -b, --blocksize=SIZE  Same as \"-B\" or \"--superblock\" but applied\n"
"                          for the training array.  Using any of the block\n"
"                          size options swithes to a blocking implementation.\n\n"
"    -h, --help            This help.\n\n"
"    -k, --neighbours=K    Classify according to the K nearest neighbors.\n"
"                          Setting this value to a number greater than 1\n"
"                          forces \"-s\" or \"--scalar\".  By default only\n"
"                          the nearest neighbour (K = 1) is calculated.\n\n"
"    -o, --output=FILE     Save the calculated results to FILE in order to\n"
"                          compare them against valid solutions.\n\n"
"    -r, --runs=N          Execute N runs for statistical purposes.  In the\n"
"                          absence of this option, three runs are performed.\n"
"                          This parameter is ignored when combined with\n"
"                          \"-o\" or \"--output\".\n\n"
"    -s, --scalar          Run a non-vectorized version of the algorithm\n"
"                          instead of the vectorized one.\n\n"
"    -t, --type=TYPE       Load data as the given TYPE.  Possible types are:\n"
"                              byte    (%d bytes)\n"
"                              short   (%d bytes)\n"
"                              int     (%d bytes)\n"
"                              float   (%d bytes)\n"
"                              double  (%d bytes)\n\n"
"NOTE: Just the common prefix of all the files needs to be specified as input.\n"
"      The \".trn\", \".tst\", \".trn.info\" and \".tst.info\" suffixes are\n"
"      added automatically and the corresponding files are assumed to reside\n"
"      under the same path.\n\n", progname,
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


static inline double ncs (struct db *train, struct db *test, double secs)
{
        debug("Dividing %lf / %lf", (secs * CPU_HZ), (double) test->dimensions *
                                                     test->count * train->count);
        return (secs * CPU_HZ) / ((double) test->dimensions * test->count *
                                  train->count);
}


int main (int argc, char **argv)
{
        char *progname, *fullname, *dumpfile, *typelabel;
        int cmdopt, has_opts, want_scalar, padding_type, r, runs, i, hits, k;
        int block_limit, test_block_limit, mem_alignment;
        int *original;
        enum valuetype type;
        struct db *db, *train_db;
        struct timestats *ts;
        struct nbhood *nbh;

        progname = argv[0];
        dumpfile = NULL;
        block_limit = 0;
        test_block_limit = 0;
        mem_alignment = 16;
        runs = DEFAULT_RUNS;
        type = FLOAT;
        typelabel = NULL;
        want_scalar = 0;
        padding_type = 1;
        has_opts = 1;
        k = 1;
        nbh = NULL;

        while (has_opts)
        {
                cmdopt = getopt_long(argc, argv, OptString, LongOpts, NULL);
                switch (cmdopt)
                {
                case -1:
                        has_opts = 0;
                        break;
                case 'B':
                        test_block_limit = atoi(optarg);
                        if (test_block_limit < 0)
                        {
                                test_block_limit = 0;
                                warning("Invalied superblock: %s", optarg);
                        }
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
                case 'k':
                        k = atoi(optarg);
                        if (k <= 0)
                        {
                                k = 1;
                                warning("Invalid value for k: %s", optarg);
                        }
                        else if (k > 1)
                        {
                                want_scalar = 1;
                                mem_alignment = 0;
                        }
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
                        mem_alignment = 0;
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

        fullname = xstrcat(argv[0], ".trn");
        printf("Loading %s as %ss\n\n", fullname,
               (typelabel == NULL ? "float" : typelabel));
        train_db = load_db(fullname, type, block_limit, mem_alignment);
        free(fullname);
        if (train_db->block_items > 0)
                train_db->block_items = adjusted_block_count(train_db->block_items);
        print_db_info(train_db);

        fullname = xstrcat(argv[0], ".tst");
        printf("\nLoading %s as %ss\n\n", fullname,
               (typelabel == NULL ? "float" : typelabel));
        db = load_db(fullname, type, test_block_limit, mem_alignment);
        free(fullname);
        print_db_info(db);
        printf("\n");

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);
        if (k > 1)
                nbh = create_neighbourhood(k, db);

        /* Save original results for accuracy testing */
        original = xmalloc(sizeof(int) * db->count);
        memcpy(original, db->klass, sizeof(int) * db->count);
        memset(db->klass, 0, sizeof(int) * db->count);

        ts = prepare_stats(runs);
        for (r = 0;  r < runs;  r++)
        {
                double t;

                printf("Run %d of %d ... ", r + 1, runs);
                fflush(stdout);
                if (k > 1)
                {
                        clear_neighbourhood(k, db, nbh);
                        start_run(ts);
                        knn(k, type, train_db, db, nbh);
                        stop_run(ts);
                }
                else
                {
                        clear_distances(db);
                        start_run(ts);
                        nn(type, want_scalar, train_db, db);
                        stop_run(ts);
                }
                t = get_last_run_time(ts);
                printf("%lf s  (%.04lf NCs)\n", t, ncs(train_db, db, t));
        }
        if (k > 1)
        {
                classify(k, db, nbh);
                free_neighbourhood(nbh);
        }

        /* Measure accuracy */
        hits = 0;
        for (i = 0;  i < db->count;  i++)
                if (db->klass[i] == original[i])
                        hits++;
        free(original);
        printf("\nClassification accuracy: %.2lf %%\n\n",
               (100.0 * hits) / db->count);

        if (dumpfile == NULL)
        {
                struct stats sts;

                calculate_stats(ts, &sts);
                printf("Timing statistics\n\n");
                printf("- Minimum time: %lf s  (%.04lf NCs)\n", sts.minimum,
                       ncs(train_db, db, sts.minimum));
                printf("- Maximum time: %lf s  (%.04lf NCs)\n", sts.maximum,
                       ncs(train_db, db, sts.maximum));
                printf("- Average time: %lf s  (%.04lf NCs)\n", sts.mean,
                       ncs(train_db, db, sts.mean));
                printf("- Standard deviation: %lf s\n\n", sts.deviation);
        }
        else
        {
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

