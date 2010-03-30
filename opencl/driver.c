#include <getopt.h>
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "db.h"
#include "util.h"
#include "stats.h"
#include "card.h"

#define KERNELFILE   "nn.cl"
#define DEFAULT_RUNS  3

static char OptString[] = "ho:p:r:";

static struct option LongOpts[] = {
        { "help",    no_argument,       NULL, 'h' },
        { "output",  required_argument, NULL, 'o' },
        { "program", required_argument, NULL, 'p' },
        { "runs",    required_argument, NULL, 'r' },
        { NULL,      0,                 NULL, 0   }
};


static void usage (const char *progname)
{
        fprintf(stderr,
"Usage: %s [options] dbfile\n\n"
"Where possible options are:\n\n"
"    -h, --help         This help.\n\n"
"    -o, --output=FILE  Save the calculated results to FILE in order to\n"
"                       compare them against valid solutions.\n\n"
"    -p, --program=FILE Load the OpenCL kernel code from FILE.  By default\n"
"                       the kernels are loaded from \"%s\".\n\n"
"    -r, --runs=N       Execute N runs for statistical purposes.  In the\n"
"                       absence of this option, three runs are performed.\n"
"                       This parameter is ignored when combined with \"-o\"\n"
"                       or \"--output\".\n\n"
"NOTE: Just the common prefix of all the files needs to be specified as input.\n"
"      The \".trn\", \".tst\", \".trn.info\" and \".tst.info\" suffixes are\n"
"      added automatically and the corresponding files are assumed to reside\n"
"      under the same path.\n\n", progname, KERNELFILE);

        exit(EXIT_FAILURE);
}


static inline double ncs (struct db *train, struct db *test, double secs)
{
        debug("Dividing %lf / %lf", (secs * GPU_HZ), (double) test->dimensions *
                                                     test->count * train->count);
        return (secs * GPU_HZ) / ((double) test->dimensions * test->count *
                                  train->count);
}


int main (int argc, char **argv)
{
        char *progname, *fullname, *dumpfile, *kernelfile;
        struct db *test_db, *train_db;
        int i, hits, *result;
        int cmdopt, has_opts, r, runs, devices;
        struct timestats *ts;

        progname = argv[0];
        dumpfile = NULL;
        kernelfile = NULL;
        runs = DEFAULT_RUNS;
        has_opts = 1;

        do {
                cmdopt = getopt_long(argc, argv, OptString, LongOpts, NULL);
                if (cmdopt == -1)
                        break;

                switch (cmdopt)
                {
                case 'h':
                        usage(progname);
                        break;
                case 'o':
                        if (dumpfile != NULL)
                                free(dumpfile);
                        dumpfile = xstrcat(optarg, NULL);
                        runs = 1;
                        break;
                case 'p':
                        if (kernelfile != NULL)
                                free(kernelfile);
                        kernelfile = xstrcat(optarg, NULL);
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
                case '?':
                        fputs("\n", stderr);
                        usage(progname);
                        break;
                default:
                        fprintf(stderr, "Unrecognized option -- %c\n\n", cmdopt);
                        usage(progname);
                        break;
                }
        } while (1);
        if (kernelfile == NULL)
                kernelfile = KERNELFILE;
        argc -= optind;
        argv += optind;
        if (argc < 1)
        {
                fprintf(stderr, "Please specify a file to load.\n\n");
                usage(progname);
        }

        /* Prepare the GPU */
        printf("Setting up GPU\n\n");
        devices = init_gpu(kernelfile);

        /* Load the databases */
        fullname = xstrcat(argv[0], ".trn");
        printf("Loading %s\n\n", fullname);
        train_db = load_db_transposed(fullname, FLOAT, 1, 16, 4);
        free(fullname);
        print_db_info(train_db);

        fullname = xstrcat(argv[0], ".tst");
        printf("\nLoading %s\n\n", fullname);
        test_db = load_db_transposed(fullname, FLOAT, devices, 64, 4);
        free(fullname);
        print_db_info(test_db);

        if (test_db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", test_db->dimensions,
                     train_db->dimensions);
        result = xmalloc(test_db->count * sizeof(int));

        printf("\nFeeding GPU with data\n\n");
        prepare_invocations(train_db, test_db);

        ts = prepare_stats(runs);
        for (r = 0;  r < runs;  r++)
        {
                double t;

                printf("Run %d of %d ... ", r + 1, runs);
                fflush(stdout);
                start_run(ts);
                invoke_kernels();
                stop_run(ts);
                t = get_last_run_time(ts);
                printf("%lf s  (%.04lf NCs)\n", t, ncs(train_db, test_db, t));
        }

        get_nn_result(test_db->count * sizeof(int), result);
        destroy_gpu();

        /* Measure accuracy */
        hits = 0;
        for (i = 0;  i < test_db->real_count;  i++)
                if (test_db->klass[i] == result[i])
                        hits++;
        printf("\nClassification accuracy: %.2lf %%\n\n",
               (100.0 * hits) / test_db->count);

        if (dumpfile == NULL)
        {
                struct stats sts;

                calculate_stats(ts, &sts);
                printf("Timing statisticis\n\n");
                printf("- Minimum time: %lf s  (%.04lf NCs)\n", sts.minimum,
                       ncs(train_db, test_db, sts.minimum));
                printf("- Maximum time: %lf s  (%.04lf NCs)\n", sts.maximum,
                       ncs(train_db, test_db, sts.maximum));
                printf("- Average time: %lf s  (%.04lf NCs)\n", sts.mean,
                       ncs(train_db, test_db, sts.mean));
                printf("- Standard deviation: %lf s\n\n", sts.deviation);
        }
        else
        {
                FILE *f;

                f = fopen(dumpfile, "w");
                if (f == NULL)
                        fatal("Could not open '%s'", dumpfile);
                for (i = 0;  i < test_db->real_count;  i++)
                        fprintf(f, "%d\n", result[i]);
                fclose(f);
        }

        free(result);
        free_db(train_db);
        free_db(test_db);
        return 0;
}

