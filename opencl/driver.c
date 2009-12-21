#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
#include "db.h"
#include "util.h"
#include "stats.h"

#define KERNELFILE   "nn.cl"
#define KERNELNAME   "nn"
#define DEFAULT_RUNS  3

static char OptString[] = "hk:op:r:";

static struct option LongOpts[] = {
        { "help",    no_argument,       NULL, 'h' },
        { "kernel",  required_argument, NULL, 'k' },
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
"    -k, --kernel=NAME  Use kernel NAME instead of default (\"%s\").\n\n"
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
"      under the same path.\n\n", progname, KERNELNAME, KERNELFILE);

        exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
        char *progname, *fullname, *dumpfile, *kernelfile, *kernelname;
        struct db *test_db, *train_db;
        int i, hits, *result;
        int cmdopt, has_opts, r, runs;
        struct gpu *nv;
        struct timestats *ts;

        progname = argv[0];
        dumpfile = NULL;
        kernelfile = NULL;
        kernelname = NULL;
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
                case 'k':
                        if (kernelname != NULL)
                                free(kernelname);
                        kernelname = xstrcat(optarg, NULL);
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
        if (kernelname == NULL)
                kernelname = KERNELNAME;
        argc -= optind;
        argv += optind;
        if (argc < 1)
        {
                fprintf(stderr, "Please specify a file to load.\n\n");
                usage(progname);
        }

        /* Load the databases */
        fullname = xstrcat(argv[0], ".trn");
        printf("Loading %s\n\n", fullname);
        train_db = load_db(fullname, FLOAT, 0, 2, 0);
        free(fullname);
        print_db_info(train_db);

        fullname = xstrcat(argv[0], ".tst");
        printf("\nLoading %s\n\n", fullname);
        test_db = load_db(fullname, FLOAT, 0, 2, 0);
        free(fullname);
        print_db_info(test_db);

        if (test_db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", test_db->dimensions,
                     train_db->dimensions);

        result = xmalloc(test_db->count * sizeof(int));
        printf("\nSetting up GPU\n");
        nv = create_gpu(kernelfile, kernelname);
        printf("Feeding GPU with data\n\n");
        send_nn_arguments(nv, train_db, test_db);

        ts = prepare_stats(runs);
        for (r = 0;  r < runs;  r++)
        {
                double t;

                printf("Run %d of %d ... ", r + 1, runs);
                fflush(stdout);
                start_run(ts);
                execute_kernel(nv);
                stop_run(ts);
                t = get_last_run_time(ts);
                printf("%lf s\n", t);
        }

        get_nn_result(nv, test_db->count * sizeof(int), result);
        destroy_gpu(nv);

        /* Measure accuracy */
        hits = 0;
        for (i = 0;  i < test_db->count;  i++)
                if (test_db->klass[i] == result[i])
                        hits++;
        printf("\nClassification accuracy: %.2lf %%\n\n",
               (100.0 * hits) / test_db->count);

        if (dumpfile == NULL)
        {
                struct stats sts;

                calculate_stats(ts, &sts);
                printf("Timing statisticis\n\n");
                printf("- Minimum time: %lf s\n", sts.minimum);
                printf("- Maximum time: %lf s\n", sts.maximum);
                printf("- Average time: %lf s\n", sts.mean);
                printf("- Standard deviation: %lf s\n\n", sts.deviation);
        }
        else
        {
                FILE *f;

                f = fopen(dumpfile, "w");
                if (f == NULL)
                        fatal("Could not open '%s'", dumpfile);
                for (i = 0;  i < test_db->count;  i++)
                        fprintf(f, "%d\n", test_db->klass[i]);
                fclose(f);
        }

        free(result);
        free_db(train_db);
        free_db(test_db);
        return 0;
}

