#include "util.h"
#include "db.h"
#include "nn.h"
#include "stats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#define DEFAULT_RUNS  3

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
"                     Only numeric values are accepted.  By default, FLOAT is\n"
"                     used.\n\n"
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
        char *progname, *trfilename, *dumpfile, *typelabel;
        int cmdopt, has_opts, want_sequential, r, runs;
        enum datatype type;
        struct db *db, *train_db;
        struct timestats *ts;
        struct stats sts;

        progname = argv[0];
        dumpfile = NULL;
        runs = 3;
        type = FLOAT;
        typelabel = "FLOAT";
        want_sequential = 0;
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
                        if (dumpfile != NULL)
                                free(dumpfile);
                        dumpfile = xstrcat(optarg, NULL);
                        runs = 1;
                        break;
                case 'h':
                        usage(progname);
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
                        want_sequential = 1;
                        break;
                case 't':
                        switch (atoi(optarg))
                        {
                        case BYTE:    type = BYTE;    typelabel = "BYTE";    break;
                        case SHORT:   type = SHORT;   typelabel = "SHORT";   break;
                        case INTEGER: type = INTEGER; typelabel = "INTEGER"; break;
                        case FLOAT:   type = FLOAT;   typelabel = "FLOAT";   break;
                        case DOUBLE:  type = DOUBLE;  typelabel = "DOUBLE";  break;
                        default: quit("Invalid type: %s", optarg);
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
        }
        argc -= optind;
        argv += optind;
        if (argc < 1)
        {
                fputs("Please specify a file to load.\n\n", stderr);
                usage(progname);
        }

        trfilename = xstrcat(argv[0], ".t");
        printf("Loading %s as %s\n", trfilename, typelabel);
        train_db = load_db(trfilename, type, !want_sequential);
        free(trfilename);

        printf("Loading %s as %s\n\n", argv[0], typelabel);
        db = load_db(argv[0], type, !want_sequential);

        if (db->dimensions != train_db->dimensions)
                quit("Dimensions do not match (%d != %d)", db->dimensions,
                      train_db->dimensions);

        ts = stats_prepare(runs);
        for (r = 0;  r < runs;  r++)
        {
                printf("Run %d of %d\n", r + 1, runs);
                if (want_sequential)
                {
                        stats_start(ts);
                        nn_seq(type, db->dimensions, train_db->count,
                               train_db->data, train_db->klass, db->count,
                               db->data, db->klass);
                        stats_stop(ts);
                }
                else
                {
                        stats_start(ts);
                        nn_vect(type, db->dimensions, train_db->count,
                                train_db->data, train_db->klass, db->count,
                                db->data, db->klass);
                        stats_stop(ts);
                }
        }
        stats_calculate(ts, &sts);
        printf("\nStatistics\n\n");
        printf("- Minimum time: %lf secs\n", sts.minimum);
        printf("- Maximum time: %lf secs\n", sts.maximum);
        printf("- Average time: %lf secs\n", sts.mean);
        printf("- Standard deviation: %lf secs\n\n", sts.deviation);

        if (dumpfile != NULL)
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

