#include "stats.h"
#include "util.h"

#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

struct timestats
{
        int runs;
        int current;
        struct timeval reference;
        double partial;
        double *measure;
};


static inline double elapsed_time (struct timeval *begin, struct timeval *end)
{
        return (end->tv_sec - begin->tv_sec) + (end->tv_usec - begin->tv_usec)
               * 1.0e-6;
}


struct timestats *prepare_stats (int runs)
{
        struct timestats *ts;

        ts = xmalloc(sizeof(struct timestats) + sizeof(double) * runs);
        memset(ts, 0, sizeof(struct timestats) + sizeof(double) * runs);
        ts->runs = runs;
        ts->measure = (double *) (ts + 1);
        return ts;
}


void start_run (struct timestats *ts)
{
        ts->partial = 0.0;
        gettimeofday(&ts->reference, NULL);
}


void pause_run (struct timestats *ts)
{
        struct timeval t;

        gettimeofday(&t, NULL);
        ts->partial += elapsed_time(&ts->reference, &t);
        debug("Partial %d = %lf secs", ts->current, ts->partial);
}


void continue_run (struct timestats *ts)
{
        gettimeofday(&ts->reference, NULL);
}


void stop_run (struct timestats *ts)
{
        struct timeval t;

        if (ts->current >= ts->runs)
                warning("Maximum number of runs reached (%d >= %d), skipping",
                        ts->current, ts->runs);

        gettimeofday(&t, NULL);
        ts->measure[ts->current] = elapsed_time(&ts->reference, &t) +
                                   ts->partial;
        debug("Measure %d = %lf secs", ts->current, ts->measure[ts->current]);
        ts->current++;
}


double get_last_run_time (struct timestats *ts)
{
        if (ts->current <= 0 || ts->current > ts->runs)
                return -1.0;
        return ts->measure[ts->current - 1];
}


void calculate_stats (struct timestats *ts, struct stats *sts)
{
        int i;
        double diff;

        sts->minimum = DBL_MAX;
        sts->maximum = 0.0;
        sts->mean = 0.0;
        sts->deviation = 0.0;

        if (ts->current < ts->runs)
                ts->runs = ts->current;

        for (i = 0;  i < ts->runs;  i++)
        {
                if (ts->measure[i] < sts->minimum)
                        sts->minimum = ts->measure[i];
                if (ts->measure[i] > sts->maximum)
                        sts->maximum = ts->measure[i];
                sts->mean += ts->measure[i];
        }
        sts->mean /= ts->runs;

        for (i = 0;  i < ts->runs;  i++)
        {
                diff = ts->measure[i] - sts->mean;
                sts->deviation += diff * diff;
        }
        sts->deviation /= ts->runs;
        sts->deviation = sqrt(sts->deviation);

        free(ts);
}

