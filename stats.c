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
        double *measure;
};


static inline double elapsed_time (struct timeval *begin, struct timeval *end)
{
        return (end->tv_sec - begin->tv_sec) + (end->tv_usec - begin->tv_usec)
               * 1.0e-6;
}


struct timestats *stats_prepare (int runs)
{
        struct timestats *ts;

        ts = xmalloc(sizeof(struct timestats) + sizeof(double) * runs);
        memset(ts, 0, sizeof(struct timestats) + sizeof(double) * runs);
        ts->runs = runs;
        ts->measure = (double *) (ts + 1);
        return ts;
}


void stats_start (struct timestats *ts)
{
        if (ts->current < ts->runs)
                gettimeofday(&ts->reference, NULL);
}


void stats_stop (struct timestats *ts)
{
        struct timeval t;

        if (ts->current < ts->runs)
        {
                gettimeofday(&t, NULL);
                ts->measure[ts->current] = elapsed_time(&ts->reference, &t);
                debug("Measure %d = %lf secs", ts->current, ts->measure[ts->current]);
                ts->current++;
        }
}


void stats_calculate (struct timestats *ts, struct stats *sts)
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

