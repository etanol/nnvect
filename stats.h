#ifndef __nnvect_stats
#define __nnvect_stats

/* Opaque structure */
struct timestats;

/* Results structure */
struct stats
{
        double minimum;
        double maximum;
        double mean;
        double deviation;
};

struct timestats *prepare_stats     (int);
void              start_run         (struct timestats *);
void              pause_run         (struct timestats *);
void              continue_run      (struct timestats *);
void              stop_run          (struct timestats *);
double            get_last_run_time (struct timestats *);
void              calculate_stats   (struct timestats *, struct stats *);

#endif /* __nnvect_stats */
