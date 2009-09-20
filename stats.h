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

struct timestats *stats_prepare   (int);
void              stats_start     (struct timestats *);
void              stats_stop      (struct timestats *);
void              stats_calculate (struct timestats *, struct stats *);

#endif /* __nnvect_stats */
