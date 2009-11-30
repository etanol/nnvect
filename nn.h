#ifndef __nnvect_nn
#define __nnvect_nn

/* Supported value types */
enum valuetype
{
        BYTE,
        SHORT,
        INT,
        FLOAT,
        DOUBLE
};

/* Forward declaration to avoid cyclic dependencies */
struct db;

int adjusted_block_count (int);
void clear_distances (struct db *);
void nn (enum valuetype, int, struct db *, struct db *);

#endif /* __nnvect_nn */
