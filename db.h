#ifndef __nnvect_db
#define __nnvect_db

enum datatype
{
        BYTE,
        SHORT,
        INTEGER,
        FLOAT,
        DOUBLE
};

struct db
{
        enum datatype type;
        int count;
        int dimensions;
        int real_dimensions;
        int has_floats;
        int *klass;
        void *data;
};

struct db *load_db (const char *, enum datatype);
void       free_db (struct db *);

#endif /* __nnvect_db */
