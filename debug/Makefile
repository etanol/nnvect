vpath %.c ..

CFLAGS  := -DDEBUG -O0 -g -msse4.1
LDFLAGS := -g -lm

PROGRAMS := simple unroll2 unroll4

common_sources := bench.c db.c knn.c nn.c util.c stats.c
simple_SOURCES := $(common_sources) nnsca-simple.c nnvec-simple.c
unroll2_SOURCES := $(common_sources) nnsca-unroll2.c nnvec-unroll2.c
unroll4_SOURCES := $(common_sources) nnsca-unroll4.c nnvec-unroll4.c

include ../auto.mk

