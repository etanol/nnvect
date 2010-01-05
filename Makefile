# Uncomment this to permanently disable SSE 4.1 if it is not available.  The
# affected implementations will not be compiled.  It also work by setting the
# same variable from the command line
#SSE4 := n

ifeq ($(SSE4),n)
sse_flags := -DNO_SSE4 -mssse3
else
sse_flags := -msse4.1
endif

CFLAGS ?= -O3 -fomit-frame-pointer
override CFLAGS  += -fopenmp $(sse_flags)
override LDFLAGS += -lm -fopenmp

PROGRAMS := simple unroll2 unroll4

common_sources := bench.c db.c knn.c nn.c util.c stats.c
simple_SOURCES := $(common_sources) nnsca-simple.c nnvec-simple.c
unroll2_SOURCES := $(common_sources) nnsca-unroll2.c nnvec-unroll2.c
unroll4_SOURCES := $(common_sources) nnsca-unroll4.c nnvec-unroll4.c

include auto.mk

