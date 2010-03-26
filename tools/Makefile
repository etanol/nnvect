CFLAGS ?= -O3 -fomit-frame-pointer
override CFLAGS += -iquote ..

PROGRAMS := shrink_db expand_db centroids

shrink_db_SOURCES := shrink_db.c db.c util.c
expand_db_SOURCES := expand_db.c db.c util.c
centroids_SOURCES := centroids.c db.c util.c

vpath %.c ..
include ../auto.mk

