CFLAGS ?= -O3 -fomit-frame-pointer
override CFLAGS += -iquote ..

PROGRAMS := shrink_db centroids

shrink_db_SOURCES := shrink_db.c db.c util.c
centroids_SOURCES := centroids.c db.c util.c

vpath %.c ..
include ../auto.mk

