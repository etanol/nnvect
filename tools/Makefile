CFLAGS ?= -O3 -fomit-frame-pointer
override CFLAGS += -iquote ..

PROGRAMS := shrink_db

shrink_db_SOURCES := shrink_db.c db.c util.c

vpath %.c ..
include ../auto.mk

