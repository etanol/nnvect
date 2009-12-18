OPENCL_HEADER_PATH := /opt/sdk/opencl/OpenCL/common/inc

CFLAGS    ?= -O3 -fomit-frame-pointer
CFLAGS    += -iquote .. -I$(OPENCL_HEADER_PATH)
LDFLAGS   += -lOpenCL

PROGRAMS := gpu_metrics clnn

gpu_metrics_SOURCES := gpu_metrics.c gpu.c util.c
clnn_SOURCES := driver.c gpu.c db.c stats.c util.c

vpath %.c ..
include ../auto.mk

