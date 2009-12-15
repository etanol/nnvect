OPENCL_HEADER_PATH := /opt/sdk/opencl/OpenCL/common/inc

CFLAGS    ?= -O3 -fomit-frame-pointer
CFLAGS    += -iquote .. -I$(OPENCL_HEADER_PATH)
LDFLAGS   += -lOpenCL

PROGRAMS := gpu_metrics

gpu_metrics_SOURCES := gpu_metrics.c util.c

include ../auto.mk

