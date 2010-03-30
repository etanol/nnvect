OPENCL_HEADER_PATH := /opt/sdk/opencl/OpenCL/common/inc

# Default compiler flags (may be overriden)
CFLAGS ?= -O3 -fomit-frame-pointer

# Mandatory compiler flags
override CFLAGS  += -iquote .. -I$(OPENCL_HEADER_PATH)
override LDFLAGS += -lm -lOpenCL

PROGRAMS := gpu_metrics clnn

gpu_metrics_SOURCES := gpu_metrics.c gpu.c util.c
clnn_SOURCES := driver.c gpu.c db.c stats.c util.c

vpath %.c ..
include ../auto.mk

objects += card.h
driver.o: card.h
card.h: gpu_metrics
	@echo " Generating $@" && \
	hz=$$(./gpu_metrics|awk '$$2=="MHz"{printf "%f",$$4*1000000;exit}') && \
	{ \
		echo "/* Automatically generated, do not edit */" ; \
		echo ; \
		echo "#ifndef GPU_HZ" ; \
		echo "#define GPU_HZ  $$hz" ; \
		echo "#endif" ; \
	} > $@

