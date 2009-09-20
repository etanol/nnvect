MAKEFLAGS += -R -r
CFLAGS    ?= -O3 -fomit-frame-pointer -msse4.1
LDFLAGS   += -lm

PROGRAMS := simple unroll2 unroll4

simple_SOURCES := bench.c db.c nn.c util.c stats.c nnsca-simple.c nnvec-simple.c
unroll2_SOURCES := bench.c db.c nn.c util.c stats.c nnsca-unroll2.c nnvec-unroll2.c
unroll4_SOURCES := bench.c db.c nn.c util.c stats.c nnsca-unroll4.c nnvec-unroll4.c


all: $(PROGRAMS)

define gen_rules
depfiles := $(patsubst %.c,.%.d,$($(1)_SOURCES))
objects  += $(patsubst %.c,%.o,$($(1)_SOURCES))

$(1): $(patsubst %.c,%.o,$($(1)_SOURCES))
	@echo " Linking    $$@" && $(CC) $(LDFLAGS) -o $$@ $$^
endef

$(foreach p,$(PROGRAMS),$(eval $(call gen_rules,$(p))))


.SUFFIXES: .c .o
%.o: %.c
	@echo " Compiling  $@" && $(CC) -pipe -Wall -MMD -MF .$(<F:.c=.d) $(CFLAGS) -c -o $@ $<


.PHONY: clean
clean:
	@-echo " Cleaning" ; rm -f $(PROGRAMS) $(depfiles) $(objects)


-include $(depfiles)

