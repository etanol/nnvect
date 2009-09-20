MAKEFLAGS += -R -r
CFLAGS    ?= -O3 -fomit-frame-pointer -msse4.1
LDFLAGS   += -lm

PROGRAMS := bench

bench_SOURCES := bench.c db.c nn.c nnsca.c nnvec.c util.c stats.c


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

