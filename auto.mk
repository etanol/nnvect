MAKEFLAGS += -R -r

                     #####################################
                     #                                   #
                     #  Automatic build rule generation  #
                     #                                   #
                     #####################################

all: $(PROGRAMS)

define build_program
depfiles += $(patsubst %.c,.%.d,$($(1)_SOURCES))
objects  += $(patsubst %.c,%.o,$($(1)_SOURCES))

$(1): $(patsubst %.c,%.o,$($(1)_SOURCES))
	@echo " Linking    $$@" && $(CC) $(LDFLAGS) -o $$@ $$^
endef

$(foreach p,$(PROGRAMS),$(eval $(call build_program,$(p))))

.SUFFIXES: .c .o
%.o: %.c
	@echo " Compiling  $@" && $(CC) -pipe -Wall -MMD -MF .$(<F:.c=.d) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	@-echo " Cleaning" ; rm -f $(PROGRAMS) $(depfiles) $(objects)


-include $(depfiles)

