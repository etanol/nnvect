MAKEFLAGS += -R -r

BASE_CFLAGS := -pipe -MMD -Wall
CFLAGS := -O0 -g -march=pentium-m

bench: bench.o nn.o db.o misc.o
	$(CC) -g -o $@ $^

.SUFFIXES: .c .o
%.o: %.c
	$(CC) $(BASE_CFLAGS) $(CFLAGS) -c -o $@ $<

-include $(wildcard *.d)

