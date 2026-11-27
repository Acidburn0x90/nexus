# Native hot-path helpers for Nexus (libnexus_fast.so).
#
# The shared object is a build artifact: never commit it.

CC := gcc
CFLAGS ?= -O3 -Wall -Wextra -Werror -std=c99
LIB := libnexus_fast.so

.PHONY: all clean

all: $(LIB)

$(LIB): nexus_fast.c
	$(CC) $(CFLAGS) -shared -fPIC -o $(LIB) nexus_fast.c

clean:
	rm -f *.so *.o
