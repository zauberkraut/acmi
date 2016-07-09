name = acmi
root = $(shell pwd -P)
src = $(root)/src
build = $(root)/build
bin = $(build)/$(name)

cc = gcc

oflags = -Os
cflags = -std=c11 $(oflags) -I/usr/local/cuda/include -Wall -Werror \
         -D_POSIX_C_SOURCE=200809L
libs = -lopenblas -lpthread -L/usr/local/cuda/lib64 -lcudart -lcublas -lm

$(bin): $(build)/main.o $(build)/util.o $(build)/mat.o $(build)/la.o \
        $(build)/blas.o $(build)/cuda.o $(build)/mmio.o
	$(cc) -o $(@) $(^) $(libs)

$(build)/%.o: $(src)/%.c
	@mkdir -p $(build)
	$(cc) -o $(@) -c $(<) $(cflags)

debug: oflags = -O0 -g3
debug: $(out)

.PHONY: clean
clean:
	rm -fr $(build) $(out)
