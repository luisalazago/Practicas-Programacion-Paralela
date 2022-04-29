#
# Departamento de Electronica y Ciencias de la Computacion
# Facultad de Ingenieria u Ciencias
# Pontificia Universidad Javeriana - CALI
#
# Sample Makefile
#

#
# Set the compilers and associated flags
#

CC		= gcc
VFLAGS		= -fstrict-aliasing -ftree-vectorize -march=native -mtune=native -fopt-info-vec-optimized -std=gnu99
# use the one below if running on a Xeon Phi x200 (or later) and Skylake-X (or later)
#VFLAGS		= -fstrict-aliasing -ftree-vectorize -march=native -mtune=native -fopt-info-vec-optimized -mprefer-vector-width=512 -std=gnu99

OPTFLAGS	= -O3
DEBUGFLAGS	=
CFLAGS		= -Wall $(VFLAGS) $(OPTFLAGS) $(DEBUGFLAGS)
LDFLAGS		= $(OPTFLAGS) $(DEBUGFLAGS)

PROGRAM 	= stream_triad

OBJS 		= stream_triad.o timer.o

#
# Targets, in order of appearance
#

all: $(PROGRAM) cpit1


$(PROGRAM): $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS) $(LIBS)

.c.o:
	$(CC) $< -c $(CFLAGS)

cpit1: cpit1.c timer.c
	mpicc -Wall -fopenmp -c cpit1.c; gcc -Wall -c timer.c; mpicc -fopenmp -o cpit1 cpit1.o timer.o

clean:
	-rm -f $(OBJS) $(PROGRAM) cpit1.o cpit1 *.s
