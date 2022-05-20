#
# Departamento de Electronica y Ciencias de la Computacion
# Pontificia Universidad Javeriana - CALI
#
# CUDA Examples
# Sample Makefile
#

#
# Set the compilers and associated flags
#

CC		= nvcc
OPTFLAGS	=
DEBUGFLAGS	=
CFLAGS		= $(OPTFLAGS) $(DEBUGFLAGS)
LDFLAGS		= $(OPTFLAGS) $(DEBUGFLAGS)

PROGRAMS 	= MatMatAdd_threads1

#
# Targets, in order of appearance
#

all : $(PROGRAMS)

MatMatAdd_threads1: MatMatAdd_threads1.cu
	$(CC) $(CFLAGS) MatMatAdd_threads1.cu -o MatMatAdd_threads1

#
# Other nice thingys
#

clean:
	rm -f *% *~ core *.o $(PROGRAMS)
