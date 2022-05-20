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

PROGRAMS 	= hello0 hello1 hello2

#
# Targets, in order of appearance
#

all: $(PROGRAMS)

hello0: hello0.cu
	$(CC) $(CFLAGS) hello0.cu -o hello0

hello1: hello1.cu
	$(CC) $(CFLAGS) hello1.cu -o hello1

hello2: hello2.cu
	$(CC) $(CFLAGS) hello2.cu -o hello2

#
# Other nice thingys
#

clean:
	rm -f *% *~ core *.o $(PROGRAMS)
