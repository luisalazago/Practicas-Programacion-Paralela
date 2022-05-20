#
# Departamento de Electronica y Ciencias de la Computacion
# Facultad de Ingenieria y Ciencias
# Pontificia Universidad Javeriana - CALI
#
# C Examples
# Sample Makefile
#

#
# Set the compilers and associated flags
#

CC		= gcc
OPTFLAGS	=
DEBUGFLAGS	=
CFLAGS		= -Wall $(OPTFLAGS) $(DEBUGFLAGS)
LDFLAGS		= $(OPTFLAGS) $(DEBUGFLAGS)

PROGRAMS	= MatMatMult9

#
# Targets, in order of appearance
#

all: $(PROGRAMS)

MatMatMult9: MatMatMult9.c
	$(CC) -pg -std=gnu99 $(CFLAGS) MatMatMult9.c -o MatMatMult9

#
# Other nice thingys
#

clean:
	rm -f *% *~ core *.o $(PROGRAMS)
