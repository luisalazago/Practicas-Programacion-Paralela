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

PROGRAMS 	= checkDeviceInfo

#
# Targets, in order of appearance
#

all: $(PROGRAMS)

checkDeviceInfo: checkDeviceInfo.cu
	$(CC) $(CFLAGS) checkDeviceInfo.cu -o checkDeviceInfo

#
# Other nice thingys
#

clean:
	rm -f *% *~ core *.o $(PROGRAMS)