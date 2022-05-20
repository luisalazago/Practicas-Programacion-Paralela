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

PROGRAMS 	= CUDA_vectorAddition_blocks CUDA_vectorAddition_threads CUDA_vectorAddition_blocks_threads

#
# Targets, in order of appearance
#

all : $(PROGRAMS)

CUDA_vectorAddition_blocks: vectorAddition_blocks.cu
	$(CC) $(CFLAGS) vectorAddition_blocks.cu -o CUDA_vectorAddition_blocks

CUDA_vectorAddition_threads: vectorAddition_threads.cu
	$(CC) $(CFLAGS) vectorAddition_threads.cu -o CUDA_vectorAddition_threads

CUDA_vectorAddition_blocks_threads: vectorAddition_blocks_threads.cu
	$(CC) $(CFLAGS) vectorAddition_blocks_threads.cu -o CUDA_vectorAddition_blocks_threads

#
# Other nice thingys
#

clean:
	rm -f *% *~ core *.o *.nvprof *.qdrep *.sqlite *.ncu-rep $(PROGRAMS)
