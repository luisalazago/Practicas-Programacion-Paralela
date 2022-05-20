#
# Makefile for make
#
# Compiler: gcc
#

CC = gcc

CFLAGS=-D_REENTRANT -Wall
LDFLAGS=-lpthread

PROGS = thread0 thread1 thread2 thread3 thread4 thread5

all: $(PROGS)

thread0: thread0.c
	gcc $(CFLAGS) thread0.c -o thread0 $(LDFLAGS)

thread1: thread1.c
	gcc $(CFLAGS) thread1.c -o thread1 $(LDFLAGS)

thread2: thread2.c
	gcc $(CFLAGS) thread2.c -o thread2 $(LDFLAGS)

thread3: thread3.c
	gcc $(CFLAGS) thread3.c -o thread3 $(LDFLAGS)

thread4: thread4.c
	gcc $(CFLAGS) thread4.c -o thread4 $(LDFLAGS)

thread5: thread5.c
	gcc $(CFLAGS) thread5.c -o thread5 $(LDFLAGS)

clean:
	rm -f $(PROGS) *~ *.o
