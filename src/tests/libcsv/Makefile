###### macro defines

CFLAGS = -g -O0 -D_REENTRANT -I. -I../../h -I../../lib
CC = gcc

LIBS = -lpthread -lm

###### targets
default: csvinfo

csvinfo: csvinfo.o ../../lib/libcsv/libcsv.o
	$(CC) -o csvinfo csvinfo.o ../../lib/libcsv/libcsv.o $(LIBS)

clean:
	rm -f csvinfo

###### objects

csvinfo.o: csvinfo.c 
	$(CC) $(CFLAGS) -c csvinfo.c 






