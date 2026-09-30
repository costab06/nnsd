###### macro defines

CFLAGS = -D_REENTRANT -I. -I../h -I../dce -I/opt/dce/include

###### targets
default: mkmtxfile vermtxfile

mkmtxfile: mkmtxfile.o 
	gcc -g -o mkmtxfile mkmtxfile.o

vermtxfile: vermtxfile.o 
	gcc -g -o vermtxfile vermtxfile.o


clean:
	rm -f mkmtxfile vermtxfile mkmtxfile.o vermtxfile.o


###### objects

mkmtxfile.o: mkmtxfile.c
	gcc -g -c $(CFLAGS) mkmtxfile.c

vermtxfile.o: vermtxfile.c
	gcc -g -c $(CFLAGS) vermtxfile.c







