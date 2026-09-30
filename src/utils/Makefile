###### macro defines

CC     = gcc
CFLAGS = -g -O3 -D_REENTRANT -I. -I../h -I../lib/cJSON

OBJS = log_utils.o tcp_utils.o daemon.o config.o rest_server.o

###### targets

default: $(OBJS)

clean:
	rm -f $(OBJS)

###### objects

log_utils.o: log_utils.c ../h/log_utils.h ../h/packet.h
	$(CC) $(CFLAGS) -c $<

tcp_utils.o: tcp_utils.c ../h/packet.h ../h/common.h ../h/log_utils.h
	$(CC) $(CFLAGS) -c $<

daemon.o: daemon.c
	$(CC) $(CFLAGS) -c $<

config.o: config.c ../h/config.h ../h/common.h ../h/log_utils.h
	$(CC) $(CFLAGS) -c $<

rest_server.o: rest_server.c ../h/rest_server.h ../h/packet.h ../h/common.h ../h/log_utils.h
	$(CC) $(CFLAGS) -c $<











