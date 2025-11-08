CC=gcc
#CFLAGS=-g
CFLAGS=

nnp:loader.o nnp.o main.o
	$(CC) $(CFLAGS) nnp.o loader.o main.o -o nnp -lm
loader.o: loader.h config.h
	$(CC) -c loader.c $(CFLAGS) -o loader.o

nnp.o: nnp.c loader.h config.h nnp.h
	$(CC) -c nnp.c $(CFLAGS) -o nnp.o
main.o: main.c config.h nnp.h

clean: prod
	rm -f nnp 
prod: 
	rm -f *.o model.bin
