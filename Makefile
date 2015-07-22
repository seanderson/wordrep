

CC = gcc
OPTFLAGS = -O2
LIBS = -lm -lpthread

all: word2vec

word2vec: word2vec.c
	$(CC) $(OPTFLAGS) -o word2vec word2vec.c $(LIBS)

clean:
	rm -rf *.o word2vec

