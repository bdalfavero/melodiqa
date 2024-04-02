cc=g++
cflags=-I/Users/benjamindalfavero/include/eigen-3.4.0/

sources=$(wildcard src/*.cpp)
objects=$(patsubst %.cpp,%.o,$(sources))

all: mq

mq: $(objects)
	$(cc) $^ -o $@

src/%.o: src/%.cpp
	$(cc) -c $< -o $@ $(cflags)