LIBS=-lafcpu
LIB_PATHS=-L/usr/lib
INCLUDES=-I/usr/include
CXX=clang++ $(COMPILER_OPTIONS)
COMPILER_OPTIONS=-std=c++11
TARGET = main

all: helloWorld.cpp Makefile
	$(CXX) helloWorld.cpp -o $(TARGET) $(INCLUDES) $(LIBS) $(LIB_PATHS)

clean:
	rm -rf $(TARGET)* *.o *~
