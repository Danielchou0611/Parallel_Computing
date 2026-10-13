# HW 2-1. This is the build the judge does, for your own machine and the login node: it compiles
# hw2-1.cpp (yours) with the base files, with exactly these flags. The judge does not read
# this Makefile; it compiles your hw2-1.cpp itself, against its own copies of the base files.
CXX      = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native

SRCS = main.cpp graph_gen.cpp hw2-1.cpp
OBJS = $(SRCS:.cpp=.o)

all: hw2-1

hw2-1: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp graph_gen.hpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) hw2-1 hw2-1-prof hw2-1-prof.o

hw2-1-prof: main.o graph_gen.o hw2-1-prof.o
	$(CXX) $(CXXFLAGS) -o $@ $^

hw2-1-prof.o: hw2-1.cpp graph_gen.hpp
	$(CXX) $(CXXFLAGS) -DENABLE_PROFILE=1 -c $< -o $@

.PHONY: all clean hw2-1-prof

