CXX      := g++
OPTFLAGS := -O2 -march=native -mavx2 -mprefer-vector-width=256 -mfma -ffast-math -fopt-info-vec -g
CXXFLAGS := -std=c++17 -Wall -Wextra -fopenmp $(OPTFLAGS)

SRCS     := $(wildcard *.cpp)
TARGETS  := $(SRCS:.cpp=)

.PHONY: all clean $(TARGETS)

all: $(TARGETS)

$(TARGETS): %: %.cpp
	@echo "Building $@..."
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)