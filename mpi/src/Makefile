CXX      := mpicxx
OPTFLAGS := -O2 -mcpu=a64fx+sve -msve-vector-bits=512 -ffast-math -fopt-info-vec
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