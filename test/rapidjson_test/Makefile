
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

# dependencies
PROJECT_DIR=../..
RAPIDJSON_PATH=$(PROJECT_DIR)/third_party

INCLUDE_PATH = -I$(RAPIDJSON_PATH)/include
               

LDFLAGS = -lpthread -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

CXXFLAGS = -pthread -std=c++11 -Wall -fPIC $(OPT)

SIMPLEDOM_SRC = $(wildcard *.cc)
SIMPLEDOM_OBJ = $(patsubst %.cc, %.o, $(SIMPLEDOM_SRC))
SIMPLEDOM_HEADER = $(wildcard *.h)

OBJS = $(SIMPLEDOM_OBJ)

BIN = simpledom

all: $(BIN)
	@echo 'Done'
	
simpledom: $(SIMPLEDOM_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@
	
.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(OBJS)
	rm -rf $(LIBS)
