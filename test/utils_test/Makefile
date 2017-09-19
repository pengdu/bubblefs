
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

# dependencies
PROJECT_DIR=../..
SRC_PATH=$(PROJECT_DIR)/src

GTEST_PATH=$(PROJECT_DIR)/third_party
GLOG_PATH=$(PROJECT_DIR)/third_party

INCLUDE_PATH = -I$(SRC_PATH) \
               -I$(GTEST_PATH)/include \
               -I$(GLOG_PATH)/include

LDFLAGS = -L$(GTEST_PATH)/lib -lgtest \
          -L$(GLOG_PATH)/lib -lglog \
          -lgflags -lpthread -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl
              #-shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

CXXFLAGS = -pthread -std=c++11 -Wall -fPIC $(OPT)

ATOMICOPS_TEST_SRC = $(wildcard *.cc)
ATOMICOPS_TEST_OBJ = $(patsubst %.cc, %.o, $(ATOMICOPS_TEST_SRC))
ATOMICOPS_TEST_HEADER = $(wildcard *.h)

OBJS = $(ATOMICOPS_TEST_OBJ)

BIN = atomicops_test

all: $(BIN)
	@echo 'Done'
	
atomicops_test: $(ATOMICOPS_TEST_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(SO_LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@
	
.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(OBJS)
	rm -rf $(LIBS)
