
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

# dependencies
include depends.mk

PROJECT_DIR=.

INCLUDE_PATH = -I$(PROJECT_DIR)/src \
               -I$(BOOST_PATH) \
               -I$(RAPIDJSON_PATH)/include \
               -I$(GTEST_PATH)/include \
               -I$(GLOG_PATH)/include \
               -I$(SNAPPY_PATH)/include \
               -I$(LEVELDB_PATH)/include \
               -I$(PROTOBUF_PATH)/include \
               -I$(SOFA_PBRPC_PATH)/include \
               -I$(GPERFTOOLS_PATH)/include

LDFLAGS = -L$(GTEST_PATH)/lib -lgtest \
          -L$(GLOG_PATH)/lib -lglog \
          -L$(SNAPPY_PATH)/lib -lsnappy \
          -L$(LEVELDB_PATH)/lib -lleveldb \
          -L$(PROTOBUF_PATH)/lib -lprotobuf \
          -L$(SOFA_PBRPC_PATH)/lib -lsofa-pbrpc \
          -L$(GPERFTOOLS_PATH)/lib -ltcmalloc_minimal \
          -lgflags -lpthread -ldl -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

CXXFLAGS = -pthread -std=c++11 -fmax-errors=3 -Wall -fPIC $(OPT)

PLATFORM_SRC = \

PLATFORM_OBJ = $(patsubst %.cc, %.o, $(PLATFORM_SRC))

UTILS_SRC = \
	
UTILS_OBJ = $(patsubst %.cc, %.o, $(UTILS_SRC))

OBJS = 

BIN = bubblefs_test

all: $(OBJS)
	@echo 'Done'
	
bubblefs_test: $(OBJS)
#	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@
	
.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(OBJS)
	rm -rf $(LIBS)
