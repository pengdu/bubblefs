
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

PROJECT_DIR=../..
SRC_PATH=$(PROJECT_DIR)/src

# dependencies
BOOST_PATH=$(PROJECT_DIR)/third_party/boost
JEMALLOC_PATH=$(PROJECT_DIR)/third_party
RAPIDJSON_PATH=$(PROJECT_DIR)/third_party
CPP_BTREE_PATH=$(PROJECT_DIR)/third_party
GFLAGS_PATH=$(PROJECT_DIR)/third_party
GTEST_PATH=$(PROJECT_DIR)/third_party
GLOG_PATH=$(PROJECT_DIR)/third_party
SNAPPY_PATH=$(PROJECT_DIR)/third_party
LEVELDB_PATH=$(PROJECT_DIR)/third_party
PROTOBUF_PATH=$(PROJECT_DIR)/third_party
PROTOC_PATH=$(PROJECT_DIR)/third_party/bin
PROTOC=$(PROTOC_PATH)/protoc
SOFA_PBRPC_PATH=$(PROJECT_DIR)/third_party
GPERFTOOLS_PATH=$(PROJECT_DIR)/thirdparty

INCLUDE_PATH = -I$(PROJECT_DIR)/src \
               -I$(BOOST_PATH) \
               -I$(RAPIDJSON_PATH)/include \
               -isystem -I$(GTEST_PATH)/include \
               -I$(SNAPPY_PATH)/include \
               -I$(LEVELDB_PATH)/include \
               -I$(PROTOBUF_PATH)/include \
               -I$(SOFA_PBRPC_PATH)/include \
               -I$(GPERFTOOLS_PATH)/include

LDFLAGS = -L$(GTEST_PATH)/lib -lgtest \
          -L$(SNAPPY_PATH)/lib -lsnappy \
          -L$(LEVELDB_PATH)/lib -lleveldb \
          -L$(PROTOBUF_PATH)/lib -lprotobuf \
          -L$(SOFA_PBRPC_PATH)/lib -lsofa-pbrpc \
          -L$(GPERFTOOLS_PATH)/lib -ltcmalloc_minimal \
          -lglog -lgflags -lpthread -lstdc++ -ldl -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

CXXFLAGS = -pthread -std=c++11 -fmax-errors=3 -Wall -fPIC $(OPT)
CXXFLAGS += -D_FILE_OFFSET_BITS=64 -D_REENTRANT -D_THREAD_SAFE

PLATFORM_SRC = \
	$(PROJECT_DIR)/src/platform/mutex.cc \
	$(PROJECT_DIR)/src/platform/logging_simple.cc

PLATFORM_OBJ = $(addsuffix .o, $(basename $(PLATFORM_SRC)))

UTILS_SRC = \
	$(PROJECT_DIR)/src/utils/thread_simple.cc
	
UTILS_OBJ = $(addsuffix .o, $(basename $(UTILS_SRC)))

OBJS = $(UTILS_OBJ) $(PLATFORM_OBJ)

LIBS =

BIN = bdcommon_test

all: $(BIN)
	@echo 'Done'
	
bdcommon_test: bdcommon_test.o $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@
	
.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(OBJS)
	rm -rf $(LIBS)
	rm -rf *.o
