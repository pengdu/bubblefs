
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

# dependencies
PROTOC=./third_party/bin/protoc
PROTOBUF_PATH=./third_party
LEVELDB_PATH=./third_party
GRPC_PATH=./third_party
GFLAGS_PATH=./third_party
GTEST_PATH=./third_party

INCLUDE_PATH = -I./src -I$(PROTOBUF_PATH)/include \
               -I$(LEVELDB_PATH)/include \
               -I$(GRPC_PATH)/include \
               -I$(GFLAGS_PATH)/include
               

LDFLAGS = -pthread -L$(PROTOBUF_PATH)/lib -lprotobuf \
          -L$(LEVELDB_PATH)/lib -lleveldb \
          -L$(GFLAGS_PATH)/lib -lgflags \
          -L$(GTEST_PATH)/lib -lgtest \
          -L$(PBRPC_PATH)/lib -lgrpc++ -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed -ldl \
          -lpthread -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

CXXFLAGS = -pthread -std=c++11 -Wall -fPIC $(OPT)
#FUSEFLAGS = -D_FILE_OFFSET_BITS=64 -DFUSE_USE_VERSION=26 -I$(FUSE_PATH)/include
#FUSE_LL_FLAGS = -D_FILE_OFFSET_BITS=64 -DFUSE_USE_VERSION=26 -I$(FUSE_LL_PATH)/include

PROTO_FILE = $(wildcard src/proto/*.proto)
PROTO_SRC = $(patsubst %.proto,%.pb.cc,$(PROTO_FILE))
PROTO_HEADER = $(patsubst %.proto,%.pb.h,$(PROTO_FILE))
PROTO_OBJ = $(patsubst %.proto,%.pb.o,$(PROTO_FILE))

UTILS_OBJ = $(patsubst %.cc, %.o, $(wildcard src/utils/*.cc))

VERSION_OBJ = src/version.o
OBJS = $(PROTO_OBJ) $(VERSION_OBJ)

BIN = 


all: $(BIN)
	@echo 'Done'
	
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@
	
%.pb.h %.pb.cc: %.proto
	$(PROTOC) --proto_path=./src/proto/ --proto_path=$(PROTOBUF_PATH)/include --cpp_out=./src/proto/ $<

.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(OBJS) $(UTILS_OJB)
	rm -rf $(PROTO_SRC) $(PROTO_HEADER)
	rm -rf $(LIBS)
