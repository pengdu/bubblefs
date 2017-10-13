
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols
# OPT ?= -O2 -pipe -Wall -W --fstrict-aliasing -Wno-invalid-offsetof -Wno-unused-parameter -fno-omit-frame-pointer

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

PROJECT_DIR=.

# dependencies
include $(PROJECT_DIR)/depends.mk

INCLUDE_PATH = -I$(PROJECT_DIR)/src \
               -I$(TBB_PATH)/include \
               -I$(BOOST_PATH) \
               -I$(LIBCO_PATH)/include \
               -I$(RAPIDJSON_PATH)/include \
               -isystem $(GTEST_PATH)/include \
               -I$(SNAPPY_PATH)/include \
               -I$(LEVELDB_PATH)/include \
               -I$(PROTOBUF_PATH)/include \
               -I$(SOFA_PBRPC_PATH)/include \
               -I$(GPERFTOOLS_PATH)/include

LDFLAGS = -L$(TBB_PATH)/lib -ltbb -Wl,-rpath=$(TBB_PATH)/lib \
          -L$(LIBCO_PATH)/lib -lcolib \
          -L$(GTEST_PATH)/lib -lgtest \
          -L$(SNAPPY_PATH)/lib -lsnappy \
          -L$(LEVELDB_PATH)/lib -lleveldb \
          -L$(PROTOBUF_PATH)/lib -lprotobuf \
          -L$(SOFA_PBRPC_PATH)/lib -lsofa-pbrpc \
          -L$(GPERFTOOLS_PATH)/lib -ltcmalloc_minimal \
          -lglog -lgflags -lpthread -lstdc++ -ldl -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

# Notes on the flags:
# 1. Added -fno-omit-frame-pointer: perf/tcmalloc-profiler use frame pointers by default
# 2. Added -D__const__= : Avoid over-optimizations of TLS variables by GCC>=4.8, like -D__const__= -D_GNU_SOURCE
# 3. Removed -Werror: Not block compilation for non-vital warnings, especially when the
#    code is tested on newer systems. If the code is used in production, add -Werror back
DFLAGS = -D_FILE_OFFSET_BITS=64 -D_REENTRANT -D_THREAD_SAFE
CXXFLAGS = -pthread -std=c++11 -fmax-errors=2 -Wall -fPIC $(DFLAGS) $(OPT)
CFLAGS = -Wall -W -fPIC $(DFLAGS) $(OPT)

# Files

SRCEXTS = .c .cc .cpp .proto

ALL_DIRS = $(PROJECT_DIR)/src/platform $(PROJECT_DIR)/src/utils
#ALL_DIRS = $(PROJECT_DIR)/src/*
ALL_SRCS = $(foreach d, $(ALL_DIRS), $(wildcard $(addprefix $(d)/*, $(SRCEXTS))))
ALL_OBJS = $(addsuffix .o, $(basename $(ALL_SRCS)))

CLIENT_SRCS = $(wildcard $(PROJECT_DIR)/src/client/*.cc)
CLIENT_OBJS = $(addsuffix .o, $(basename $(CLIENT_SRCS)))

DB_SRCS = $(wildcard $(PROJECT_DIR)/src/db/*.cc)
DB_OBJS = $(addsuffix .o, $(basename $(DB_SRCS)))

PLATFORM_SRCS = \
		$(PROJECT_DIR)/src/platform/bdcommon_logging.cc \
        $(PROJECT_DIR)/src/platform/mutex.cc
PLATFORM_OBJS = $(addsuffix .o, $(basename $(PLATFORM_SRCS))) 

PROTO_FILES = $(wildcard src/proto/*.proto)
PROTO_SRCS = $(patsubst %.proto,%.pb.cc, $(PROTO_FILES))
PROTO_HDRS = $(patsubst %.proto,%.pb.h, $(PROTO_FILES))
PROTO_OBJS = $(patsubst %.proto,%.pb.o, $(PROTO_FILES))

RPC_SRCS = $(wildcard $(PROJECT_DIR)/src/rpc/*.cc)
RPC_OBJS = $(addsuffix .o, $(basename $(RPC_SRCS)))

UTILS_SRCS = \
        $(PROJECT_DIR)/src/utils/bdcommon_str_util.cc \
		$(PROJECT_DIR)/src/utils/bdcommon_thread.cc \
        $(PROJECT_DIR)/src/utils/hash.cc \
        $(PROJECT_DIR)/src/utils/string_format.cc \
        $(PROJECT_DIR)/src/utils/stringpiece.cc
UTILS_OBJS = $(addsuffix .o, $(basename $(UTILS_SRCS)))

OBJS = $(PLATFORM_OBJS) $(UTILS_OBJS) $(PROTO_OBJS) $(RPC_OBJS) $(DB_OBJS) $(CLIENT_OBJS)

LIBS =
 
BIN = $(ALL_OBJS)

.PHONY:all
all: $(BIN)
	@echo "# Done"

.PHONY:clean
clean:
	@echo "# Clean"
	rm -rf $(ALL_OBJS)
	rm -rf $(BIN)
	rm -rf $(PROTO_SRCS) $(PROTO_HDRS)
	rm -rf *.o

#.SECONDARY: $(PROTO_SRCS)
.PRECIOUS: $(PROTO_SRCS)

# Depends
$(PROTO_OBJS): $(PROTO_HDRS)

# Targets

# Tests	
dmlc_registry_test: $(PROJECT_DIR)/src/utils/dmlc_registry_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

caffe2_registry_test: $(PROJECT_DIR)/src/utils/caffe2_registry_test.o $(PROJECT_DIR)/src/utils/caffe2_typeid.o
	$(CXX) $^ -o $@ $(LDFLAGS)

%.pb.cc %.pb.h: %.proto
	@echo "# Protoc gen $@"
	$(PROTOC) --proto_path=$(PROJECT_DIR)/src/proto --proto_path=/usr/local/include --cpp_out=$(PROJECT_DIR)/src/proto/ $<

%.o: %.cc
	@echo "# Compiling cc $@"
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@

%.o:%.cpp
	@echo "# Compiling cpp $@"
	@$(CXX) -c $(HDRPATHS) $(CXXFLAGS) $< -o $@

%.o:%.c
	@echo "# Compiling c $@"
	@$(CC) -c $(HDRPATHS) $(CFLAGS) $< -o $@
