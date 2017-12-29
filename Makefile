
# OPT ?= -O2 -DNDEBUG # (A) Production use (optimized mode)
OPT ?= -g2 -Werror # (B) Debug mode, w/ full line-level debugging symbols
# OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols
# OPT ?= -O2 -pipe -Wall -W --fstrict-aliasing -Wno-invalid-offsetof -Wno-unused-parameter -fno-omit-frame-pointer

#CXX=/opt/compiler/gcc-4.8.2/bin/g++

PROJECT_DIR=.
THIRD_PARTY_DIR=$(PROJECT_DIR)/third_party

# Dependencies
include $(PROJECT_DIR)/depends.mk

INCLUDE_PATH = -I$(PROJECT_DIR)/src \
               -I$(THIRD_PARTY_DIR)/include \
               -I$(BOOST_PATH) \
               -isystem $(GTEST_PATH)/include

LDFLAGS = -L$(TBB_PATH)/lib -ltbb -Wl,-rpath=$(TBB_PATH)/lib \
          -L$(GTEST_PATH)/lib -lgtest \
          -L$(SNAPPY_PATH)/lib -lsnappy \
          -L$(LEVELDB_PATH)/lib -lleveldb \
          -L$(PROTOBUF_PATH)/lib -lprotobuf \
          -L$(SOFA_PBRPC_PATH)/lib -lsofa-pbrpc \
          -L$(GPERFTOOLS_PATH)/lib -ltcmalloc_minimal \
          -lgflags -lpthread -lstdc++ -ldl -lz -lrt

SO_LDFLAGS += -rdynamic $(DEPS_LDPATH) $(SO_DEPS_LDFLAGS) -lpthread -lrt -lz -ldl \
              -shared -Wl,--version-script,so-version-script # hide symbol of third_party libs

# Compiler
#CXX = g++
CXX = clang

# Compiler opts
GCC_OPTS = -fmax-errors=2
CLANG_OPTS = -ferror-limit=2

# Notes on the flags:
# 1. Added -fno-omit-frame-pointer: perf/tcmalloc-profiler use frame pointers by default
# 2. Added -D__const__= : Avoid over-optimizations of TLS variables by GCC>=4.8, like -D__const__= -D_GNU_SOURCE
# 3. Removed -Werror: Not block compilation for non-vital warnings, especially when the
#    code is tested on newer systems. If the code is used in production, add -Werror back
DFLAGS = -D_FILE_OFFSET_BITS=64 -D_REENTRANT -D_THREAD_SAFE
CXXFLAGS = -Wall -fPIC -std=c++11 -pthread $(DFLAGS) $(OPT)
CFLAGS = -Wall -W -fPIC $(DFLAGS) $(OPT)

ifeq ($(CXX), g++)
CXXFLAGS += $(GCC_OPTS)
else ifeq ($(CXX), clang)
CXXFLAGS += $(CLANG_OPTS)
endif

# Files

SRCEXTS = .c .cc .cpp .proto

ALL_DIRS = $(PROJECT_DIR)/src/*
ALL_SRCS = $(foreach d, $(ALL_DIRS), $(wildcard $(addprefix $(d)/*, $(SRCEXTS))))
ALL_OBJS = $(addsuffix .o, $(basename $(ALL_SRCS)))

PLATFORM_UTILS_DIRS = $(PROJECT_DIR)/src/platform $(PROJECT_DIR)/src/utils
PLATFORM_UTILS_SRCS = $(foreach d, $(PLATFORM_UTILS_DIRS), $(wildcard $(addprefix $(d)/*, $(SRCEXTS))))
PLATFORM_UTILS_OBJS = $(addsuffix .o, $(basename $(PLATFORM_UTILS_SRCS)))

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
    $(PROJECT_DIR)/src/utils/status.cc \
    $(PROJECT_DIR)/src/utils/string_format.cc \
    $(PROJECT_DIR)/src/utils/stringpiece.cc
UTILS_OBJS = $(addsuffix .o, $(basename $(UTILS_SRCS))) 

OBJS = $(PLATFORM_OBJS) $(UTILS_OBJS) $(PROTO_OBJS) $(RPC_OBJS)

LIBS =
 
BINS = $(PLATFORM_UTILS_OBJS)

# Commands

.PHONY:all
all: $(BINS)
	@echo "# Done"

.PHONY:clean
clean:
	@echo "# Clean"
	rm -rf $(ALL_OBJS)
	rm -rf $(BINS)
	rm -rf $(PROTO_SRCS) $(PROTO_HDRS)
	rm -rf *.o

#.SECONDARY: $(PROTO_SRCS)
.PRECIOUS: $(PROTO_SRCS)

# Make

# Depends
$(PROTO_OBJS): $(PROTO_HDRS)

# Tests	
.PHONY:test
TEST_BINS = ambry_bit_util_test
test: $(TEST_BINS)
	@echo "# Test"

.PHONY:test_clean
test_clean:
	@echo "# Test Clean"
	rm -rf $(TEST_BINS)

dmlc_registry_test: $(PROJECT_DIR)/src/utils/dmlc_registry_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

caffe2_registry_test: $(PROJECT_DIR)/src/utils/caffe2_registry_test.o $(PROJECT_DIR)/src/utils/caffe2_typeid.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ambry_bit_util_test: $(PROJECT_DIR)/src/utils/ambry_bit_util.o $(PROJECT_DIR)/src/utils/ambry_bit_util_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compile & Link
%.pb.cc %.pb.h: %.proto
	@echo "# Protoc gen $@"
	$(PROTOC) --proto_path=$(PROJECT_DIR)/src/proto --proto_path=/usr/local/include --cpp_out=$(PROJECT_DIR)/src/proto/ $<

%.o: %.cc
	@echo "# Compiling cc $@"
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c $< -o $@

%.o:%.cpp
	@echo "# Compiling cpp $@"
	@$(CXX) -c $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

%.o:%.c
	@echo "# Compiling c $@"
	@$(CC) -c $(CFLAGS) $(INCLUDE_PATH) $< -o $@
