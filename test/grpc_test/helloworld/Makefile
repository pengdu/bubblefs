#
# Copyright 2015, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)

PROJ_DIR = $(shell pwd)/../../..
SRC_DIR = $(PROJ_DIR)/src
CURR_DIR = $(shell pwd)
DEPS_PREFIX=${PROJ_DIR}/third_party

CXX = g++
CPPFLAGS += -pthread -I${DEPS_PREFIX}/include
CXXFLAGS += -std=c++11 -fmax-errors=1
LDFLAGS += -pthread -L${DEPS_PREFIX}/lib -lprotobuf -lpthread -lgrpc++ -lgrpc \
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed \
           -ldl -lz -lrt
PROTOC = ${DEPS_PREFIX}/bin/protoc
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= ${DEPS_PREFIX}/bin/grpc_cpp_plugin

PROTOS_PATH = ../proto

vpath %.proto $(PROTOS_PATH)

all: greeter_client greeter_server greeter_async_client greeter_async_client2 greeter_async_server

greeter_client: helloworld.pb.o helloworld.grpc.pb.o greeter_client.o
	$(CXX) $^ $(LDFLAGS) -o $@

greeter_server: helloworld.pb.o helloworld.grpc.pb.o greeter_server.o
	$(CXX) $^ $(LDFLAGS) -o $@

greeter_async_client: helloworld.pb.o helloworld.grpc.pb.o greeter_async_client.o
	$(CXX) $^ $(LDFLAGS) -o $@

greeter_async_client2: helloworld.pb.o helloworld.grpc.pb.o greeter_async_client2.o
	$(CXX) $^ $(LDFLAGS) -o $@

greeter_async_server: helloworld.pb.o helloworld.grpc.pb.o greeter_async_server.o
	$(CXX) $^ $(LDFLAGS) -o $@

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

.PRECIOUS: %.pb.cc
%.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=. $<

clean:
	rm -f *.o *.pb.cc *.pb.h greeter_client greeter_server greeter_async_client greeter_async_client2 greeter_async_server
