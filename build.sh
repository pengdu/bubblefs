#!/usr/bin/env bash
set -x -e
# Every script you write should include set -e at the top.
# This tells bash that it should exit the script 
# if any statement returns a non-true return value.
# The benefit of using -e is that it prevents errors snowballing into serious issues
# when they could have been caught earlier. Again, for readability you may want to use set -o errexit.

########################################
# download & build depend software
########################################

WORK_DIR=`pwd`
DEPS_PACKAGE=`pwd`/third_pkg
DEPS_SOURCE=`pwd`/third_src
DEPS_PREFIX=`pwd`/third_party
DEPS_BUILD=`pwd`/build
FLAG_DIR=`pwd`/flag_build
DEPS_CONFIG="--prefix=${DEPS_PREFIX} --disable-shared --with-pic"

# export PATH=${DEPS_PREFIX}/bin:$PATH
mkdir -p ${DEPS_SOURCE} ${DEPS_PREFIX} ${DEPS_BUILD} ${FLAG_DIR}

mkdir -p ${DEPS_PREFIX}/bin ${DEPS_PREFIX}/lib ${DEPS_PREFIX}/include

if [ ! -f "${FLAG_DIR}/dl_third" ]; then
    touch "${FLAG_DIR}/dl_third"
fi

cd ${DEPS_SOURCE}

# boost
if [ ! -f "${FLAG_DIR}/boost_1_65_0" ] \
    || [ ! -d "${DEPS_PREFIX}/boost" ]; then
    cd ${DEPS_PREFIX}
    if [ -d "${DEPS_PREFIX}/boost" ]; then
    	rm -rf ${DEPS_PREFIX}/boost
    fi
    tar zxvf ${DEPS_PACKAGE}/boost_1_65_0.tar.gz -C .
    mv boost_1_65_0 boost
    touch "${FLAG_DIR}/boost_1_65_0"
fi

# jemalloc
if [ ! -f "${FLAG_DIR}/jemalloc_5_0_1" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libjemalloc.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/jemalloc" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/jemalloc" ] \
    	|| [ -d "${DEPS_BUILD}/jemalloc" ]; then
    	rm -rf ${DEPS_SOURCE}/jemalloc
    	rm -rf ${DEPS_BUILD}/jemalloc
    fi
    unzip ${DEPS_PACKAGE}/jemalloc-5.0.1.zip -d .
    mv jemalloc-5.0.1 jemalloc
    cd jemalloc
    ./autogen.sh
    ./configure --prefix=${DEPS_BUILD}/jemalloc
    make -j4
    make install_bin install_include install_lib
    cd ${DEPS_BUILD}/jemalloc
    cp -a lib/libjemalloc.a ${DEPS_PREFIX}/lib
    cp -a include/jemalloc ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/jemalloc_5_0_1"
fi

# rapidjson
if [ ! -f "${FLAG_DIR}/rapidjson_1_1_0" ] \
    || [ ! -d "${DEPS_PREFIX}/include/rapidjson" ]; then
    cd ${DEPS_PREFIX}/include
    if [ -d "${DEPS_PREFIX}/include/rapidjson" ]; then
        rm -rf ${DEPS_PREFIX}/include/rapidjson
    fi
    unzip ${DEPS_PACKAGE}/rapidjson-1.1.0.zip -d .
    mv rapidjson-1.1.0 rapidjson
    touch "${FLAG_DIR}/rapidjson_1_1_0"
fi

# cpp-btree
if [ ! -f "${FLAG_DIR}/cpp-btree_1_0_1" ] \
    || [ ! -d "${DEPS_PREFIX}/include/cpp-btree" ]; then
    cd ${DEPS_PREFIX}/include
    if [ -d "${DEPS_PREFIX}/include/cpp-btree" ]; then
        rm -rf ${DEPS_PREFIX}/include/cpp-btree
    fi
    tar zxvf ${DEPS_PACKAGE}/cpp-btree-1.0.1.tar.gz -C .
    mv cpp-btree-1.0.1 cpp-btree
    touch "${FLAG_DIR}/cpp-btree_1_0_1"
fi

# gflags
if [ ! -f "${FLAG_DIR}/gflags_2_1_1" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/gflags" ] ; then
    	rm -rf ${DEPS_SOURCE}/gflags
    fi
    unzip ${DEPS_PACKAGE}/gflags-2.2.1.zip -d .
    mv gflags-2.2.1 gflags
    cd gflags
    cmake -DGFLAGS_NAMESPACE=google -DCMAKE_CXX_FLAGS=-fPIC
    make -j4
    make test
    sudo make install
    touch "${FLAG_DIR}/gflags_2_1_1"
fi

# googletest
if [ ! -f "${FLAG_DIR}/googletest_1_8_0" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libgtest.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/gtest" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/googletest" ]; then
    	rm -rf ${DEPS_SOURCE}/googletest
    fi
    unzip ${DEPS_PACKAGE}/googletest-release-1.8.0.zip -d .
    mv googletest-release-1.8.0 googletest
    cd googletest/googletest
    g++ -isystem include -I. -pthread -c src/gtest-all.cc
    ar -rv libgtest.a gtest-all.o
    cp -a libgtest.a ${DEPS_PREFIX}/lib
    cp -a include/gtest ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/googletest_1_8_0"
fi

# glog
if [ ! -f "${FLAG_DIR}/glog_0_3_5" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libglog.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/glog" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/glog" ] \
    	|| [ -d "${DEPS_BUILD}/glog" ]; then
    	rm -rf ${DEPS_SOURCE}/glog
    	rm -rf ${DEPS_BUILD}/glog
    fi
    unzip ${DEPS_PACKAGE}/glog-0.3.5.zip -d .
    mv glog-0.3.5 glog
    mkdir ${DEPS_BUILD}/glog
    cd ${DEPS_BUILD}/glog
    export CXXFLAGS="-fPIC" && cmake ${DEPS_SOURCE}/glog && make VERBOSE=1
    make
    sudo make install
    cp -a libglog.a ${DEPS_PREFIX}/lib
    cp -a glog ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/glog_0_3_5"
fi

# snappy
# use cmake 3.x or above
if [ ! -f "${FLAG_DIR}/snappy_1_1_7" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libsnappy.a" ] \
    || [ ! -f "${DEPS_PREFIX}/include/snappy.h" ] \
    || [ ! -f "${DEPS_PREFIX}/include/snappy-stubs-public.h" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/snappy" ] \
    	|| [ -d "${DEPS_BUILD}/snappy" ]; then
    	rm -rf ${DEPS_SOURCE}/snappy
    	rm -rf ${DEPS_BUILD}/snappy
    fi
    unzip ${DEPS_PACKAGE}/snappy-1.1.7.zip -d .
    mv snappy-1.1.7 snappy
    mkdir ${DEPS_BUILD}/snappy
    cd ${DEPS_BUILD}/snappy
    cmake ${DEPS_SOURCE}/snappy
    make -j4
    cp -a libsnappy.a ${DEPS_PREFIX}/lib
    cp -a ${DEPS_SOURCE}/snappy/snappy.h ${DEPS_PREFIX}/include
    cp -a snappy-stubs-public.h ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/snappy_1_1_7"
fi

# leveldb
if [ ! -f "${FLAG_DIR}/leveldb_1_2_0" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libleveldb.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/leveldb" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/leveldb" ]; then
    	rm -rf ${DEPS_SOURCE}/leveldb
    fi
    unzip ${DEPS_PACKAGE}/leveldb-1.20.zip -d .
    mv leveldb-1.20 leveldb
    cd leveldb
    make -j4
    cp -a out-static/libleveldb.a ${DEPS_PREFIX}/lib
    cp -a include/leveldb ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/leveldb_1_2_0"
fi

# protobuf
if [ ! -f "${FLAG_DIR}/protobuf_3_3_2" ] \
    || [ ! -f "${DEPS_PREFIX}/bin/protoc" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libprotobuf.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/google/protobuf" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/protobuf" ] \
    	|| [ -d "${DEPS_BUILD}/protobuf" ]; then
    	rm -rf ${DEPS_SOURCE}/protobuf
    	rm -rf ${DEPS_BUILD}/protobuf
    fi
    unzip ${DEPS_PACKAGE}/protobuf-3.3.2.zip -d .
    mv protobuf-3.3.2 protobuf
    cd protobuf
    mkdir ${DEPS_BUILD}/protobuf
    ./autogen.sh
    ./configure --prefix=${DEPS_BUILD}/protobuf
    make -j4
    make check
    make install
    cd ${DEPS_BUILD}/protobuf
    cp -a bin/protoc ${DEPS_PREFIX}/bin
    cp -a lib/libprotobuf.a ${DEPS_PREFIX}/lib
    cp -a include/google ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/protobuf_3_3_2"
fi

# sofa-pbrpc
if [ ! -f "${FLAG_DIR}/sofa-pbrpc_1_1_3" ] \
	|| [ ! -f "${DEPS_PREFIX}/bin/sofa-pbrpc-client" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libsofa-pbrpc.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/sofa/pbrpc" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/sofa-pbrpc" ] \
    	|| [ -d "${DEPS_BUILD}/sofa-pbrpc" ]; then
    	rm -rf ${DEPS_SOURCE}/sofa-pbrpc
    	rm -rf ${DEPS_BUILD}/sofa-pbrpc
    fi
    unzip ${DEPS_PACKAGE}/sofa-pbrpc-1.1.3.zip -d .
    mv sofa-pbrpc-1.1.3 sofa-pbrpc
    cd sofa-pbrpc
    mkdir ${DEPS_BUILD}/sofa-pbrpc
    sed -i '/BOOST_HEADER_DIR=/ d' depends.mk
    sed -i '/PROTOBUF_DIR=/ d' depends.mk
    sed -i '/SNAPPY_DIR=/ d' depends.mk
    echo "BOOST_HEADER_DIR=${DEPS_PREFIX}/boost" >> depends.mk
    echo "PROTOBUF_DIR=${DEPS_PREFIX}" >> depends.mk
    echo "SNAPPY_DIR=${DEPS_PREFIX}" >> depends.mk
    echo "PREFIX=${DEPS_BUILD}/sofa-pbrpc" >> depends.mk
    make -j4
    make install
    cd ${DEPS_BUILD}/sofa-pbrpc
    cp -a bin/sofa-pbrpc-client ${DEPS_PREFIX}/bin
    cp -a lib/libsofa-pbrpc.a ${DEPS_PREFIX}/lib
    cp -a include/sofa ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/sofa-pbrpc_1_1_3"
fi

cd ${WORK_DIR}

########################################
# build
########################################

#make clean
#make -j4

