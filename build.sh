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
FLAG_DIR=`pwd`/flag_build
DEPS_CONFIG="--prefix=${DEPS_PREFIX} --disable-shared --with-pic"

# export PATH=${DEPS_PREFIX}/bin:$PATH
mkdir -p ${DEPS_SOURCE} ${DEPS_PREFIX} ${FLAG_DIR}

mkdir -p ${DEPS_PREFIX}/bin ${DEPS_PREFIX}/lib ${DEPS_PREFIX}/include

if [ ! -f "${FLAG_DIR}/dl_third" ]; then
    touch "${FLAG_DIR}/dl_third"
fi

cd ${DEPS_SOURCE}

# ldconfig for searching in /usr/lib
# echo "/usr/local/lib" >> /etc/ld.so.conf for searching in /usr/local/lib
# in .bashrc or /etc/profile or shell, export LD_LIBRARY_PATH=/PATH/TO/LIB:$LD_LIBRARY_PATH for searching lib*.so* in other dirs
# or use -rpath/-R for runtime shared-lib searching, prio > LD_LIBRARY_PATH

#Pre-installed
#sudo apt-get install autoconf automake libtool zlib1g zlib1g-dev

# boost
if [ ! -f "${FLAG_DIR}/boost_1_65_0" ] \
    || [ ! -d "${DEPS_PREFIX}/boost" ]; then
    cd ${DEPS_PREFIX}
    if [ -d "${DEPS_PREFIX}/boost" ]; then
    	rm -rf ${DEPS_PREFIX}/boost
    fi
    tar zxvf ${DEPS_PACKAGE}/boost_1_65_0.tar.gz -C .
    mv boost_1_65_0 boost
    #cd boost
    #./bootstrap.sh
    #./b2
    #sudo ./b2 install 
    touch "${FLAG_DIR}/boost_1_65_0"
fi

# rapidjson
if [ ! -f "${FLAG_DIR}/rapidjson_1_1_0" ] \
    || [ ! -d "${DEPS_PREFIX}/include/rapidjson" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/rapidjson" ]; then
        rm -rf ${DEPS_SOURCE}/rapidjson
    fi
    unzip ${DEPS_PACKAGE}/rapidjson-1.1.0.zip -d .
    mv rapidjson-1.1.0 rapidjson && cd rapidjson
    cp -a include/rapidjson ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/rapidjson_1_1_0"
fi

# jemalloc
if [ ! -f "${FLAG_DIR}/jemalloc_5_0_1" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libjemalloc.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/jemalloc" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/jemalloc" ]; then
    	rm -rf ${DEPS_SOURCE}/jemalloc
    fi
    unzip ${DEPS_PACKAGE}/jemalloc-5.0.1.zip -d .
    mv jemalloc-5.0.1 jemalloc && cd jemalloc
    mkdir build
    ./autogen.sh
    ./configure --prefix=${DEPS_SOURCE}/jemalloc/build
    make -j4
    make install_bin install_include install_lib
    cd build
    cp -a lib/libjemalloc.a ${DEPS_PREFIX}/lib
    cp -a include/jemalloc ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/jemalloc_5_0_1"
fi

# gflags
if [ ! -f "${FLAG_DIR}/gflags_2_0" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/gflags" ] ; then
    	rm -rf ${DEPS_SOURCE}/gflags
    fi
    unzip ${DEPS_PACKAGE}/gflags-2.0.zip -d .
    mv gflags-2.0 gflags && cd gflags
    ./configure
    make -j4
    make check
    sudo make install
    sudo ldconfig
    touch "${FLAG_DIR}/gflags_2_0"
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

# snappy
# use cmake 3.x or above
if [ ! -f "${FLAG_DIR}/snappy_1_1_7" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libsnappy.a" ] \
    || [ ! -f "${DEPS_PREFIX}/include/snappy.h" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/snappy" ]; then
    	rm -rf ${DEPS_SOURCE}/snappy
    fi
    unzip ${DEPS_PACKAGE}/snappy-1.1.7.zip -d .
    mv snappy-1.1.7 snappy && cd snappy
    mkdir build && cd build
    cmake ..
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
    mv leveldb-1.20 leveldb && cd leveldb
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
    if [ -d "${DEPS_SOURCE}/protobuf" ]; then
    	rm -rf ${DEPS_SOURCE}/protobuf
    fi
    unzip ${DEPS_PACKAGE}/protobuf-3.3.2.zip -d .
    mv protobuf-3.3.2 protobuf && cd protobuf
    mkdir build
    ./autogen.sh
    ./configure --prefix=${DEPS_SOURCE}/protobuf/build
    make -j4
    make check
    make install
    cd build
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
    if [ -d "${DEPS_SOURCE}/sofa-pbrpc" ]; then
    	rm -rf ${DEPS_SOURCE}/sofa-pbrpc
    fi
    unzip ${DEPS_PACKAGE}/sofa-pbrpc-1.1.3.zip -d .
    mv sofa-pbrpc-1.1.3 sofa-pbrpc && cd sofa-pbrpc
    mkdir build
    sed -i '/BOOST_HEADER_DIR=/ d' depends.mk
    sed -i '/PROTOBUF_DIR=/ d' depends.mk
    sed -i '/SNAPPY_DIR=/ d' depends.mk
    echo "BOOST_HEADER_DIR=${DEPS_PREFIX}/boost" >> depends.mk
    echo "PROTOBUF_DIR=${DEPS_PREFIX}" >> depends.mk
    echo "SNAPPY_DIR=${DEPS_PREFIX}" >> depends.mk
    echo "PREFIX=${DEPS_SOURCE}/sofa-pbrpc/build" >> depends.mk
    make -j4
    make install
    cd build
    cp -a bin/sofa-pbrpc-client ${DEPS_PREFIX}/bin
    cp -a lib/libsofa-pbrpc.a ${DEPS_PREFIX}/lib
    cp -a include/sofa ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/sofa-pbrpc_1_1_3"
fi

# libunwind for gperftools
if [ ! -f "${FLAG_DIR}/libunwind_1_0_0" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libunwind.a" ] \
    || [ ! -f "${DEPS_PREFIX}/include/libunwind.h" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/libunwind" ]; then
        rm -rf ${DEPS_SOURCE}/libunwind
    fi
    unzip ${DEPS_PACKAGE}/libunwind-vanilla_pathscale.zip -d .
    mv libunwind-vanilla_pathscale libunwind && cd libunwind
    mkdir build
    ./autogen.sh
    ./configure --prefix=${DEPS_SOURCE}/libunwind/build --disable-shared --with-pic
    make CFLAGS=-fPIC -j4
    make CFLAGS=-fPIC install
    cd build
    cp -a lib/libunwind.a ${DEPS_PREFIX}/lib
    cp -a include/libunwind.h ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/libunwind_1_0_0"
fi

# gperftools (tcmalloc)
if [ ! -f "${FLAG_DIR}/gperftools_2_5_0" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libtcmalloc.a" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libtcmalloc_minimal.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/gperftools" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/gperftools" ]; then
        rm -rf ${DEPS_SOURCE}/gperftools
    fi
    unzip ${DEPS_PACKAGE}/gperftools-gperftools-2.5.zip -d .
    mv gperftools-gperftools-2.5 gperftools && cd gperftools
    mkdir build
    ./autogen.sh
    ./configure --prefix=${DEPS_SOURCE}/gperftools/build --disable-shared --with-pic CPPFLAGS=-I${DEPS_PREFIX}/include LDFLAGS=-L${DEPS_PREFIX}/lib
    make -j4
    make install
    cd build
    cp -a lib/libtcmalloc.a lib/libtcmalloc_minimal.a ${DEPS_PREFIX}/lib
    cp -a include/gperftools ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/gperftools_2_5_0"
fi

:<<\EOF
# tbb
if [ ! -f "${FLAG_DIR}/tbb_2017_U7" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libtbb.so" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libtbb.so.2" ] \
    || [ ! -d "${DEPS_PREFIX}/include/tbb" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/tbb" ]; then
        rm -rf ${DEPS_SOURCE}/tbb
    fi
    unzip ${DEPS_PACKAGE}/tbb-2017_U7.zip -d .
    mv tbb-2017_U7 tbb && cd tbb
    make
    # Note: replace linux_intel64_gcc_cc4.8_libc2.19_kernel3.13.0_release with your system
    cd build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.13.0_release
    cp -a libtbb.so libtbb.so.2 ${DEPS_PREFIX}/lib
    cd ../..
    cp -a include/tbb ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/tbb_2017_U7"
fi

# glog
if [ ! -f "${FLAG_DIR}/glog_0_3_4" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/glog" ] ; then
    	rm -rf ${DEPS_SOURCE}/glog
    fi
    unzip ${DEPS_PACKAGE}/glog-0.3.4.zip -d .
    mv glog-0.3.4 glog
    cd glog
    ./configure
    make -j4
    sudo make install
    sudo ldconfig
    touch "${FLAG_DIR}/glog_0_3_4"
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

# libco
if [ ! -f "${FLAG_DIR}/libco-master" ] \
	|| [ ! -f "${DEPS_PREFIX}/lib/libcolib.a" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/libco" ]; then
    	rm -rf ${DEPS_SOURCE}/libco
    fi
    unzip ${DEPS_PACKAGE}/libco-master.zip -d .
    mv libco-master libco
    cp -a libco ${DEPS_PREFIX}/include
    cd libco
    make
    cp -a lib/libcolib.a ${DEPS_PREFIX}/lib
    touch "${FLAG_DIR}/libco-master"
fi

# libuv
if [ ! -f "${FLAG_DIR}/libuv_1_12_0" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/libuv" ] ; then
    	rm -rf ${DEPS_SOURCE}/libuv
    fi
    unzip ${DEPS_PACKAGE}/libuv-1.12.0.zip -d .
    mv libuv-1.12.0 libuv && cd libuv
    sh autogen.sh
    ./configure
    make -j4
    make check
    sudo make install
    sudo ldconfig
    touch "${FLAG_DIR}/libuv_1_12_0"
fi

# redis
if [ ! -f "${FLAG_DIR}/redis_3_2_11" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/redis" ] ; then
        rm -rf ${DEPS_SOURCE}/redis
    fi
    tar zxvf ${DEPS_PACKAGE}/redis-3.2.11.tar.gz -C .
    mv redis-3.2.11 redis
    cd redis
    make
    make test
    touch "${FLAG_DIR}/redis_3_2_11"
fi

# hiredis
if [ ! -f "${FLAG_DIR}/hiredis_0_13_3" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libhiredis.a" ] \
    || [ ! -d "${DEPS_PREFIX}/include/hiredis" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/hiredis" ] ; then
        rm -rf ${DEPS_SOURCE}/hiredis
    fi
    unzip ${DEPS_PACKAGE}/hiredis-0.13.3.zip -d .
    mv hiredis-0.13.3 hiredis
    cd hiredis
    make
    mkdir ${DEPS_PREFIX}/include/hiredis
    cp -a hiredis.h read.h sds.h ${DEPS_PREFIX}/include/hiredis
    cp -a libhiredis.a ${DEPS_PREFIX}/lib
    touch "${FLAG_DIR}/hiredis_0_13_3"
fi

# zeromq
if [ ! -f "${FLAG_DIR}/zeromq_4_2_2" ] \
    || [ ! -f "${DEPS_PREFIX}/lib/libzmq.a" ] \
    || [ ! -f "${DEPS_PREFIX}/include/zmq.h" ]; then
    cd ${DEPS_SOURCE}
    if [ -d "${DEPS_SOURCE}/zeromq" ] ; then
        rm -rf ${DEPS_SOURCE}/zeromq
    fi
    tar zxvf ${DEPS_PACKAGE}/zeromq-4.2.2.tar.gz -C .
    mv zeromq-4.2.2 zeromq && cd zeromq
    mkdir build
    ./configure --prefix=${DEPS_SOURCE}/zeromq/build
    make && make check
    make install
    cd build
    cp -a lib/libzmq.a ${DEPS_PREFIX}/lib
    cp -a include/zmq.h include/zmq_utils.h ${DEPS_PREFIX}/include
    touch "${FLAG_DIR}/zeromq_4_2_2"
fi
EOF

cd ${WORK_DIR}

echo "build complete"
echo "Done"
########################################
# build
########################################

#make clean
#make -j4

