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

if [ -d "${DEPS_SOURCE}" ]; then
    sudo rm -rf ${DEPS_SOURCE}
fi

if [ -d "${DEPS_PREFIX}" ]; then
    sudo rm -rf ${DEPS_PREFIX}
fi

if [ -d "${FLAG_DIR}" ]; then
    sudo rm -rf ${FLAG_DIR}
fi
