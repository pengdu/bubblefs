/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

// Pebble/src/framework/dr/common/common.h

#ifndef BUBBLEFS_UTILS_PEBBLE_DR_COMMON_H_
#define BUBBLEFS_UTILS_PEBBLE_DR_COMMON_H_

#include <netinet/in.h>
#include <sys/types.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <exception>
#include <list>
#include <map>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include "platform/types.h"

#define T_VIRTUAL_CALL()

namespace bubblefs {
namespace mypebble { namespace dr {

class TEnumIterator : public std::iterator<std::forward_iterator_tag,
    std::pair<int, const char*> > {
public:
    TEnumIterator(int n,
        int* enums,
        const char** names) :
        ii_(0), n_(n), enums_(enums), names_(names) {
    }

    int operator ++() {
        return ++ii_;
    }

    bool operator != (const TEnumIterator& end) {
        (void)end;  // avoid "unused" warning with NDEBUG
        assert(end.n_ == -1);
        return (ii_ != n_);
    }

    std::pair<int, const char*> operator*() const {
        return std::make_pair(enums_[ii_], names_[ii_]);
    }

private:
    int ii_;
    const int n_;
    int* enums_;
    const char** names_;
};


} // namespace dr

class TOutput {
public:
    TOutput() : f_(&errorTimeWrapper) {}

    inline void setOutputFunction(void (*function)(const char *)) {
        f_ = function;
    }

    inline void operator()(const char *message) {
        f_(message);
    }

    // It is important to have a const char* overload here instead of
    // just the string version, otherwise errno could be corrupted
    // if there is some problem allocating memory when constructing
    // the string.
    void perror(const char *message, int errno_copy);
    inline void perror(const std::string &message, int errno_copy) {
        perror(message.c_str(), errno_copy);
    }

    void printf(const char *message, ...);

    static void errorTimeWrapper(const char* msg);

    /** Just like strerror_r but returns a C++ string object. */
    static std::string strerror_s(int errno_copy);

private:
    void (*f_)(const char *);
};

extern TOutput GlobalOutput;

class TException : public std::exception {
public:
    TException():
        message_() {}

    explicit TException(const std::string& message) :
        message_(message) {}

    virtual ~TException() throw() {}

    virtual const char* what() const throw() {
        if (message_.empty()) {
            return "Default TException.";
        } else {
            return message_.c_str();
        }
    }

protected:
    std::string message_;
};

} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_DR_COMMON_H_