// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/base/static_assert.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_STATIC_ASSERT_H_
#define BUBBLEFS_UTILS_TOFT_BASE_STATIC_ASSERT_H_

#include "platform/macros.h"

#ifdef BASE_CXX11_ENABLED

#define MYTOFT_STATIC_ASSERT(e, ...) static_assert(e, "" __VA_ARGS__)

#else

namespace bubblefs {
namespace mytoft {

template <bool x> struct static_assertion_failure;

template <> struct static_assertion_failure<true> {
    enum { value = 1 };
};

template<int x> struct static_assert_test {};

} // namespace mytoft
} // namespace bubblefs

// Static assertions during compilation, Usage:
// MYTOFT_STATIC_ASSERT(sizeof(Foo) == 48, "Size of Foo must equal to 48");
#define MYTOFT_STATIC_ASSERT(e, ...) \
    typedef ::bubblefs::mytoft::static_assert_test < \
            sizeof(::bubblefs::mytoft::static_assertion_failure<static_cast<bool>(e)>)> \
            static_assert_failed##__LINE__

#endif // BASE_CXX11_ENABLED

#endif // BUBBLEFS_UTILS_TOFT_BASE_STATIC_ASSERT_H_