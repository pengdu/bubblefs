// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/base/uncopyable.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_UNCOPYABLE_H_
#define BUBBLEFS_UTILS_TOFT_BASE_UNCOPYABLE_H_

#include "platform/macros.h"

namespace bubblefs {
namespace mytoft {

/// The private base class way
namespace uncopyable_details  // protection from unintended ADL
{
class Uncopyable
{
    DECLARE_UNCOPYABLE(Uncopyable);
protected:
    Uncopyable() {}
    ~Uncopyable() {}
};
} // namespace uncopyable_details

typedef uncopyable_details::Uncopyable Uncopyable;

/*
Usage:
class Foo : private Uncopyable
{
};
*/

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TOFT_BASE_UNCOPYABLE_H_