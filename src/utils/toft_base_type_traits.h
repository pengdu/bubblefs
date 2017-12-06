// Copyright (c) 2013, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 2013-02-28

// toft/base/type_traits.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_TYPE_TRAITS_H_
#define BUBBLEFS_UTILS_TOFT_BASE_TYPE_TRAITS_H_

//#pragma once

#include <features.h>

#if __GNUC_PREREQ(4, 5) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define MYTOFT_HAS_STD_TYPE_TRAITS 1
#endif

#ifdef MYTOFT_HAS_STD_TYPE_TRAITS
#include <type_traits>
#endif

#undef MYTOFT_HAS_STD_TYPE_TRAITS

#endif // BUBBLEFS_UTILS_TOFT_BASE_TYPE_TRAITS_H_