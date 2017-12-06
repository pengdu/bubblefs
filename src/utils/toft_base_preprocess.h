// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/base/preprocess.h
// toft/base/preprocess/disallow_in_header.h
// toft/base/preprocess/join.h
// toft/base/preprocess/stringize.h
// toft/base/preprocess/varargs.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_PREPROCESS_H_
#define BUBBLEFS_UTILS_TOFT_BASE_PREPROCESS_H_

#include "utils/toft_base_static_assert.h"

/// disallow macro be used in header files
///
/// @example
/// #define SOMEMACRO() TOFT_PP_DISALLOW_IN_HEADER_FILE()
/// A compile error will be issued if SOMEMACRO() is used in header files
#ifdef __GNUC__
# define MYTOFT_PP_DISALLOW_IN_HEADER_FILE() \
    MYTOFT_STATIC_ASSERT(__INCLUDE_LEVEL__ == 0, "This macro can not be used in header files");
#else
# define MYTOFT_PP_DISALLOW_IN_HEADER_FILE()
#endif

/// Helper macro to join 2 tokens
/// example: TOFT_PP_JOIN(UCHAR_MAX, SCHAR_MIN) -> 255(-128)
/// The following piece of macro magic joins the two
/// arguments together, even when one of the arguments is
/// itself a macro (see 16.3.1 in C++ standard). The key
/// is that macro expansion of macro arguments does not
/// occur in TOFT_PP_DO_JOIN2 but does in TOFT_PP_DO_JOIN.
#define MYTOFT_PP_JOIN(X, Y) MYTOFT_PP_DO_JOIN(X, Y)
#define MYTOFT_PP_DO_JOIN(X, Y) MYTOFT_PP_DO_JOIN2(X, Y)
#define MYTOFT_PP_DO_JOIN2(X, Y) X##Y

/// Converts the parameter X to a string after macro replacement
/// on X has been performed.
/// example: TOFT_PP_STRINGIZE(UCHAR_MAX) -> "255"
#define MYTOFT_PP_STRINGIZE(X) MYTOFT_PP_DO_STRINGIZE(X)
#define MYTOFT_PP_DO_STRINGIZE(X) #X

// Count the number of va args
#define MYTOFT_PP_N_ARGS(...) \
    MYTOFT_PP_N_ARGS_HELPER1(,##__VA_ARGS__, \
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define MYTOFT_PP_N_ARGS_HELPER1(...) MYTOFT_PP_N_ARGS_HELPER2(__VA_ARGS__)
#define MYTOFT_PP_N_ARGS_HELPER2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                               x11, x12, x13, x14, x15, n, ...) n


// Apply unary macro m to each arg of va
#define MYTOFT_PP_FOR_EACH_ARGS(m, ...) \
    MYTOFT_PP_FOR_EACH_ARGS_(MYTOFT_PP_N_ARGS(__VA_ARGS__), m, ##__VA_ARGS__)

#define MYTOFT_PP_FOR_EACH_ARGS_(n, m, ...) \
    MYTOFT_PP_JOIN(MYTOFT_PP_FOR_EACH_ARGS_, n)(m, ##__VA_ARGS__)

#define MYTOFT_PP_FOR_EACH_ARGS_0(m)
#define MYTOFT_PP_FOR_EACH_ARGS_1(m, a1) m(a1)
#define MYTOFT_PP_FOR_EACH_ARGS_2(m, a1, a2) m(a1) m(a2)
#define MYTOFT_PP_FOR_EACH_ARGS_3(m, a1, a2, a3) m(a1) m(a2) m(a3)
#define MYTOFT_PP_FOR_EACH_ARGS_4(m, a1, a2, a3, a4) m(a1) m(a2) m(a3) m(a4)
#define MYTOFT_PP_FOR_EACH_ARGS_5(m, a1, a2, a3, a4, a5) \
    m(a1) m(a2) m(a3) m(a4) m(a5)
#define MYTOFT_PP_FOR_EACH_ARGS_6(m, a1, a2, a3, a4, a5, a6) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6)
#define MYTOFT_PP_FOR_EACH_ARGS_7(m, a1, a2, a3, a4, a5, a6, a7) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7)
#define MYTOFT_PP_FOR_EACH_ARGS_8(m, a1, a2, a3, a4, a5, a6, a7, a8) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8)
#define MYTOFT_PP_FOR_EACH_ARGS_9(m, a1, a2, a3, a4, a5, a6, a7, a8, a9) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9)
#define MYTOFT_PP_FOR_EACH_ARGS_10(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10)
#define MYTOFT_PP_FOR_EACH_ARGS_11(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10) m(a11)
#define MYTOFT_PP_FOR_EACH_ARGS_12(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10) m(a11) m(a12)
#define MYTOFT_PP_FOR_EACH_ARGS_13(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10) m(a11) m(a12) m(a13)
#define MYTOFT_PP_FOR_EACH_ARGS_14(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10) m(a11) m(a12) m(a13) m(a14)
#define MYTOFT_PP_FOR_EACH_ARGS_15(m, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) \
    m(a1) m(a2) m(a3) m(a4) m(a5) m(a6) m(a7) m(a8) m(a9) m(a10) m(a11) m(a12) m(a13) m(a14) m(a15)

/// prevent macro substitution for function-like macros
/// if macro 'min()' was defined:
/// 'int min()' whill be substituted, but
/// 'int min MYTOFT_PP_PREVENT_MACRO_SUBSTITUTION()' will not be substituted.
#define MYTOFT_PP_PREVENT_MACRO_SUBSTITUTION

#endif // BUBBLEFS_UTILS_TOFT_BASE_PREPROCESS_H_
