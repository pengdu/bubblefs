/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* Implementations of various class and method modifier attributes. */

#ifndef BUBBLEFS_UTILS_MOZILLA_ATTRIBUTES_H_
#define BUBBLEFS_UTILS_MOZILLA_ATTRIBUTES_H_

#include <assert.h>

/*
 * MOZ_ALWAYS_INLINE is a macro which expands to tell the compiler that the
 * method decorated with it must be inlined, even if the compiler thinks
 * otherwise.  This is only a (much) stronger version of the inline hint:
 * compilers are not guaranteed to respect it (although they're much more likely
 * to do so).
 *
 * The MOZ_ALWAYS_INLINE_EVEN_DEBUG macro is yet stronger. It tells the
 * compiler to inline even in DEBUG builds. It should be used very rarely.
 */
#if defined(_MSC_VER)
#  define MOZ_ALWAYS_INLINE_EVEN_DEBUG     __forceinline
#elif defined(__GNUC__)
#  define MOZ_ALWAYS_INLINE_EVEN_DEBUG     __attribute__((always_inline)) inline
#else
#  define MOZ_ALWAYS_INLINE_EVEN_DEBUG     inline
#endif

#if !defined(DEBUG)
#  define MOZ_ALWAYS_INLINE     MOZ_ALWAYS_INLINE_EVEN_DEBUG
#elif defined(_MSC_VER) && !defined(__cplusplus)
#  define MOZ_ALWAYS_INLINE     __inline
#else
#  define MOZ_ALWAYS_INLINE     inline
#endif

/*
 * The MOZ_CONSTEXPR specifier declares that a C++11 compiler can evaluate a
 * function at compile time. A constexpr function cannot examine any values
 * except its arguments and can have no side effects except its return value.
 * The MOZ_CONSTEXPR_VAR specifier tells a C++11 compiler that a variable's
 * value may be computed at compile time.  It should be prefered to just
 * marking variables as MOZ_CONSTEXPR because if the compiler does not support
 * constexpr it will fall back to making the variable const, and some compilers
 * do not accept variables being marked both const and constexpr.
 */
#define MOZ_CONSTEXPR       constexpr
#define MOZ_CONSTEXPR_VAR   constexpr
#define MOZ_CONSTEXPR_TMPL  constexpr

#define MOZ_ASSERT(x) assert(x)
#define MOZ_RELEASE_ASSERT(x) assert(x)

/*
 * MOZ_IMPLICIT: Applies to constructors. Implicit conversion constructors
 * are disallowed by default unless they are marked as MOZ_IMPLICIT. This
 * attribute must be used for constructors which intend to provide implicit
 * conversions.
 */
#define MOZ_IMPLICIT __attribute__((annotate("moz_implicit")))

/* 
 * MOZ_INHERIT_TYPE_ANNOTATIONS_FROM_TEMPLATE_ARGS: Applies to template class
 * declarations where an instance of the template should be considered, for
 * static analysis purposes, to inherit any type annotations (such as
 * MOZ_MUST_USE and MOZ_STACK_CLASS) from its template arguments.
 */
#define MOZ_INHERIT_TYPE_ANNOTATIONS_FROM_TEMPLATE_ARGS \
  __attribute__((annotate("moz_inherit_type_annotations_from_template_args")))
  
#if defined(__clang__) || defined(__GNUC__)
#  define MOZ_LIKELY(x)   (__builtin_expect(!!(x), 1))
#  define MOZ_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#  define MOZ_LIKELY(x)   (!!(x))
#  define MOZ_UNLIKELY(x) (!!(x))
#endif

#endif // BUBBLEFS_UTILS_MOZILLA_ATTRIBUTES_H_