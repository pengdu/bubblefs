// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/compiler_specific.h

#ifndef BUBBLEFS_PLATFORM_COMPILER_SPECIFIC_H_
#define BUBBLEFS_PLATFORM_COMPILER_SPECIFIC_H_

#include "platform/platform.h"

#if defined(COMPILER_MSVC)

// Macros for suppressing and disabling warnings on MSVC.
//
// Warning numbers are enumerated at:
// http://msdn.microsoft.com/en-us/library/8x5x43k7(VS.80).aspx
//
// The warning pragma:
// http://msdn.microsoft.com/en-us/library/2c8f766e(VS.80).aspx
//
// Using __pragma instead of #pragma inside macros:
// http://msdn.microsoft.com/en-us/library/d9x1s805.aspx

// MSVC_SUPPRESS_WARNING disables warning |n| for the remainder of the line and
// for the next line of the source file.
#define MSVC_SUPPRESS_WARNING(n) __pragma(warning(suppress:n))

// MSVC_PUSH_DISABLE_WARNING pushes |n| onto a stack of warnings to be disabled.
// The warning remains disabled until popped by MSVC_POP_WARNING.
#define MSVC_PUSH_DISABLE_WARNING(n) __pragma(warning(push)) \
                                     __pragma(warning(disable:n))

// MSVC_PUSH_WARNING_LEVEL pushes |n| as the global warning level.  The level
// remains in effect until popped by MSVC_POP_WARNING().  Use 0 to disable all
// warnings.
#define MSVC_PUSH_WARNING_LEVEL(n) __pragma(warning(push, n))

// Pop effects of innermost MSVC_PUSH_* macro.
#define MSVC_POP_WARNING() __pragma(warning(pop))

#define MSVC_DISABLE_OPTIMIZE() __pragma(optimize("", off))
#define MSVC_ENABLE_OPTIMIZE() __pragma(optimize("", on))

// Allows exporting a class that inherits from a non-exported base class.
// This uses suppress instead of push/pop because the delimiter after the
// declaration (either "," or "{") has to be placed before the pop macro.
//
// Example usage:
// class EXPORT_API Foo : NON_EXPORTED_BASE(public Bar) {
//
// MSVC Compiler warning C4275:
// non dll-interface class 'Bar' used as base for dll-interface class 'Foo'.
// Note that this is intended to be used only when no access to the base class'
// static data is done through derived classes or inline methods. For more info,
// see http://msdn.microsoft.com/en-us/library/3tdb471s(VS.80).aspx
#define NON_EXPORTED_BASE(code) MSVC_SUPPRESS_WARNING(4275) \
                                code

#else  // Not MSVC

#define MSVC_SUPPRESS_WARNING(n)
#define MSVC_PUSH_DISABLE_WARNING(n)
#define MSVC_PUSH_WARNING_LEVEL(n)
#define MSVC_POP_WARNING()
#define MSVC_DISABLE_OPTIMIZE()
#define MSVC_ENABLE_OPTIMIZE()
#define NON_EXPORTED_BASE(code) code

#endif  // COMPILER_MSVC


// The C++ standard requires that static const members have an out-of-class
// definition (in a single compilation unit), but MSVC chokes on this (when
// language extensions, which are required, are enabled). (You're only likely to
// notice the need for a definition if you take the address of the member or,
// more commonly, pass it to a function that takes it as a reference argument --
// probably an STL function.) This macro makes MSVC do the right thing. See
// http://msdn.microsoft.com/en-us/library/34h23df8(v=vs.100).aspx for more
// information. Use like:
//
// In .h file:
//   struct Foo {
//     static const int kBar = 5;
//   };
//
// In .cc file:
//   STATIC_CONST_MEMBER_DEFINITION const int Foo::kBar;
#if defined(COMPILER_MSVC)
#define STATIC_CONST_MEMBER_DEFINITION __declspec(selectany)
#else
#define STATIC_CONST_MEMBER_DEFINITION
#endif

/**
 * Macro for marking functions as having public visibility.
 * Ported from folly/CPortability.h
 */
#ifndef GNUC_PREREQ
#if defined __GNUC__ && defined __GNUC_MINOR__
#define GNUC_PREREQ(maj, min) \
  ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
#define GNUC_PREREQ(maj, min) 0
#endif
#endif

#if defined(COMPILER_GCC)
#define NORETURN __attribute__((noreturn))
#define NORETURN_PTR __attribute__((__noreturn__))
#else
#define NORETURN
#define NORETURN_PTR
#endif

// Annotate a variable or function indicating it's ok if the variable or function is not used.
// (Typically used to silence a compiler warning when the assignment
// is important for some other reason.)
// Use like:
//   int x ALLOW_UNUSED = ...;
//   int fool() ALLOW_UNUSED;
#if defined(COMPILER_GCC)
#define ALLOW_UNUSED __attribute__((unused))
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ALLOW_UNUSED
#define ATTRIBUTE_UNUSED
#endif

// Annotate a function indicating it should not be inlined.
// Use like:
//   NOINLINE void DoStuff() { ... }
#if defined(COMPILER_GCC)
#define NOINLINE __attribute__((noinline))
#elif defined(COMPILER_MSVC)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

#if defined(BASE_CXX11_ENABLED)
#define INLINE inline
#else
#define INLINE
#endif

#ifndef BASE_FORCE_INLINE
#if defined(COMPILER_MSVC)
#define BASE_FORCE_INLINE    __forceinline
#else
#define BASE_FORCE_INLINE inline __attribute__((always_inline))
#endif
#endif  // BASE_FORCE_INLINE

// Specify memory alignment for structs, classes, etc.
// Use like:
//   class ALIGNAS(16) MyClass { ... }
//   ALIGNAS(16) int array[4];
#if defined(COMPILER_MSVC)
#define ALIGNAS(byte_alignment) __declspec(align(byte_alignment))
#elif defined(COMPILER_GCC)
#define ALIGNAS(byte_alignment) __attribute__((aligned(byte_alignment)))
#endif

// Return the byte alignment of the given type (available at compile time).  Use
// sizeof(type) prior to checking __alignof to workaround Visual C++ bug:
// http://goo.gl/isH0C
// Use like:
//   ALIGNOF(int32_t)  // this would be 4
#if defined(COMPILER_MSVC)
#define ALIGNOF(type) (sizeof(type) - sizeof(type) + __alignof(type))
#elif defined(COMPILER_GCC)
#define ALIGNOF(type) __alignof__(type)
#endif

// Annotate a virtual method indicating it must be overriding a virtual
// method in the parent class.
// Use like:
//   virtual void foo() OVERRIDE;
#if defined(__clang__) || defined(COMPILER_MSVC)
#define OVERRIDE override
#elif defined(COMPILER_GCC) && __cplusplus >= 201103 && \
      (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) >= 40700
// GCC 4.7 supports explicit virtual overrides when C++11 support is enabled.
#define OVERRIDE override
#else
#define OVERRIDE
#endif

// Annotate a virtual method indicating that subclasses must not override it,
// or annotate a class to indicate that it cannot be subclassed.
// Use like:
//   virtual void foo() FINAL;
//   class B FINAL : public A {};
#if defined(__clang__) || defined(COMPILER_MSVC)
#define FINAL final
#elif defined(COMPILER_GCC) && __cplusplus >= 201103 && \
      (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) >= 40700
// GCC 4.7 supports explicit virtual overrides when C++11 support is enabled.
#define FINAL final
#else
#define FINAL
#endif

// Annotate a function indicating the caller must examine the return value.
// Use like:
//   int foo() WARN_UNUSED_RESULT;
// To explicitly ignore a result, see |ignore_result()| in "butil/basictypes.h".
// FIXME(gejun): GCC 3.4 report "unused" variable incorrectly (actually used).
#if defined(COMPILER_GCC) && __cplusplus >= 201103 && \
      (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) >= 40700
#define WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#define MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define WARN_UNUSED_RESULT
#define MUST_USE_RESULT
#endif

// Tell the compiler a function is using a printf-style format string.
// |format_param| is the one-based index of the format string parameter;
// |dots_param| is the one-based index of the "..." parameter.
// For v*printf functions (which take a va_list), pass 0 for dots_param.
// (This is undocumented but matches what the system C headers do.)
#if defined(COMPILER_GCC)
#define PRINTF_FORMAT(format_param, dots_param) \
    __attribute__((format(printf, format_param, dots_param)))
#else
#define PRINTF_FORMAT(format_param, dots_param)
#endif

#if defined(COMPILER_GCC)
#define PRINTF_ATTRIBUTE(string_index, first_to_check) \
   __attribute__((__format__(__printf__, string_index, first_to_check)))
#else
#define PRINTF_ATTRIBUTE(string_index, first_to_check)
#endif

#if defined(COMPILER_GCC)
#define SCANF_ATTRIBUTE(string_index, first_to_check) \
   __attribute__((__format__(__scanf__, string_index, first_to_check)))
#else
#define SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// WPRINTF_FORMAT is the same, but for wide format strings.
// This doesn't appear to yet be implemented in any compiler.
// See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=38308 .
#define WPRINTF_FORMAT(format_param, dots_param)
// If available, it would look like:
//   __attribute__((format(wprintf, format_param, dots_param)))

// MemorySanitizer annotations.
#if defined(MEMORY_SANITIZER) && !defined(OS_NACL)
#include <sanitizer/msan_interface.h>

// Mark a memory region fully initialized.
// Use this to annotate code that deliberately reads uninitialized data, for
// example a GC scavenging root set pointers from the stack.
#define MSAN_UNPOISON(p, s)  __msan_unpoison(p, s)
#else  // MEMORY_SANITIZER
#define MSAN_UNPOISON(p, s)
#endif  // MEMORY_SANITIZER

// Macro useful for writing cross-platform function pointers.
#if !defined(CDECL)
#if defined(OS_WIN)
#define CDECL __cdecl
#else  // defined(OS_WIN)
#define CDECL
#endif  // defined(OS_WIN)
#endif  // !defined(CDECL)

// Mark a branch likely or unlikely to be true.
// We can't remove the BAIDU_ prefix because the name is likely to conflict,
// namely kylin already has the macro.
#if defined(COMPILER_GCC)
#  if defined(__cplusplus)
#    define PREDICT_TRUE(x) (__builtin_expect((bool)(x), true))
#    define PREDICT_FALSE(x) (__builtin_expect((bool)(x), false))
#    define LIKELY(expr) (__builtin_expect((bool)(expr), true))
#    define UNLIKELY(expr) (__builtin_expect((bool)(expr), false))
#  else
#    define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#    define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#    define LIKELY(expr) (__builtin_expect(!!(expr), 1))
#    define UNLIKELY(expr) (__builtin_expect(!!(expr), 0))
#  endif
#else
#  define PREDICT_TRUE(x)
#  define PREDICT_FALSE(x)
#  define LIKELY(expr) (expr)
#  define UNLIKELY(expr) (expr)
#endif

// BAIDU_DEPRECATED void dont_call_me_anymore(int arg);
// ...
// warning: 'void dont_call_me_anymore(int)' is deprecated
#if defined(COMPILER_GCC)
# define DEPRECATED __attribute__((deprecated))
#elif defined(COMPILER_MSVC)
# define DEPRECATED __declspec(deprecated)
#else
# define DEPRECATED
#endif

#if defined(COMPILER_GCC)
# define ATTRIBUTE_COLD __attribute__((cold))
#else
# define ATTRIBUTE_COLD
#endif

// Mark function as weak. This is GCC only feature.
#if defined(COMPILER_GCC)
# define ATTRIBUTE_WEAK __attribute__((weak))
#else
# define ATTRIBUTE_WEAK
#endif

#if defined(COMPILER_GCC)
# define ATTRIBUTE_PACKED __attribute__((packed))
#else
# define ATTRIBUTE_PACKED
#endif

// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
#if defined(COMPILER_GCC)
# define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
# define PREFETCH(addr, rw, locality)
#endif

#ifndef NOEXCEPT
# if defined(BASE_CXX11_ENABLED)
#  define NOEXCEPT noexcept(true)
# else
#  define NOEXCEPT
# endif
#endif

#ifndef THROW_EXCEPT
# if defined(BASE_CXX11_ENABLED)
#  define THROW_EXCEPT noexcept(false)
# else
#  define THROW_EXCEPT
# endif
#endif

// Cacheline related --------------------------------------

#ifndef CACHE_LINE_SIZE
  #if defined(__s390__)
    #define CACHE_LINE_SIZE 256U
  #elif defined(__powerpc__) || defined(__aarch64__)
    #define CACHE_LINE_SIZE 128U
  #else
    #define CACHE_LINE_SIZE 64U
  #endif
#endif

#ifdef _MSC_VER
# define CACHE_LINE_ALIGNMENT __declspec(align(CACHE_LINE_SIZE))
#elifdef __GNUC__
# define CACHE_LINE_ALIGNMENT __attribute__((aligned(CACHE_LINE_SIZE)))
#else
# define CACHE_LINE_ALIGNMENT
#endif /* _MSC_VER */

/*! \brief Whether cxx11 thread local is supported */
#ifndef BASE_CXX11_THREAD_LOCAL
#if defined(_MSC_VER)
#define BASE_CXX11_THREAD_LOCAL (_MSC_VER >= 1900)
#elif defined(__clang__)
#define BASE_CXX11_THREAD_LOCAL (__has_feature(cxx_thread_local))
#else
#define BASE_CXX11_THREAD_LOCAL (__cplusplus >= 201103L)
#endif
#endif

/*!
 * \brief Enable std::thread related modules,
 *  Used to disable some module in mingw compile.
 */
#ifndef BASE_ENABLE_STD_THREAD
#define BASE_ENABLE_STD_THREAD (__cplusplus >= 201103L)
#endif

////////////////////////////////////////////////////////////

// dynamic cast reroute: if RTTI is disabled, go to reinterpret_cast
template <typename Dst, typename Src>
INLINE Dst dynamic_cast_if_rtti(Src ptr) {
#ifdef __GXX_RTTI
  return dynamic_cast<Dst>(ptr);
#else
  return reinterpret_cast<Dst>(ptr);
#endif
}

#endif  // BUBBLEFS_PLATFORM_COMPILER_SPECIFIC_H_