/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Microsoft Corporation
 *
 * -=- Robust Distributed System Nucleus (rDSN) -=-
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// rdsn/include/dsn/cpp/callocator.h

#ifndef BUBBLEFS_UTILS_RDSN_CALLOCATOR_H_
#define BUBBLEFS_UTILS_RDSN_CALLOCATOR_H_

#include <inttypes.h>

namespace bubblefs {
namespace myrdsn {

typedef void *(*t_allocate)(uint32_t);
typedef void (*t_deallocate)(void *);

template <t_allocate a, t_deallocate d>
class callocator_object
{
public:
    void *operator new(size_t size) { return a(uint32_t(size)); }

    void operator delete(void *p) { d(p); }

    void *operator new[](size_t size) { return a((uint32_t)size); }

    void operator delete[](void *p) { d(p); }
};

//typedef callocator_object<dsn_transient_malloc, dsn_transient_free> transient_object;

template <typename T, t_allocate a, t_deallocate d>
class callocator
{
public:
    typedef T value_type;
    typedef size_t size_type;
    typedef const T *const_pointer;

    template <typename _Tp1>
    struct rebind
    {
        typedef callocator<_Tp1, a, d> other;
    };

    static T *allocate(size_type n) { return static_cast<T *>(a(uint32_t(n * sizeof(T)))); }

    static void deallocate(T *p, size_type n) { d(p); }

    callocator() throw() {}
    template <typename U>
    callocator(const callocator<U, a, d> &ac) throw()
    {
    }
    ~callocator() throw() {}
};

//template <typename T>
//using transient_allocator = callocator<T, dsn_transient_malloc, dsn_transient_free>;

} // namespace myrdsn
} // namespace bubblefs