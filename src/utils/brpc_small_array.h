
// brpc/src/butil/macros.h

#ifndef BUBBLEFS_UTILS_BRPC_SMALL_ARRAY_H_
#define BUBBLEFS_UTILS_BRPC_SMALL_ARRAY_H_

#include "platform/macros.h"

// DEFINE_SMALL_ARRAY(MyType, my_array, size, 64);
//   my_array is typed `MyType*' and as long as `size'. If `size' is not
//   greater than 64, the array is allocated on stack.
//
// NOTE: NEVER use ARRAY_SIZE(my_array) which is always 1.

#if defined(__cplusplus)
namespace bubblefs {
namespace base {
namespace internal {
template <typename T> struct ArrayDeleter {
    ArrayDeleter() : arr(0) {}
    ~ArrayDeleter() { delete[] arr; }
    T* arr;
};
} // namespace internal
} // namespace base
} // namespace bubblefs

// Many versions of clang does not support variable-length array with non-pod
// types, have to implement the macro differently.
#if !defined(__clang__)
# define DEFINE_SMALL_ARRAY(Tp, name, size, maxsize)                    \
    Tp* name = 0;                                                       \
    const unsigned name##_size = (size);                                \
    const unsigned name##_stack_array_size = (name##_size <= (maxsize) ? name##_size : 0); \
    Tp name##_stack_array[name##_stack_array_size];                     \
    ::bubblefs::base::internal::ArrayDeleter<Tp> name##_array_deleter;            \
    if (name##_stack_array_size) {                                      \
        name = name##_stack_array;                                      \
    } else {                                                            \
        name = new (::std::nothrow) Tp[name##_size];                    \
        name##_array_deleter.arr = name;                                \
    }
#else
// This implementation works for GCC as well, however it needs extra 16 bytes
// for ArrayCtorDtor.
namespace bubblefs {
namespace base {
namespace internal {
template <typename T> struct ArrayCtorDtor {
    ArrayCtorDtor(void* arr, unsigned size) : _arr((T*)arr), _size(size) {
        for (unsigned i = 0; i < size; ++i) { new (_arr + i) T; }
    }
    ~ArrayCtorDtor() {
        for (unsigned i = 0; i < _size; ++i) { _arr[i].~T(); }
    }
private:
    T* _arr;
    unsigned _size;
};
} // namespace internal
} // namespace base
} // namespace bubblefs

# define DEFINE_SMALL_ARRAY(Tp, name, size, maxsize)                    \
    Tp* name = 0;                                                       \
    const unsigned name##_size = (size);                                \
    const unsigned name##_stack_array_size = (name##_size <= (maxsize) ? name##_size : 0); \
    char name##_stack_array[sizeof(Tp) * name##_stack_array_size];      \
    ::bubblefs::base::internal::ArrayDeleter<char> name##_array_deleter;          \
    if (name##_stack_array_size) {                                      \
        name = (Tp*)name##_stack_array;                                 \
    } else {                                                            \
        name = (Tp*)new (::std::nothrow) char[sizeof(Tp) * name##_size];\
        name##_array_deleter.arr = (char*)name;                         \
    }                                                                   \
    const ::bubblefs::base::internal::ArrayCtorDtor<Tp> name##_array_ctor_dtor(name, name##_size);
#endif // !defined(__clang__)
#endif  // __cplusplus

#endif // BUBBLEFS_UTILS_BRPC_SMALL_ARRAY_H_