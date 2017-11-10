
#ifndef BUBBLEFS_PLATFORM_ATOMIC_SYNC_BUILTIN_GCC_H_
#define BUBBLEFS_PLATFORM_ATOMIC_SYNC_BUILTIN_GCC_H_

#define GCC_SYNC_FETCH_AND_ADD(ptr, value, ...) __sync_fetch_and_add(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_FETCH_AND_SUB(ptr, value, ...) __sync_fetch_and_sub(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_FETCH_AND_OR(ptr, value, ...) __sync_fetch_and_or(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_FETCH_AND_AND(ptr, value, ...) __sync_fetch_and_and(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_FETCH_AND_XOR(ptr, value, ...) __sync_fetch_and_xor(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_FETCH_AND_NAND(ptr, value, ...) __sync_fetch_and_nand(ptr, value, ##__VA_ARGS__)

#define GCC_SYNC_ADD_AND_FETCH(ptr, value, ...) __sync_add_and_fetch(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_SUB_AND_FETCH(ptr, value, ...) __sync_sub_and_fetch(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_OR_AND_FETCH(ptr, value, ...) __sync_or_and_fetch(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_AND_AND_FETCH(ptr, value, ...) __sync_and_and_fetch(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_XOR_AND_FETCH(ptr, value, ...) __sync_xor_and_fetch(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_NAND_AND_FETCH(ptr, value, ...) __sync_nand_and_fetch(ptr, value, ##__VA_ARGS__)

#define GCC_SYNC_BOOL_COMPARE_AND_SWAP(ptr, value, newval, ...) __sync_bool_compare_and_swap(ptr, value, newval, ##__VA_ARGS__)
#define GCC_SYNC_VAL_COMPARE_AND_SWAP(ptr, value, newval, ...) __sync_val_compare_and_swap(ptr, value, newval, ##__VA_ARGS__)

#define GCC_SYNC_SYNCHRONIZE(...) __sync_synchronize(##__VA_ARGS__)
#define GCC_SYNC_LOCK_TEST_AND_SET(ptr, value, ...) __sync_lock_test_and_set(ptr, value, ##__VA_ARGS__)
#define GCC_SYNC_LOCK_RELEASE(ptr, ...) __sync_lock_release(ptr, ##__VA_ARGS__)

#endif // BUBBLEFS_PLATFORM_ATOMIC_SYNC_BUILTIN_GCC_H_