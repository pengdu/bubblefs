/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/utils/Common.h

#ifndef BUBBLEFS_UTILS_PADDLE_UTIL_H_
#define BUBBLEFS_UTILS_PADDLE_UTIL_H_

#include <sys/syscall.h>  // for syscall()
#include <sys/types.h>
#include <assert.h>
#include <fenv.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "platform/base_error.h"

#if defined(__ANDROID__) && (__ANDROID_API__ < 21)
inline int rand_r(unsigned int* seedp) {
  (void)seedp;
  return rand();
}
#endif

/**
 * Loop over the elements in a container
 * TODO(yuyang18): It's this foreach useful? Why not use C++ 11 foreach,
 *                 or make it a inline method?
 * Example:
 * FOR_EACH(it, array) {
 *  sum += *it;
 * }
 */
#define FOR_EACH(iterator_name, container)                              \
  for (auto iterator_name = (container).begin(), e = (container).end(); \
       iterator_name != e;                                              \
       ++iterator_name)

/**
 * Loop over the elements in a container in reverse order
 * TODO(yuyang18): It's this foreach useful? Why not use C++ 11 foreach,
 *                 or make it a inline method?
 * Example:
 * FOR_EACH_R(it, array) {
 *  sum += *it;
 * }
 */
#define FOR_EACH_R(iterator_name, container)                              \
  for (auto iterator_name = (container).rbegin(), e = (container).rend(); \
       iterator_name != e;                                                \
       ++iterator_name)

namespace bubblefs {
namespace mypaddle {

#ifdef PADDLE_TYPE_DOUBLE
using real = double;
#else
using real = float;
#endif  
  
// return the thread id used by glog
pid_t getTID();

/**
 * return the 1-based index of the highest bit set
 *
 * for x > 0:
 * \f[
 *    findLastSet(x) = 1 + \floor*{\log_{2}x}
 * \f]
 */
inline constexpr size_t findLastSet(size_t x) {
  return std::is_same<size_t, unsigned int>::value
             ? (x ? 8 * sizeof(x) - __builtin_clz(x) : 0)
             : (std::is_same<size_t, unsigned long>::value  // NOLINT
                    ? (x ? 8 * sizeof(x) - __builtin_clzl(x) : 0)
                    : (x ? 8 * sizeof(x) - __builtin_clzll(x) : 0));
}

/**
 * calculate the non-negative remainder of a/b
 * @param[in] a
 * @param[in] b, should be positive
 * @return the non-negative remainder of a / b
 */
inline int mod(int a, int b) {
  int r = a % b;
  return r >= 0 ? r : r + b;
}

/**
 * find the value given a key k from container c.
 * If the key can be found, the value is stored in *value
 * return true if the key can be found. false otherwise.
 */
template <class K, class V, class C>
bool mapGet(const K& k, const C& c, V* value) {
  auto it = c.find(k);
  if (it != c.end()) {
    *value = it->second;
    return true;
  } else {
    return false;
  }
}

template <class Container, class T>
static bool contains(const Container& container, const T& val) {
  return std::find(container.begin(), container.end(), val) != container.end();
}

/**
 * pop and get the front element of a container
 */
template <typename Container>
typename Container::value_type pop_get_front(Container& c) {
  typename Container::value_type v;
  swap(v, c.front());
  c.pop_front();
  return v;
}

/**
 * Initialize some creators or initFunctions for layers and data
 * providers.
 * Client codes should call this function before they refer any other
 * codes that use the layer class and data provider class.
 *
 * Codes inside 'core' directory can call initMain which calls
 * runInitFunctions directly, while codes outside core can simply
 * call runInitFunctions if they don't need the commandline flags
 * designed for PADDLE main procedure.
 */
void runInitFunctions();

// read the whole file into a string
std::string readFile(const std::string& fileName);

// copy file to path
void copyFileToPath(const std::string& file, const std::string& path);

// test file exist or not
bool fileExist(const char* filename);
// touch file if not exist
void touchFile(const char* filename);
// make dir if not exist
void mkDir(const char* filename);
void mkDirRecursively(const char* filename);

void rmDir(const char* folderName);

// load a file list file into a vector(fileList)
void loadFileList(const std::string& fileListFileName,
                  std::vector<std::string>& fileList);

/**
 * Register a function, the function will be called in initMain(). Functions
 * with higher priority will be called first. The execution order of functions
 * with same priority is not defined.
 */
void registerInitFunction(std::function<void()> func, int priority = 0);
class InitFunction {
public:
  explicit InitFunction(std::function<void()> func, int priority = 0) {
    registerInitFunction(func, priority);
  }
};

/**
 * Return value: memory usage ratio (from 0-1)
 */
double getMemoryUsage();

/**
 * split array by index.
 * used by sync multi thread task,
 * each thread call calcSplitArrayInterval with thread id,
 * get a interval as return.
 * input:
 * *totalSize* is array size,
 * *tId* is thread id, *tSize* is total worker thread num
 * output:
 * start and end index as a std::pair
 */
inline std::pair<size_t, size_t> calcSplitArrayInterval(size_t totalSize,
                                                        size_t tId,
                                                        size_t tSize) {
  size_t start = totalSize * tId / tSize;
  size_t end = totalSize * (tId + 1) / tSize;
  return std::make_pair(start, end);
}

/**
 * same as above, but split at boundary of block.
 */
inline std::pair<size_t, size_t> calcSplitArrayInterval(size_t totalSize,
                                                        size_t tId,
                                                        size_t tSize,
                                                        size_t blockSize) {
  size_t numBlocks = totalSize / blockSize;
  if (numBlocks * blockSize < totalSize) {
    numBlocks++;
  }

  auto interval = calcSplitArrayInterval(numBlocks, tId, tSize);
  size_t start = std::min(interval.first * blockSize, totalSize);
  size_t end = std::min(interval.second * blockSize, totalSize);

  return std::make_pair(start, end);
}

// Calculate the number of pservers/dservers based
// on the host list and port_num.
size_t calculateServiceNum(const std::string& pservers, int ports_num);

/**
 * sort and unique ids vector.
 */
inline void uniqueIds(std::vector<uint32_t>& ids) {
  std::sort(ids.begin(), ids.end());
  auto endpos = std::unique(ids.begin(), ids.end());
  ids.erase(endpos, ids.end());
}

/**
 * Read Type value
 */
template <typename T>
T readT(char*& p, const char* pEnd) {
  int minus = pEnd - p - sizeof(T);
  assert(0 > minus);
  T v = *reinterpret_cast<T*>(p);
  p += sizeof(T);
  return v;
}

void memcpyWithCheck(void* dest,
                     const void* src,
                     size_t num,
                     const void* srcEnd);

/**
 * A global sync thread pool, has #FLAGS_trainer_count of threads.
 * can be used in main thread.
 */
class SyncThreadPool;
SyncThreadPool* getGlobalSyncThreadPool(int trainer_count);

namespace path {

// directory separator
const char sep = '/';

// Return the base name of pathname path.
std::string basename(const std::string& path);

// Return the directory name of path. If the path does not contains any
// directory, it returns an empty string.
std::string dirname(const std::string& path);

/*
  Join two path components intelligently.
  The return value is the concatenation of part1 and part2 with exactly one
  directory separator (path.sep) following each non-empty part except the last,
  meaning that the result will only end in a separator if the last part is
  empty.
  If a component is an absolute path, all previous components are thrown away
  and joining continues from the absolute path component.
*/
std::string join(const std::string& part1, const std::string& part2);

template <typename... Args>
std::string join(const std::string& part1,
                 const std::string& part2,
                 Args... args) {
  return join(join(part1, part2), args...);
}

}  // namespace path

/**
 * A Checker for each invoke of method in same thread.
 */
class SameThreadChecker {
public:
  SameThreadChecker() {}

  /**
   * Disable copy
   */
  SameThreadChecker(const SameThreadChecker& other) = delete;
  SameThreadChecker& operator=(const SameThreadChecker& other) = delete;

  /**
   * Each invoke of check method should be in same thread, otherwise, it will
   * failed and core dump.
   */
  void check() {
    std::thread::id curThreadId = std::this_thread::get_id();
    std::call_once(onceFlag_, [&] { invokeThreadId_ = curThreadId; });
    PANIC_ENFORCE_EQ(invokeThreadId_, curThreadId); // "This method should invoke in same thread"
  }

private:
  std::once_flag onceFlag_;
  std::thread::id invokeThreadId_;
};

/**
 * Key-Value Cache Helper.
 *
 * It store a object instance global. User can invoke get method by key and a
 * object creator callback. If there is a instance stored in cache, then it will
 * return a shared_ptr of it, otherwise, it will invoke creator callback, create
 * a new instance store global, and return it.
 *
 * The cache instance will release when nobody hold a reference to it.
 *
 * The KType is the key type.
 * The VType is the value type.
 * The Hash is the key hasher object.
 */
template <typename KType, typename VType, typename Hash>
class WeakKVCache {
public:
  WeakKVCache() {}

  std::shared_ptr<VType> get(const KType& key,
                             const std::function<VType*()>& creator) {
    std::lock_guard<std::mutex> guard(this->lock_);
    auto it = this->storage_.find(key);
    if (it != this->storage_.end()) {
      auto& val = it->second;
      auto retVal = val.lock();
      if (retVal != nullptr) {
        return retVal;
      }  // else fall trough. Because it is WeakPtr Cache.
    }
    auto rawPtr = creator();
    assert(rawPtr != nullptr);
    std::shared_ptr<VType> retVal(rawPtr);
    this->storage_[key] = retVal;
    return retVal;
  }

private:
  std::mutex lock_;
  std::unordered_map<KType, std::weak_ptr<VType>, Hash> storage_;
};

/**
 * @brief The ScopedCallbacks class is a callback invoker when object is
 *        created and destroyed.
 */
template <typename CallbackType, typename... Args>
class ScopedCallbacks {
public:
  ScopedCallbacks(CallbackType enter, CallbackType exit, Args&... args)
      : exit_(std::bind(exit, args...)) {
    enter(args...);
  }

  ScopedCallbacks(const ScopedCallbacks& other) = delete;
  ScopedCallbacks& operator=(const ScopedCallbacks& other) = delete;

  ~ScopedCallbacks() { exit_(); }

private:
  std::function<void()> exit_;
};

/**
 * std compatible allocator with memory alignment.
 * @tparam T type of allocator elements.
 * @tparam Alignment the alignment in bytes.
 */
template <typename T, size_t Alignment>
class AlignedAllocator {
public:
  /// std campatible typedefs.
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  T* address(T& r) const { return &r; }

  const T* address(const T& r) const { return &r; }

  size_t max_size() const {
    return std::numeric_limits<size_t>::max() / sizeof(T);
  }

  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };

  bool operator==(const AlignedAllocator& other) const { return true; }

  bool operator!=(const AlignedAllocator& other) const {
    return !(*this == &other);
  }

  void construct(const T* p, const T& t) const {
    void* pv = const_cast<T*>(p);
    new (pv) T(t);
  }

  void deallocate(const T* p, const size_type n) const {
    (void)(n);  // UNUSED n
    free(const_cast<T*>(p));
  }

  void destroy(const T* p) const { p->~T(); }

  AlignedAllocator() {}
  ~AlignedAllocator() {}

  AlignedAllocator(const AlignedAllocator&) {}
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  /**
   * @brief allocate n elements of type T, the first address is aligned by
   *        Alignment bytes.
   * @param n element count.
   * @return begin address of allocated buffer
   * @throw std::length_error for n * sizeof(T) is overflowed.
   * @throw std::bad_alloc
   */
  T* allocate(const size_type n) const {
    if (n == 0) {
      return nullptr;
    }
    if (n > max_size()) {
      throw std::length_error("AlignAllocator<T>::allocate() - Int Overflow.");
    }
    void* r = nullptr;
    PANIC_ENFORCE_EQ(posix_memalign(&r, Alignment * 8, sizeof(T) * n), 0);
    if (r == nullptr) {
      throw std::bad_alloc();
    } else {
      return static_cast<T*>(r);
    }
  }

  template <typename U>
  T* allocate(const std::size_t n, const U* /* const hint */) const {
    return this->allocate(n);
  }

private:
  AlignedAllocator& operator=(const AlignedAllocator&);  // disable
};

}  // namespace mypaddle
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PADDLE_UTIL_H_