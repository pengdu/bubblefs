// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/base/Singleton.h

#ifndef BUBBLEFS_UTILS_MUDO_SINGLETON_H_
#define BUBBLEFS_UTILS_MUDO_SINGLETON_H_

#include <assert.h>
#include <stdlib.h> // atexit
#include <pthread.h>
#include "platform/macros.h"
#include <boost/concept_check.hpp>

namespace bubblefs {
namespace mymuduo
{
  
namespace detail
{
// This doesn't detect inherited member functions!
// http://stackoverflow.com/questions/1966362/sfinae-to-check-for-inherited-member-functions
template<typename T>
struct has_no_destroy
{
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  template <typename C> static char test(decltype(&C::no_destroy));
#else
  template <typename C> static char test(typeof(&C::no_destroy));
#endif
  template <typename C> static int32_t test(...);
  const static bool value = sizeof(test<T>(0)) == 1;
};
} // namespace detail

template<typename T>
class Singleton
{
 public:
  static T& instance()
  {
    pthread_once(&ponce_, &Singleton::init);
    assert(value_ != NULL);
    return *value_;
  }

 private:
  Singleton();
  ~Singleton();

  static void init()
  {
    value_ = new T();
    if (!detail::has_no_destroy<T>::value)
    {
      ::atexit(destroy);
    }
  }

  static void destroy()
  {
    typedef char T_must_be_complete_type[sizeof(T) == 0 ? -1 : 1];
    T_must_be_complete_type dummy; (void) dummy;

    delete value_;
    value_ = NULL;
  }

 private:
  static pthread_once_t ponce_;
  static T*             value_;
  
  DISALLOW_COPY_AND_ASSIGN(Singleton);
};

template<typename T>
pthread_once_t Singleton<T>::ponce_ = PTHREAD_ONCE_INIT;

template<typename T>
T* Singleton<T>::value_ = NULL;

} // namespace mymuduo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MUDO_SINGLETON_H_