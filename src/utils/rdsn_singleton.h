/*
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

// android/platform_system_core/libutils/include/utils/Singleton.h
// rdsn/include/dsn/utility/singleton.h

#ifndef BUBBLEFS_UTILS_RDSN_SINGLETON_H_
#define BUBBLEFS_UTILS_RDSN_SINGLETON_H_

#include <mutex>
#include <atomic>
#include "platform/mutexlock.h"

namespace bubblefs {
  
namespace myandroid {
  
// Singleton<TYPE> may be used in multiple libraries, only one of which should
// define the static member variables using BUBBLEFS_SINGLETON_STATIC_INSTANCE.
// Turn off -Wundefined-var-template so other users don't get:
// instantiation of variable 'bubblefs::Singleton<TYPE>::sLock' required here,
// but no definition is available
 
template <typename TYPE>
class Singleton
{
public:
    static TYPE& getInstance() {
        MutexLock _l(&sLock);
        TYPE* instance = sInstance;
        if (instance == 0) {
            instance = new TYPE();
            sInstance = instance;
        }
        return *instance;
    }

    static bool hasInstance() {
        MutexLock _l(&sLock);
        return sInstance != 0;
    }
    
protected:
    ~Singleton() { }
    Singleton() { }

private:
    Singleton(const Singleton&);
    Singleton& operator = (const Singleton&);
    static port::Mutex sLock;
    static TYPE* sInstance;
};

/*
 * use BUBBLEFS_SINGLETON_STATIC_INSTANCE(TYPE) in your implementation file
 * (eg: <TYPE>.cpp) to create the static instance of Singleton<>'s attributes,
 * and avoid to have a copy of them in each compilation units Singleton<TYPE>
 * is used.
 * NOTE: we use a version of Mutex ctor that takes a parameter, because
 * for some unknown reason using the default ctor doesn't emit the variable!
 */

#define MYANDROID_SINGLETON_STATIC_INSTANCE(TYPE)                 \
    template<> ::bubblefs::port::Mutex  \
        (::bubblefs::myandroid::Singleton< TYPE >::sLock)();  \
    template<> TYPE* ::bubblefs::myandroid::Singleton< TYPE >::sInstance(0);  /* NOLINT */ \
    template class ::bubblefs::myandroid::Singleton< TYPE >;

}  // namespace myandroid

namespace rdsn {
namespace myrdsn {
  
template <typename T>
class Singleton
{
public:
    Singleton() {}

    static T &instance()
    {
        if (nullptr == _instance) {
            // lock
            while (0 != _l.exchange(1, std::memory_order_acquire)) {
                while (_l.load(std::memory_order_consume) == 1) {
                }
            }

            // re-check and assign
            if (nullptr == _instance) {
                auto tmp = new T();
                std::atomic_thread_fence(std::memory_order::memory_order_seq_cst);
                _instance = tmp;
            }

            // unlock
            _l.store(0, std::memory_order_release);
        }
        return *_instance;
    }

    static T &fast_instance() { return *_instance; }

    static bool is_instance_created() { return nullptr != _instance; }

protected:
    static T *_instance;
    static std::atomic<int> _l;

private:
    Singleton(const Singleton &);
    Singleton &operator=(const Singleton &);
};

// ----- inline implementations -------------------------------------------------------------------

template <typename T>
T *Singleton<T>::_instance = 0;
template <typename T>
std::atomic<int> Singleton<T>::_l(0);
  
} // namespace utils
} // namespace myrdsn

} // namespace bubblefs

#endif // BUBBLEFS_UTILS_RDSN_SINGLETON_H_