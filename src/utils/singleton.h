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

// android/platform_system_core/libutils/include/utils/Singleton.h

#ifndef BUBBLEFS_UTILS_SINGLETON_H
#define BUBBLEFS_UTILS_SINGLETON_H

#include <sys/types.h>
#include <stdlib.h>
#include "platform/mutexlock.h"

namespace bubblefs {
  
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
        MutexLock _l(sLock);
        TYPE* instance = sInstance;
        if (instance == 0) {
            instance = new TYPE();
            sInstance = instance;
        }
        return *instance;
    }

    static bool hasInstance() {
        MutexLock _l(sLock);
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

#define BUBBLEFS_SINGLETON_STATIC_INSTANCE(TYPE)                 \
    template<> ::bubblefs::port::Mutex  \
        (::bubblefs::Singleton< TYPE >::sLock)();  \
    template<> TYPE* ::bubblefs::Singleton< TYPE >::sInstance(0);  /* NOLINT */ \
    template class ::bubblefs::Singleton< TYPE >;
  
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SINGLETON_H