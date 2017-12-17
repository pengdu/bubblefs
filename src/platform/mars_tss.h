// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/unix/thread/tss.h

#ifndef BUBBLEFS_PLATFORM_MARS_TSS_H_
#define BUBBLEFS_PLATFORM_MARS_TSS_H_

#include <pthread.h>

namespace bubblefs {
namespace mymars {

typedef void (*cleanup_route)(void*);

class Tss {
  public:
    explicit Tss(cleanup_route cleanup) {
        pthread_key_create(&_key, cleanup);
    }

    ~Tss() {
        pthread_key_delete(_key);
    }

    void* get() const {
        return pthread_getspecific(_key);
    }

    void set(void* value) {
        pthread_setspecific(_key, value);
    }

  private:
    Tss(const Tss&);
    Tss& operator =(const Tss&);

  private:
    pthread_key_t _key;
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_TSS_H_