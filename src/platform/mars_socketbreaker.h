// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/unix/socket/socketbreaker.h

#ifndef BUBBLEFS_PLATFORM_MARS_SOCKETBREAKER_H_
#define BUBBLEFS_PLATFORM_MARS_SOCKETBREAKER_H_ 

#include "platform/mars_lock.h"

namespace bubblefs {
namespace mymars {

class SocketBreaker {
  public:
    SocketBreaker();
    ~SocketBreaker();

    bool IsCreateSuc() const;
    bool ReCreate();
    void Close();

    bool Break();
    bool Clear();

    bool IsBreak() const;
    int  BreakerFD() const;

  private:
    SocketBreaker(const SocketBreaker&);
    SocketBreaker& operator=(const SocketBreaker&);

  private:
    int   pipes_[2];
    bool  create_success_;
    bool  broken_;
    Mutex mutex_;
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_SOCKETBREAKER_H_