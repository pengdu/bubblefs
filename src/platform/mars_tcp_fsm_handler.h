// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/socket/tcp_fsm_handler.h

#ifndef BUBBLEFS_PLATFORM_MARS_TCP_FSM_HANDLER_H_
#define BUBBLEFS_PLATFORM_MARS_TCP_FSM_HANDLER_H_

#include <limits.h>
#include <algorithm>
#include "platform/macros.h"
#include "platform/mars_socketbreaker.h"
#include "platform/mars_socketselect.h"

namespace bubblefs {
namespace mymars {

template<class InputIterator>
bool TcpFSMHandler(InputIterator _first, InputIterator _last, SocketBreaker& _breaker, bool _select_anyway)
{
    //xverbose_function();
    //xgroup2_define(group);

    SocketSelect sel(_breaker, true);
    sel.PreSelect();

    bool have_runing_status = false;
    int timeout = INT_MAX;

    for (InputIterator it = _first; it != _last; ++it)
    {
        if (!(*it)->IsEndStatus()) have_runing_status = true;

        timeout = std::min(timeout, (*it)->Timeout());
        (*it)->PreSelect(sel, group);
    }

    if (!have_runing_status && !_select_anyway)
    {
        //xinfo2(TSF"all end status") >> group;
        return false;
    }

    int ret = 0;
    if (INT_MAX == timeout)
    {
        ret = sel.Select();
    } else {
        timeout = std::max(0, timeout);
        ret = sel.Select(timeout);
    }

    // select error
    if (ret < 0) { return false;}
    // user break
    if (sel.IsException()) { return false; }
    if (sel.IsBreak()) { return false; }

    for (InputIterator it = _first; it != _last; ++it)
    {
        (*it)->AfterSelect(sel, group);
    }

    return true;
}

template<class InputIterator>
void TcpFSMHandlerRunloop(InputIterator _first, InputIterator _last, SocketBreaker& _breaker, bool _select_anyway)
{
    //xinfo_function();

     while (TcpFSMHandler(_first, _last, _breaker, _select_anyway)) {}
}

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_TCP_FSM_HANDLER_H_