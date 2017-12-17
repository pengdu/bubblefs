// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/socket/ipv6_address_utils.h

#ifndef comm_in6_v4mapped_macros_h
#define comm_in6_v4mapped_macros_h

#include <netinet/in.h>

namespace bubblefs {
namespace mymars {

#ifndef __APPLE__
#define IN6ADDR_V4MAPPED_INIT {{{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00}}}
#endif

#define IN6ADDR_NAT64_INIT {{{ 0x00, 0x64, 0xff, 0x9b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }}}

const in6_addr in6addr_v4mapped_init  = IN6ADDR_V4MAPPED_INIT;
const in6_addr in6addr_nat64_init     = IN6ADDR_NAT64_INIT;

inline void IN6_SET_ADDR_V4MAPPED(in6_addr* a6, const in_addr* a4) {
    *a6 = in6addr_v4mapped_init;
    a6->s6_addr32[3] = a4->s_addr;
}

inline void IN6_SET_ADDR_NAT64(in6_addr* a6, const in_addr* a4) {
    *a6 = in6addr_nat64_init;
    a6->s6_addr32[3] = a4->s_addr;
}

inline bool IN6_IS_ADDR_NAT64(const in6_addr* a6) {
    return a6->s6_addr32[0] == htonl(0x0064ff9b);
}

} // namespace mymars
} // namespace bubblefs

#endif