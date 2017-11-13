/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

// Pebble/src/framework/dr/common/dr_define.h


#ifndef BUBBLEFS_UTILS_PEBBLE_DEFINE_H_
#define BUBBLEFS_UTILS_PEBBLE_DEFINE_H_

#include <stdint.h>
#include <string>
#include <vector>

namespace bubblefs {
namespace mypebble {
namespace dr {
namespace protocol {

/**
 * Enumerated definition of the message types that the Thrift protocol
 * supports.
 */
enum TMessageType {
    T_CALL       = 1,
    T_REPLY      = 2,
    T_EXCEPTION  = 3,
    T_ONEWAY     = 4,
};
} // namespace protocol

} // namespace dr
} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_DEFINE_H_