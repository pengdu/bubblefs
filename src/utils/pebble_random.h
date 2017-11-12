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

// Pebble/src/common/random.h

#ifndef BUBBLEFS_UTILS_PEBBLE_RANDOM_H_
#define BUBBLEFS_UTILS_PEBBLE_RANDOM_H_

#include <stdint.h>
#include <stdlib.h>
#include "utils/pebble_uncopyable.h"

namespace bubblefs {
namespace mypebble {

class TrueRandom {
    DISALLOW_COPY_AND_ASSIGN(TrueRandom);

public:
    TrueRandom();
    ~TrueRandom();

    uint32_t NextUInt32();

    uint32_t NextUInt32(uint32_t max_random);

    bool NextBytes(void* buffer, size_t size);

private:
    int32_t m_fd;
}; // class TrueRandom

} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_RANDOM_H_