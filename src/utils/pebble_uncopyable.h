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

// Pebble/src/common/uncopyable.h

#ifndef BUBBLEFS_UTILS_PEBBLE_UNCOPYABLE_H_
#define BUBBLEFS_UTILS_PEBBLE_UNCOPYABLE_H_

#include "platform/macros.h"

namespace bubblefs {
namespace mypebble {

class Uncopyable {
protected:
    Uncopyable() {}
    ~Uncopyable() {}

private:
    Uncopyable(const Uncopyable&);
    const Uncopyable& operator=(const Uncopyable&);
}; // class Uncopyable

} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_UNCOPYABLE_H_