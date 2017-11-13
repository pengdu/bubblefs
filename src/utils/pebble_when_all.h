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

// Pebble/src/framework/when_all.cpp

// Pebble/src/framework/when_all.h

#ifndef BUBBLEFS_UTILS_PEBBLE_WHEN_ALL_H_
#define BUBBLEFS_UTILS_PEBBLE_WHEN_ALL_H_

#include <vector>
#include "platform/pebble_logging.h"
#include "platform/types.h"

namespace bubblefs {
namespace mypebble {

typedef std::function<int32_t(uint32_t*, uint32_t*)> Call;

template <typename ServiceMethod,
          typename Stub,
          typename Response,
          typename RetCode>
Call AddCall(const ServiceMethod& service_method,
             const Stub& stub,
             Response response,
             RetCode ret_code) {
    return std::bind(service_method,
                     stub,
                     ret_code,
                     response,
                     std::placeholders::_1,
                     std::placeholders::_2);
}

template <typename ServiceMethod,
          typename Stub,
          typename Response,
          typename RetCode,
          typename Request>
Call AddCall(const ServiceMethod& service_method,
             const Stub& stub,
             Response response,
             RetCode ret_code,
             const Request& request) {
    return std::bind(service_method,
                     stub,
                     std::ref(request),
                     ret_code,
                     response,
                     std::placeholders::_1,
                     std::placeholders::_2);
}

template <typename ServiceMethod,
          typename Stub,
          typename Response,
          typename RetCode,
          typename RequestArg1,
          typename RequestArg2>
Call AddCall(const ServiceMethod& service_method,
             const Stub& stub,
             Response response,
             RetCode ret_code,
             const RequestArg1& request_arg1,
             const RequestArg2& request_arg2) {
    return std::bind(service_method,
                     stub,
                     std::ref(request_arg1),
                     std::ref(request_arg2),
                     ret_code,
                     response,
                     std::placeholders::_1,
                     std::placeholders::_2);
}

template <typename ServiceMethod,
          typename Stub,
          typename Response,
          typename RetCode,
          typename RequestArg1,
          typename RequestArg2,
          typename RequestArg3>
Call AddCall(const ServiceMethod& service_method,
             const Stub& stub,
             Response response,
             RetCode ret_code,
             const RequestArg1& request_arg1,
             const RequestArg2& request_arg2,
             const RequestArg3& request_arg3) {
    return std::bind(service_method,
                     stub,
                     std::ref(request_arg1),
                     std::ref(request_arg2),
                     std::ref(request_arg3),
                     ret_code,
                     response,
                     std::placeholders::_1,
                     std::placeholders::_2);
}

// WhenAll macro
#define WhenAllInit(num) \
    uint32_t num_parallel = num; \
    uint32_t num_called = num_parallel;

#define WhenAllCall(num) \
    f##num(&num_called, &num_parallel);

#define WhenAllCheck() \
    PEBBLE_PLOG_IF_ERROR(num_called != 0, \
        "num_parallel: %u, num_called: %u", num_parallel, num_called);

void WhenAll(const std::vector<Call>& f_list);

// f1
void WhenAll(const Call& f1);

// f2
void WhenAll(const Call& f1,
             const Call& f2);

// f3
void WhenAll(const Call& f1,
             const Call& f2,
             const Call& f3);

} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_WHEN_ALL_H_