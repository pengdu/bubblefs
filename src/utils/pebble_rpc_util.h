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

// Pebble/src/framework/rpc_util.inh

#ifndef BUBBLEFS_UTILS_PEBBLE_RPC_UTIL_H_
#define BUBBLEFS_UTILS_PEBBLE_RPC_UTIL_H_

#include "utils/pebble_rpc.h"

namespace bubblefs {
namespace mypebble {

class CoroutineSchedule;

/// @brief RPC Util错误码定义
typedef enum {
    kRPC_UTIL_ERROR_BASE                        = kRPC_RPC_UTIL_ERROR_BASE,
    kRPC_UTIL_CO_SCHEDULE_IS_NULL               = kRPC_UTIL_ERROR_BASE - 1, // 协程控制器为空
    kRPC_UTIL_NOT_IN_COROUTINE                  = kRPC_UTIL_ERROR_BASE - 2, // 发送请求不在协程中
    kRPC_UTIL_ALREADY_IN_COROUTINE              = kRPC_UTIL_ERROR_BASE - 3, // 请求处理已经在协程中
    kRPC_UTIL_INVALID_NUM_PARALLEL_IN_COROUTINE = kRPC_UTIL_ERROR_BASE - 4, // 无效的并行调用次数
} CoroutineRpcErrorCode;

class RpcUtilErrorStringRegister {
public:
    static void RegisterErrorString() {
        SetErrorString(kRPC_UTIL_CO_SCHEDULE_IS_NULL, "co schedule is null");
        SetErrorString(kRPC_UTIL_NOT_IN_COROUTINE, "not in coroutine");
        SetErrorString(kRPC_UTIL_ALREADY_IN_COROUTINE, "already in coroutine");
    }
};


/// @brief RPC实现通用工具部分，如提供基于协程的同步调用
class RpcUtil {
public:
    RpcUtil(IRpc* rpc, CoroutineSchedule* coroutine_schedule);
    ~RpcUtil();

    /// @brief 同步发送，在协程中执行
    int32_t SendRequestSync(int64_t handle,
                    const RpcHead& rpc_head,
                    const uint8_t* buff,
                    uint32_t buff_len,
                    const OnRpcResponse& on_rsp,
                    int32_t timeout_ms);

    /// @brief 并行发送，在协程中执行
    void SendRequestParallel(int64_t handle,
                             const RpcHead& rpc_head,
                             const uint8_t* buff,
                             uint32_t buff_len,
                             const OnRpcResponse& on_rsp,
                             int32_t timeout_ms,
                             int32_t* ret_code,
                             uint32_t* num_called,
                             uint32_t* num_parallel);


    int32_t ProcessRequest(int64_t handle, const RpcHead& rpc_head,
        const uint8_t* buff, uint32_t buff_len);

private:
    void SendRequestInCoroutine(int64_t handle,
                    const RpcHead& rpc_head,
                    const uint8_t* buff,
                    uint32_t buff_len,
                    const OnRpcResponse& on_rsp,
                    uint32_t timeout_ms,
                    int32_t* ret);

    void SendRequestParallelInCoroutine(int64_t handle,
                                        const RpcHead& rpc_head,
                                        const uint8_t* buff,
                                        uint32_t buff_len,
                                        const OnRpcResponse& on_rsp,
                                        uint32_t timeout_ms,
                                        int32_t* ret_code,
                                        uint32_t* num_called,
                                        uint32_t* num_parallel);

    int32_t ProcessRequestInCoroutine(int64_t handle, const RpcHead& rpc_head,
        const uint8_t* buff, uint32_t buff_len);

    int32_t OnResponse(int32_t ret,
                       const uint8_t* buff,
                       uint32_t buff_len,
                       int64_t co_id);

    int32_t OnResponseParallel(int32_t ret,
                               const uint8_t* buff,
                               uint32_t buff_len,
                               int32_t* ret_code,
                               const OnRpcResponse& on_rsp,
                               int64_t co_id);
private:
    /// @brief 异步结果结构定义
    struct AsyncResult {
        AsyncResult() {
            _ret      = 0;
            _buff     = NULL;
            _buff_len = 0;
            _ret_code = 0;
        }

        int32_t         _ret;
        const uint8_t*  _buff;
        uint32_t        _buff_len;
        int32_t*        _ret_code;
        OnRpcResponse   _on_rsp;
    };

    IRpc* m_rpc;
    CoroutineSchedule* m_coroutine_schedule;
    AsyncResult m_result;
};

} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_RPC_UTIL_H_