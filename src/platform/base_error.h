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

// Pebble/src/common/error.h

#ifndef BUBBLEFS_PLATFORM_BASE_ERROR_H_
#define BUBBLEFS_PLATFORM_BASE_ERROR_H_

#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * 错误输出原则:
 *  1) common库，功能聚焦api明确，返回错误码，记录错误信息供用户获取，不输出日志
 *  2) framework，关键流程错误必须输出到日志，注意避免重复输出
 *  3) 对于暴露给用户的接口，返回错误应该是最根本错误原因，避免返回中间错误信息
 */

// '\n'
#define CHAR_NEW_LINE 010

#define STR_ERRORNO() (errno == 0 ? "None" : strerror(errno)) 

/// @brief 格式化输出log信息到buff
#define LOG_MESSAGE(buff, buff_len, fmt, ...) \
    snprintf((buff), (buff_len), "[%s:%d](%s)" fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    
/// @brief 记录最后错误信息，内部使用，要求buff名字为m_last_error
#define _LOG_LAST_ERROR(fmt, ...) \
    LOG_MESSAGE((m_last_error), (sizeof(m_last_error)), fmt, ##__VA_ARGS__)   
    
#define PRINTF_INFO(fmt, ...) \
    fprintf(stderr, "INFO [%s:%d](%s) " fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    
#define PRINTF_WARN(fmt, ...) \
    fprintf(stderr, "WARN [%s:%d](%s) errno: %s," fmt, \
    __FILE__, __LINE__, __FUNCTION__, STR_ERRORNO(), ##__VA_ARGS__)
    
#define PRINTF_ERROR(fmt, ...) \
    fprintf(stderr, "ERROR [%s:%d](%s) errno: %s," fmt, \
    __FILE__, __LINE__, __FUNCTION__, STR_ERRORNO(), ##__VA_ARGS__)

#define PRINTF_CHECK(condition, msg) \
    if (!condition) { \
      std::string str_msg(msg); \
      PRINTF_ERROR("%s%c", str_msg.c_str(), CHAR_NEW_LINE); \
    }
    
#define PRINTF_CHECK_EQ(condition, val) \
    if (val != condition) { \
      PRINTF_ERROR("%s is not EQ %s%c", #condition, #val, CHAR_NEW_LINE); \
    }
    
#define PRINTF_CHECK_GT(condition, val) \
    if (val >= condition) { \
      PRINTF_ERROR("%s is not GT %s%c", #condition, #val, CHAR_NEW_LINE); \
    } 
    
#define PANIC(fmt, ...) \
    PRINTF_ERROR(fmt, ##__VA_ARGS__); \
    PRINTF_ERROR("%cPanic%c", CHAR_NEW_LINE, CHAR_NEW_LINE); \
    abort()
  
#define PANIC_ENFORCE(condition, msg) \
    if (!condition) { \
      std::string str_msg(msg); \
      PANIC("%s%c", str_msg.c_str(), CHAR_NEW_LINE); \
    }
    
#define PANIC_ENFORCE_EQ(condition, val) \
    if (val != condition) { \
      PANIC("%s is not EQ %s%c", #condition, #val, CHAR_NEW_LINE); \
    }
    
#define PANIC_ENFORCE_GT(condition, val) \
    if (val >= condition) { \
      PANIC("%s is not GT %s%c", #condition, #val, CHAR_NEW_LINE); \
    }    

#define EXIT_FAIL(fmt, ...) \
    PRINTF_ERROR(fmt, ##__VA_ARGS__); \
    PRINTF_ERROR("%cExit%c", CHAR_NEW_LINE, CHAR_NEW_LINE); \
    exit(EXIT_FAILURE)
    
namespace bubblefs {
 
/// @brief 每个模块错误码BASE定义，模块错误码取值为XXX_BASE - N
enum ERROR_CODE_BASE {
    NO_ERROR                    = 0,
    RPC_ERROR_CODE_BASE         = -1000,
    SESSION_ERROR_CODE_BASE     = -2000,
    MESSAGE_ERROR_CODE_BASE     = -3000,
    NAMING_ERROR_CODE_BASE      = -4000,
    ROUTER_ERROR_CODE_BASE      = -5000,
    TIMER_ERROR_CODE_BASE       = -6000,
    PIPE_ERROR_CODE_BASE        = -7000,
    COROUTINE_ERROR_CODE_BASE   = -8000,
    CHANNEL_ERROR_CODE_BASE     = -9000,
    PROCESSOR_ERROR_CODE_BASE   = -10000,
    USER_ERROR_CODE_BASE        = -100000,
};  

/// @brief 获取错误描述
const char* GetErrorString(int32_t error_code);

/// @brief 设置错误描述
void SetErrorString(int32_t error_code, const char* error_string);
  
} // namespace bubblefs
    
#endif // BUBBLEFS_PLATFORM_BASE_ERROR_H_