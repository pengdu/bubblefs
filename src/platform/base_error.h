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

#include <assert.h>
#include <errno.h>
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
#define ENABLE_DEBUG 1

#define CHAR_NEW_LINE 10

#define STR_ERRORNO() (errno == 0 ? "None" : strerror(errno)) 

/// @brief 格式化输出log信息到buff
#define SPRINT_LOG_MESSAGE(buff, buff_len, fmt, ...) \
    snprintf((buff), (buff_len), "[%s:%d](%s) " fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    
/// @brief 记录最后错误信息，内部使用，要求buff名字为m_last_error
#define _LOG_LAST_ERROR(fmt, ...) \
    SPRINT_LOG_MESSAGE((m_last_error), (sizeof(m_last_error)), fmt, ##__VA_ARGS__)   
    
/// print utils    
#define PRINTF_INFO(fmt, ...) \
    fprintf(stdout, "INFO [%s:%d](%s) " fmt, \
            __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
    fflush(stdout)
    
#define PRINTF_WARN(fmt, ...) \
    fprintf(stderr, "WARN [%s:%d](%s) errno: %d %s, " fmt, \
            __FILE__, __LINE__, __FUNCTION__, errno, STR_ERRORNO(), ##__VA_ARGS__); \
    fflush(stderr)
    
#define PRINTF_ERROR(fmt, ...) \
    fprintf(stderr, "ERROR [%s:%d](%s) errno: %d %s, " fmt, \
            __FILE__, __LINE__, __FUNCTION__, errno, STR_ERRORNO(), ##__VA_ARGS__); \
    fflush(stderr)
    
#define PRINTF_ASSERT(fmt, ...) \
    fprintf(stderr, "ASSERT [%s:%d](%s) errno: %d %s, " fmt, \
            __FILE__, __LINE__, __FUNCTION__, errno, STR_ERRORNO(), ##__VA_ARGS__); \
    fflush(stderr)
    
#define PRINTF_TRACE(fmt, ...) \
    if (ENABLE_DEBUG) { \
      fprintf(stdout, "TRACE [%s:%d](%s) " fmt, \
              __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
      fflush(stdout); \
    }

#define PRINTF_TEST_DONE() \
    PRINTF_INFO("TEST DOWN \n") 
    
#define PRINTF_CHECK(c, fmt, ...) \
    if (!(c)) { \
      PRINTF_ASSERT("%s is False" fmt, #c, ##__VA_ARGS__); \
    }
    
#define PRINTF_CHECK_TRUE(c) \
    if (!(c)) { \
      PRINTF_ASSERT("%s is not TRUE \n", #c); \
    }
    
#define PRINTF_CHECK_FALSE(c) \
    if (c) { \
      PRINTF_ASSERT("%s is not FALSE \n", #c); \
    }
    
#define PRINTF_CHECK_EQ(c, val) \
    if ((c) != (val)) { \
      PRINTF_ASSERT("%s is not EQ %s \n", #c, #val); \
    }
    
#define PRINTF_CHECK_NE(c, val) \
    if ((c) == (val)) { \
      PRINTF_ASSERT("%s is not NE %s \n", #c, #val); \
    }    
    
#define PRINTF_CHECK_GE(c, val) \
    if ((c) < (val)) { \
      PRINTF_ASSERT("%s is not GE %s \n", #c, #val); \
    }
    
#define PRINTF_CHECK_GT(c, val) \
    if ((c) <= (val)) { \
      PRINTF_ASSERT("%s is not GT %s \n", #c, #val); \
    } 
    
#define PRINTF_CHECK_LE(c, val) \
    if ((c) > (val)) { \
      PRINTF_ASSERT("%s is not LE %s \n", #c, #val); \
    } 
    
#define PRINTF_CHECK_LT(c, val) \
    if ((c) >= (val)) { \
      PRINTF_ASSERT("%s is not LT %s \n", #c, #val); \
    } 

/// panic utils
#define PANIC(fmt, ...) \
    PRINTF_ERROR(fmt, ##__VA_ARGS__); \
    PRINTF_ERROR("\n Panic \n"); \
    abort()
  
#define PANIC_ENFORCE(c, fmt, ...) \
    if (!(c)) { \
      PRINTF_ERROR("%s is False" fmt, #c, ##__VA_ARGS__); \
      PRINTF_ERROR("\n Panic \n"); \
      abort(); \
    }
    
#define PANIC_ENFORCE_EQ(c, val) \
    if ((c) != (val)) { \
      PANIC("%s is not EQ %s \n", #c, #val); \
    }
    
#define PANIC_ENFORCE_NE(c, val) \
    if ((c) == (val)) { \
      PANIC("%s is not NE %s \n", #c, #val); \
    }
    
#define PANIC_ENFORCE_GE(c, val) \
    if ((c) < (val)) { \
      PANIC("%s is not GE %s \n", #c, #val); \
    }
    
#define PANIC_ENFORCE_GT(c, val) \
    if ((c) <= (val)) { \
      PANIC("%s is not GT %s \n", #c, #val); \
    }     
    
#define PANIC_ENFORCE_LE(c, val) \
    if ((c) > (val)) { \
      PANIC("%s is not LE %s \n", #c, #val); \
    } 
    
#define PANIC_ENFORCE_LT(c, val) \
    if ((c) >= (val)) { \
      PANIC("%s is not LT %s \n", #c, #val); \
    } 

#define EXIT_FAIL(fmt, ...) \
    PRINTF_ERROR(fmt, ##__VA_ARGS__); \
    PRINTF_ERROR("\n EXIT_FAILURE \n"); \
    exit(EXIT_FAILURE)
    
#endif // BUBBLEFS_PLATFORM_BASE_ERROR_H_