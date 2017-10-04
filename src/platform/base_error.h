
#ifndef BUBBLEFS_PLATFORM_BASE_ERROR_H_
#define BUBBLEFS_PLATFORM_BASE_ERROR_H_

#include <error.h>
#include <stdio.h>

/**
 * 错误输出原则:
 *  1) common库，功能聚焦api明确，返回错误码，记录错误信息供用户获取，不输出日志
 *  2) framework，关键流程错误必须输出到日志，注意避免重复输出
 *  3) 对于暴露给用户的接口，返回错误应该是最根本错误原因，避免返回中间错误信息
 */

/// @brief 格式化输出log信息到buff
#define LOG_MESSAGE(buff, buff_len, fmt, ...) \
    snprintf((buff), (buff_len), "[%s:%d](%s)" fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    
/// @brief 记录最后错误信息，内部使用，要求buff名字为m_last_error
#define _LOG_LAST_ERROR(fmt, ...) \
    LOG_MESSAGE((m_last_error), (sizeof(m_last_error)), fmt, ##__VA_ARGS__)
    
#define FPRINTF_INFO(fmt, ...) \
    fprintf(stdout, "INFO [%s:%d](%s)" fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)    
    
#define FPRINTF_ERR(fmt, ...) \
    fprintf(stderr, "ERROR [%s:%d](%s)" fmt, \
    __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    
#endif // BUBBLEFS_PLATFORM_BASE_ERROR_H_