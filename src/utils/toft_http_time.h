// Copyright 2011, The Toft Authors.
// Author: Yongsong Liu <lyscsu@gmail.com>

// toft/net/http/time.h

#ifndef BUBBLEFS_UTILS_TOFT_HTTP_TIME_H_
#define BUBBLEFS_UTILS_TOFT_HTTP_TIME_H_

#include <time.h>
#include <string>

namespace bubblefs {
namespace mytoft {

bool ParseHttpTime(const char* str, time_t* time);
inline bool ParseHttpTime(const std::string& str, time_t* time)
{
    return ParseHttpTime(str.c_str(), time);
}

size_t FormatHttpTime(time_t time, char* str, size_t str_length);
bool FormatHttpTime(time_t time, std::string* str);

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TOFT_HTTP_TIME_H_