// Copyright (c) 2016, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/utils/fragment.h

#ifndef  BUBBLEFS_UTILS_RANGE_FRAGMENT_UTIL_H_
#define  BUBBLEFS_UTILS_RANGE_FRAGMENT_UTIL_H_

#include <list>
#include <string>

namespace bubblefs {
namespace core {

class RangeFragment {
public:
    // caller should use Lock to avoid data races
    // On success, return true. Otherwise, return false due to invalid argumetns
    bool AddToRange(const std::string& start, const std::string& end);

    bool IsCompleteRange() const;

    bool IsCoverRange(const std::string& start, const std::string& end) const;

    std::string DebugString() const;

private:
    std::list<std::pair<std::string, std::string> > range_;
};

} // namespace core
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_RANGE_FRAGMENT_UTIL_H_
