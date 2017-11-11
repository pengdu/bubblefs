// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// tera/src/common/console/progress_bar.h

#ifndef BUBBLEFS_UTILS_BDCOM_PROGRESS_BAR_H_
#define BUBBLEFS_UTILS_BDCOM_PROGRESS_BAR_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string>

namespace bubblefs {
namespace mybdcom {

class ProgressBar {
public:
    enum DisplayMode {
        // brief mode, only a progress bar and easy to use
        // >>>>>>>>>>>------------------------------- 28%
        BRIEF,

        // add some enhanced config
        // >>>>>>>>>-------------------------- 28% 30M 20KB/s 00:00:05
        ENHANCED
    };

    ProgressBar(DisplayMode mode = BRIEF,
                uint64_t total_size = 100,
                uint32_t length = 80,
                std::string unit = "",
                char ch1 = '>',
                char ch2 = '-');
    ~ProgressBar();

    void Refresh(int32_t cur_size);
    void AddAndRefresh(int32_t size);
    void Done();

    int32_t GetPercent() {
        return (int32_t)(cur_size_ * 100 / total_size_);
    }

private:
    void FillFlushBufferBrief(int64_t cur_size);
    void FillFlushBufferEnhanced(int64_t cur_size);
    int32_t GetScreenWidth();
    int32_t GetTime();

private:
    DisplayMode mode_;
    int64_t total_size_;
    int64_t cur_size_;
    int32_t bar_length_;
    std::string unit_;
    char char_1_;
    char char_2_;

    time_t start_time_;
    time_t cur_time_;
    char *flush_buffer_;
};

}  // namespace mybdcom
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_PROGRESS_BAR_H_