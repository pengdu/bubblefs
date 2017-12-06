// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// AirSim/AirLib/include/common/common_utils/AsyncTasker.hpp

#ifndef BUBBLEFS_UTILS_AIRSIM_ASYNC_TASKER_H_
#define BUBBLEFS_UTILS_AIRSIM_ASYNC_TASKER_H_

#include <functional>
#include "platform/macros.h"
#include "utils/airsim_ctpl_stl.h"

namespace bubblefs {
namespace myairsim {
  
class AsyncTasker {
public:
    AsyncTasker(unsigned int thread_count = 4)
        : threads_(thread_count), error_handler_([](std::exception e) {UNUSED_PARAM(e);})
    {
    }

    void setErrorHandler(std::function<void(std::exception&)> errorHandler) {
        error_handler_ = errorHandler;
    }

    void execute(std::function<void()> func, unsigned int iterations = 1)
    {
        if (iterations < 1)
            return;

        if (iterations == 1)
        {
            threads_.push([=](int i) {
                UNUSED_PARAM(i);
                try {
                    func();
                }
                catch (std::exception& e) {
                    error_handler_(e);
                };
            });
        }
        else {
            threads_.push([=](int i) {
                UNUSED_PARAM(i);
                try {
                    for (unsigned int itr = 0; itr < iterations; ++itr) {
                        func();
                    }
                }
                catch (std::exception& e) {
                    // if task failed we shouldn't try additional iterations.
                    error_handler_(e);
                };
            });
        }
    }

private:
    ctpl::thread_pool threads_;
    std::function<void(std::exception&)> error_handler_;
};

} // namespace myairsim
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_AIRSIM_ASYNC_TASKER_H_