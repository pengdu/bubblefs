// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// AirSim/AirLib/include/common/common_utils/ScheduledExecutor.hpp

#ifndef commn_utils_ScheduledExecutor_hpp
#define commn_utils_ScheduledExecutor_hpp

#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include <system_error>
#include <mutex>
#include <cstdint>

namespace bubblefs {
namespace myairsim {
namespace common_utils {

class ScheduledExecutor {
public:
    ScheduledExecutor()
    {}
    ScheduledExecutor(const std::function<bool(uint64_t)>& callback, uint64_t period_nanos)
    {
        initialize(callback, period_nanos);
    }
    ~ScheduledExecutor()
    {
        stop();
    }
    void initialize(const std::function<bool(uint64_t)>& callback, uint64_t period_nanos)
    {
        callback_ = callback;
        period_nanos_ = period_nanos;
        started_ = false;
    }
    
    void cleanupThread(std::thread& th)
    {
        if (th.joinable()) {
            th.detach();
        }
    }

    void start()
    {
        started_ = true;
        sleep_time_avg_ = 0;
        period_count_ = 0;
        cleanupThread(th_);
        th_ = std::thread(&ScheduledExecutor::executorLoop, this);
    }

    void stop()
    {
        if (started_) {
            started_ = false;
            try {
                if (th_.joinable()) {
                    th_.join();
                }
            }
            catch(const std::system_error& /* e */)
            { }
        }
    }

    bool isRunning()
    {
        return started_;
    }

    double getSleepTimeAvg()
    {
        //TODO: make this function thread safe by using atomic types
        //right now this is not implemented for performance and that
        //return of this function is purely informational/debugging purposes
        return sleep_time_avg_;
    }

    uint64_t getPeriodCount()
    {
        return period_count_;
    }

    void lock()
    {
        mutex_.lock();
    }
    void unlock()
    {
        mutex_.unlock();
    }

private:
    typedef std::chrono::high_resolution_clock clock;
    typedef uint64_t TTimePoint;
    typedef uint64_t TTimeDelta;
    template <typename T>
    using duration = std::chrono::duration<T>;

    static TTimePoint nanos()
    {
        return clock::now().time_since_epoch().count();
    }

    static void sleep_for(TTimePoint delay_nanos)
    {
        /*
        This is spin loop implementation which may be suitable for sub-millisecond resolution.
        //TODO: investigate below alternatives
        On Windows we can use multimedia timers however this requires including entire Win32 header.
        On Linux we can use nanosleep however below 2ms delays in real-time scheduler settings this 
        probbaly does spin loop anyway.
        */

        if (delay_nanos >= 5000000LL) { //put thread to sleep
            std::this_thread::sleep_for(std::chrono::duration<double>(delay_nanos / 1.0E9));
        }
        else { //for more precise timing, do spinning
            auto start = nanos();
            while ((nanos() - start) < delay_nanos) {
                std::this_thread::yield();
                //std::this_thread::sleep_for(std::chrono::duration<double>(0));
            }
        }
    }

    void executorLoop()
    {
        TTimePoint call_end = nanos();
        while (started_) {
            TTimePoint period_start = nanos();
            TTimeDelta since_last_call = period_start - call_end;
            
            //is this first loop?
            if (period_count_ > 0) {
                //when we are doing work, don't let other thread to cause contention
                std::lock_guard<std::mutex> locker(mutex_);

                bool result = callback_(since_last_call);
                if (!result) {
                    started_ = result;
                }
            }
            
            call_end = nanos();

            TTimeDelta elapsed_period = nanos() - period_start;
            //prevent underflow: https://github.com/Microsoft/AirSim/issues/617
            TTimeDelta delay_nanos = period_nanos_ > elapsed_period ? period_nanos_ - elapsed_period : 0;
            //moving average of how much we are sleeping
            sleep_time_avg_ = 0.25f * sleep_time_avg_ + 0.75f * delay_nanos;
            ++period_count_;
            if (delay_nanos > 0 && started_)
                sleep_for(delay_nanos);
        }
    }

private:
    uint64_t period_nanos_;
    std::thread th_;
    std::function<bool(uint64_t)> callback_;
    std::atomic_bool started_;

    double sleep_time_avg_;
    uint64_t period_count_;

    std::mutex mutex_;
};

} // namespace common_utils
} // namespace myairsim
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_AIRSIM_SCHEDULED_EXCECUTOR_H_