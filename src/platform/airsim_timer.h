
// AirSim/AirLib/include/common/common_utils/Timer.hpp

#ifndef BUBBLEFS_PLATFORM_AIRSIM_TIMER_H_
#define BUBBLEFS_PLATFORM_AIRSIM_TIMER_H_

#include <chrono>

namespace bubblefs {
namespace myairsim {
namespace common_utils {
  
    class Timer {
    public:
        Timer() 
        {
            started_ = false;
        }
        void start() 
        {
            started_ = true;
            start_ = now();
        }
        void stop() 
        {
            started_ = false;
            end_ = now();
        }
        double seconds() 
        {
            auto diff = static_cast<double>(end() - start_);
            return  diff / 1000000.0;
        }
        double milliseconds() 
        {
            return static_cast<double>(end() - start_) / 1000.0;
        }
        double microseconds() 
        {
            return static_cast<double>(end() - start_);
        }
        bool started() 
        {
            return started_;
        }
    private:
        int64_t now() {
            return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        }
        int64_t end() {
            if (started_) {
                // not stopped yet, so return "elapsed time so far".
                end_ = now();
            }
            return end_;
        }
        int64_t start_;
        int64_t end_;
        bool started_;
    };
    
} // namespace common_utils
} // namespace myairsim
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_AIRSIM_TIMER_H_