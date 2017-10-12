
#ifndef BUBBLEFS_UTILS_STATISTICS_H_
#define BUBBLEFS_UTILS_STATISTICS_H_

#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include "utils/status.h"

namespace bubblefs {
namespace core {  

struct HistogramData {
  double median;
  double percentile95;
  double percentile99;
  double average;
  double standard_deviation;
  // zero-initialize new members since old Statistics::histogramData()
  // implementations won't write them.
  double max = 0.0;
};

enum StatsLevel {
  // Collect all stats except time inside mutex lock AND time spent on
  // compression.
  kExceptDetailedTimers,
  // Collect all stats except the counters requiring to get time inside the
  // mutex lock.
  kExceptTimeForMutex,
  // Collect all stats, including measuring duration of mutex operations.
  // If getting time is expensive on the platform to run, it can
  // reduce scalability to more threads, especially for writes.
  kAll,
};
  
// Analyze the performance of a db
class Statistics {
 public:
  virtual ~Statistics() {}

  virtual uint64_t getTickerCount(uint32_t tickerType) const = 0;
  virtual void histogramData(uint32_t type,
                             HistogramData* const data) const = 0;
  virtual std::string getHistogramString(uint32_t type) const { return ""; }
  virtual void recordTick(uint32_t tickerType, uint64_t count = 0) = 0;
  virtual void setTickerCount(uint32_t tickerType, uint64_t count) = 0;
  virtual uint64_t getAndResetTickerCount(uint32_t tickerType) = 0;
  virtual void measureTime(uint32_t histogramType, uint64_t time) = 0;

  // Resets all ticker and histogram stats
  virtual Status Reset() {
    return Status::NotSupported("Not implemented");
  }

  // String representation of the statistic object.
  virtual std::string ToString() const {
    // Do nothing by default
    return std::string("ToString(): not implemented");
  }

  // Override this function to disable particular histogram collection
  virtual bool HistEnabledForType(uint32_t type) const = 0;

  StatsLevel stats_level_ = kExceptDetailedTimers;
};

// Create a concrete DBStatistics object
std::shared_ptr<Statistics> CreateDBStatistics();

} // namespace core  
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_STATISTICS_H_