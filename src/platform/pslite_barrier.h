
// parameter_server/src/util/barrier.h

#ifndef BUBBLEFS_PLATFORM_PSLITE_BARRIER_H_
#define BUBBLEFS_PLATFORM_PSLITE_BARRIER_H_

#include <condition_variable>
#include <mutex>

#include "platform/base_error.h"
#include "platform/macros.h"

namespace bubblefs {
namespace mypslite {

class Barrier {
 public:
  explicit Barrier(int num_threads)
      : num_to_block_(num_threads), num_to_exit_(num_threads) {}

  // return true if this is the last thread
  bool Block() {
    std::unique_lock<std::mutex> l(mu_);
    num_to_block_--;
    PANIC_ENFORCE_GE(num_to_block_, 0);

    if (num_to_block_ > 0) {
      while (num_to_block_ > 0) cv_.wait(l);
    } else {
      cv_.notify_all();
    }

    num_to_exit_--;
    PANIC_ENFORCE_GE(num_to_exit_, 0);
    return (num_to_exit_ == 0);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Barrier);
  std::mutex mu_;
  std::condition_variable cv_;
  int num_to_block_;
  int num_to_exit_;
};

}  // namespace mypslite
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_PSLITE_BARRIER_H_