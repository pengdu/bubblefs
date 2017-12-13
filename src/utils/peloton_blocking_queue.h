
#ifndef BUBBLEFS_UTILS_PELOTON_BLOCKING_QUEUE_H_
#define BUBBLEFS_UTILS_PELOTON_BLOCKING_QUEUE_H_

#include <deque>
#include <mutex>
#include <utility>

namespace bubblefs {
namespace mypeloton {

//===--------------------------------------------------------------------===//
// Blocking Queue -- Supports multiple consumers and multiple producers.
//===--------------------------------------------------------------------===//

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue(const size_t &size) : elements_(size), numElements_(size) {}

  BlockingQueue(const BlockingQueue&) = delete;             // disable copying
  BlockingQueue& operator=(const BlockingQueue&) = delete;  // disable assignment

  // Enqueues one item, allocating extra space if necessary
  void Enqueue(T& el) {
    std::unique_lock<std::mutex> lock(queueLock_);
    elements_.emplace_back(el);
    numElements_++;
  }

  void Enqueue(const T& el) {
    std::unique_lock<std::mutex> lock(queueLock_);
    elements_.emplace_back(el);
    numElements_++;
  }

  // Dequeues one item, returning true if an item was found
  // or false if the queue appeared empty
  bool Dequeue(T& el) {
    std::unique_lock<std::mutex> lock(queueLock_);
    if (0 >= numElements_)
      return false;
    // Becuase of the previous statement, the right swap() can be found
    // via argument-dependent lookup (ADL).
    std::swap(elements_.front(), el);
    elements_.pop_front();
    numElements_--;
    return true;
  }
  
  bool IsEmpty() {
    std::unique_lock<std::mutex> lock(queueLock_);
    return 0 == numElements_;
  }

  int Size() {
    std::unique_lock<std::mutex> lock(queueLock_);
    return numElements_;
  }

 private:

  std::deque<T> elements_;
  int numElements_;
  std::mutex queueLock_;
};

}  // namespace peloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_BLOCKING_QUEUE_H_