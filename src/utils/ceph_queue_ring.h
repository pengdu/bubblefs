
// ceph/src/common/QueueRing.h

#ifndef BUBBLEFS_UTILS_CEPH_QUEUE_RING_H_
#define BUBBLEFS_UTILS_CEPH_QUEUE_RING_H_

#include <atomic>
#include <list>
#include <vector>
#include "platform/macros.h"
#include "platform/mutexlock.h"

namespace bubblefs {
namespace myceph {
  
template <class T>
class QueueRing {
  struct QueueBucket {
    port::Mutex lock;
    port::CondVar cond;
    typename std::list<T> entries;

    QueueBucket() : lock(), cond(&lock) {}
    QueueBucket(const QueueBucket& rhs) : lock(), cond(&lock) {
      entries = rhs.entries;
    }

    void enqueue(const T& entry) {
      lock.Lock();
      if (entries.empty()) {
        cond.Signal();
      }
      entries.push_back(entry);
      lock.Unlock();
    }

    void dequeue(T *entry) {
      lock.Lock();
      if (entries.empty()) {
        cond.Wait();
      };
      assert(!entries.empty());
      *entry = entries.front();
      entries.pop_front();
      lock.Unlock();
    };
  };

  std::vector<QueueBucket> buckets;
  int num_buckets;

  std::atomic<int64_t> cur_read_bucket = { 0 };
  std::atomic<int64_t> cur_write_bucket = { 0 };

public:
  QueueRing(int n) : buckets(n), num_buckets(n) {
  }

  void enqueue(const T& entry) {
    buckets[++cur_write_bucket % num_buckets].enqueue(entry);
  };

  void dequeue(T *entry) {
    buckets[++cur_read_bucket % num_buckets].dequeue(entry);
  }
};  
  
} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CEPH_QUEUE_RING_H_