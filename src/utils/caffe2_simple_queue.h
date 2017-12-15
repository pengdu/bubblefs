/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/utils/simple_queue.h

#ifndef BUBBLEFS_UTILS_CAFFE2_SIMPLE_QUEUE_H_
#define BUBBLEFS_UTILS_CAFFE2_SIMPLE_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include "platform/base_error.h"
#include "platform/macros.h"

namespace bubblefs {
namespace mycaffe2 {
  
// This queue reduces the chance to allocate memory for deque
template <typename T, int N>
class SmallQueue {
public:
    SmallQueue() : _begin(0), _size(0), _full(NULL) {}
    
    void push(const T& val) {
        if (_full != NULL && !_full->empty()) {
            _full->push_back(val);
        } else if (_size < N) {
            int tail = _begin + _size;
            if (tail >= N) {
                tail -= N;
            }
            _c[tail] = val;
            ++_size;
        } else {
            if (_full == NULL) {
                _full = new std::deque<T>;
            }
            _full->push_back(val);
        }
    }
    bool pop(T* val) {
        if (_size > 0) {
            *val = _c[_begin];
            ++_begin;
            if (_begin >= N) {
                _begin -= N;
            }
            --_size;
            return true;
        } else if (_full && !_full->empty()) {
            *val = _full->front();
            _full->pop_front();
            return true;
        }
        return false;
    }
    bool empty() const {
        return _size == 0 && (_full == NULL || _full->empty());
    }

    size_t size() const {
        return _size + (_full ? _full->size() : 0);
    }

    void clear() {
        _size = 0;
        _begin = 0;
        if (_full) {
            _full->clear();
        }
    }

    ~SmallQueue() {
        delete _full;
        _full = NULL;
    }
    
private:
    DISALLOW_COPY_AND_ASSIGN(SmallQueue);
    
    int _begin;
    int _size;
    T _c[N];
    std::deque<T>* _full;
};    
  
// This is a very simple queue that Yangqing wrote when bottlefeeding the baby,
// so don't take it seriously. What it does is a minimal thread-safe queue that
// allows me to run network as a DAG.
//
// A usual work pattern looks like this: one or multiple producers push jobs
// into this queue, and one or multiple workers pops jobs from this queue. If
// nothing is in the queue but NoMoreJobs() is not called yet, the pop calls
// will wait. If NoMoreJobs() has been called, pop calls will return false,
// which serves as a message to the workers that they should exit.
template <typename T>
class SimpleQueue {
 public:
  SimpleQueue() : no_more_jobs_(false) {}

  // Pops a value and writes it to the value pointer. If there is nothing in the
  // queue, this will wait till a value is inserted to the queue. If there are
  // no more jobs to pop, the function returns false. Otherwise, it returns
  // true.
  bool Pop(T* value) {
    std::unique_lock<std::mutex> mutex_lock(mutex_);
    while (queue_.size() == 0 && !no_more_jobs_) cv_.wait(mutex_lock);
    if (queue_.size() == 0 && no_more_jobs_) return false;
    *value = queue_.front();
    queue_.pop();
    return true;
  }

  int size() {
    std::unique_lock<std::mutex> mutex_lock(mutex_);
    return queue_.size();
  }

  // Push pushes a value to the queue.
  bool Push(const T& value) {
    {
      std::lock_guard<std::mutex> mutex_lock(mutex_);
      if (no_more_jobs_) {
        PRINTF_ERROR("Cannot push to a closed queue.\n");
        return false;
      }
      queue_.push(value);
    }
    cv_.notify_one();
    return true;
  }

  // NoMoreJobs() marks the close of this queue. It also notifies all waiting
  // Pop() calls so that they either check out remaining jobs, or return false.
  // After NoMoreJobs() is called, this queue is considered closed - no more
  // Push() functions are allowed, and once existing items are all checked out
  // by the Pop() functions, any more Pop() function will immediately return
  // false with nothing set to the value.
  void NoMoreJobs() {
    {
      std::lock_guard<std::mutex> mutex_lock(mutex_);
      no_more_jobs_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
  bool no_more_jobs_;
  // We do not allow copy constructors.
  SimpleQueue(const SimpleQueue& /*src*/) {}
};

}  // namespace mycaffe2
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE2_SIMPLE_QUEUE_H_