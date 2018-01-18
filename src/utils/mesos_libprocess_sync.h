// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

// mesos/3rdparty/libprocess/include/process/mutex.hpp
// mesos/3rdparty/libprocess/include/process/rwlock.hpp
// mesos/3rdparty/libprocess/include/process/latch.hpp
// mesos/3rdparty/libprocess/src/semaphore.hpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_SYNC_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_SYNC_H_

#include <semaphore.h>
#include <atomic>
#include <deque>
#include <memory>
#include <queue>

#include "utils/mesos_stout_some.h"
#include "utils/mesos_stout_synchronized.h"
#include "utils/mesos_libprocess_owned.h"
#include "utils/mesos_libprocess_owned.h"
#include "utils/mesos_libprocess_future.h"

namespace bubblefs {
namespace mymesos {
namespace process {

class Mutex
{
public:
  Mutex() : data(new Data()) {}

  Future<Nothing> lock()
  {
    Future<Nothing> future = Nothing();

    synchronized (data->lock) {
      if (!data->locked) {
        data->locked = true;
      } else {
        Promise<Nothing> waiter;
        future = waiter.future();
        data->waiters.push(std::move(waiter));
      }
    }

    return future;
  }

  void unlock()
  {
    // NOTE: We need to grab the promise 'date->promises.front()' but
    // set it outside of the critical section because setting it might
    // trigger callbacks that try to reacquire the lock.
    Option<Promise<Nothing>> waiter;

    synchronized (data->lock) {
      if (!data->waiters.empty()) {
        // TODO(benh): Skip a future that has been discarded?
        waiter = std::move(data->waiters.front());
        data->waiters.pop();
      } else {
        data->locked = false;
      }
    }

    if (waiter.isSome()) {
      waiter->set(Nothing());
    }
  }

private:
  struct Data
  {
    Data() : locked(false) {}

    ~Data()
    {
      // TODO(benh): Fail promises?
    }

    // Rather than use a process to serialize access to the mutex's
    // internal data we use a 'std::atomic_flag'.
    std::atomic_flag lock = ATOMIC_FLAG_INIT;

    // Describes the state of this mutex.
    bool locked;

    // Represents "waiters" for this lock.
    std::queue<Promise<Nothing>> waiters;
  };

  std::shared_ptr<Data> data;
};


/**
 * ReadWriteLock is a lock that allows concurrent reads and
 * exclusive writes.
 *
 * To prevent starvation of write lock requests, reads will
 * queue when one or more write lock requests is waiting, even
 * if the read lock is currently acquired.
 */
class ReadWriteLock
{
public:
  ReadWriteLock() : data(new Data()) {}

  // TODO(bmahler): Consider returning a 'Locked' object in the
  // future as the mechanism for unlocking, rather than exposing
  // unlocking functions for all to call.
  Future<Nothing> write_lock()
  {
    Future<Nothing> future = Nothing();

    synchronized (data->lock) {
      if (!data->write_locked && data->read_locked == 0u) {
        data->write_locked = true;
      } else {
        Waiter w{Waiter::WRITE};
        future = w.promise.future();
        data->waiters.push(std::move(w));
      }
    }

    return future;
  }

  void write_unlock()
  {
    // NOTE: We need to satisfy the waiter(s) futures outside the
    // critical section because it might trigger callbacks which
    // try to reacquire a read or write lock.
    std::queue<Waiter> unblocked;

    synchronized (data->lock) {
      CHECK(data->write_locked);
      CHECK_EQ(data->read_locked, 0u);

      data->write_locked = false;

      if (!data->waiters.empty()) {
        switch (data->waiters.front().type) {
          case Waiter::READ:
            // Dequeue the group of readers at the front.
            while (!data->waiters.empty() &&
                   data->waiters.front().type == Waiter::READ) {
              unblocked.push(std::move(data->waiters.front()));
              data->waiters.pop();
            }

            data->read_locked = unblocked.size();

            break;

          case Waiter::WRITE:
            unblocked.push(std::move(data->waiters.front()));
            data->waiters.pop();
            data->write_locked = true;

            CHECK_EQ(data->read_locked, 0u);

            break;
          }
      }
    }

    while (!unblocked.empty()) {
      unblocked.front().promise.set(Nothing());
      unblocked.pop();
    }
  }

  // TODO(bmahler): Consider returning a 'Locked' object in the
  // future as the mechanism for unlocking, rather than exposing
  // unlocking functions for all to call.
  Future<Nothing> read_lock()
  {
    Future<Nothing> future = Nothing();

    synchronized (data->lock) {
      if (!data->write_locked && data->waiters.empty()) {
        data->read_locked++;
      } else {
        Waiter w{Waiter::READ};
        future = w.promise.future();
        data->waiters.push(std::move(w));
      }
    }

    return future;
  }

  void read_unlock()
  {
    // NOTE: We need to satisfy the waiter future outside the
    // critical section because it might trigger callbacks which
    // try to reacquire a read or write lock.
    Option<Waiter> waiter;

    synchronized (data->lock) {
      CHECK(!data->write_locked);
      CHECK_GT(data->read_locked, 0u);

      data->read_locked--;

      if (data->read_locked == 0u && !data->waiters.empty()) {
        CHECK_EQ(data->waiters.front().type, Waiter::WRITE);

        waiter = std::move(data->waiters.front());
        data->waiters.pop();
        data->write_locked = true;
      }
    }

    if (waiter.isSome()) {
      waiter->promise.set(Nothing());
    }
  }

private:
  struct Waiter
  {
    enum { READ, WRITE } type;
    Promise<Nothing> promise;
  };

  struct Data
  {
    Data() : read_locked(0), write_locked(false) {}

    ~Data()
    {
      // TODO(zhitao): Fail promises?
    }

    // The state of the lock can be either:
    //   (1) Unlocked: an incoming read or write grabs the lock.
    //
    //   (2) Read locked (by one or more readers): an incoming write
    //       will queue in the waiters. An incoming read will proceed
    //       if no one is waiting, otherwise it will queue.
    //
    //   (3) Write locked: incoming reads and writes will queue.

    size_t read_locked;
    bool write_locked;
    std::queue<Waiter> waiters;

    // Rather than use a process to serialize access to the
    // internal data we use a 'std::atomic_flag'.
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
  };

  std::shared_ptr<Data> data;
};

template <typename T>
class Queue
{
public:
  Queue() : data(new Data()) {}

  void put(const T& t)
  {
    // NOTE: We need to grab the promise 'date->promises.front()' but
    // set it outside of the critical section because setting it might
    // trigger callbacks that try to reacquire the lock.
    Owned<Promise<T>> promise;

    synchronized (data->lock) {
      if (data->promises.empty()) {
        data->elements.push(t);
      } else {
        promise = data->promises.front();
        data->promises.pop_front();
      }
    }

    if (promise.get() != nullptr) {
      promise->set(t);
    }
  }

  Future<T> get()
  {
    synchronized (data->lock) {
      if (data->elements.empty()) {
        data->promises.push_back(Owned<Promise<T>>(new Promise<T>()));
        return data->promises.back()->future();
      } else {
        Future<T> future = Future<T>(data->elements.front());
        data->elements.pop();
        return future;
      }
    }
  }

  size_t size() const
  {
    synchronized (data->lock) {
      return data->elements.size();
    }
  }

private:
  struct Data
  {
    Data() = default;

    ~Data()
    {
      // TODO(benh): Fail promises?
    }

    // Rather than use a process to serialize access to the queue's
    // internal data we use a 'std::atomic_flag'.
    std::atomic_flag lock = ATOMIC_FLAG_INIT;

    // Represents "waiters" for elements from the queue.
    std::deque<Owned<Promise<T>>> promises;

    // Represents elements already put in the queue.
    std::queue<T> elements;
  };

  std::shared_ptr<Data> data;
};

class KernelSemaphore
{
public:
  KernelSemaphore()
  {
    assert(sem_init(&semaphore, 0, 0) == 0);
  }

  KernelSemaphore(const KernelSemaphore& other) = delete;

  ~KernelSemaphore()
  {
    assert(sem_destroy(&semaphore) == 0);
  }

  KernelSemaphore& operator=(const KernelSemaphore& other) = delete;

  void wait()
  {
    int result = sem_wait(&semaphore);

    while (result != 0 && errno == EINTR) {
      result = sem_wait(&semaphore);
    }

    assert(result == 0);
  }

  void signal()
  {
    assert(sem_post(&semaphore) == 0);
  }

private:
  sem_t semaphore;
};

// Provides a "decomissionable" kernel semaphore which allows us to
// effectively flush all waiters and keep any future threads from
// waiting. In order to be able to decomission the semaphore we need
// to keep around the number of waiters so we can signal them all.
class DecomissionableKernelSemaphore : public KernelSemaphore
{
public:
  void wait()
  {
    // NOTE: we must check `commissioned` AFTER we have incremented
    // `waiters` otherwise we might race with `decomission()` and fail
    // to properly get signaled.
    waiters.fetch_add(1);

    if (!comissioned.load()) {
      waiters.fetch_sub(1);
      return;
    }

    KernelSemaphore::wait();

    waiters.fetch_sub(1);
  }

  void decomission()
  {
    comissioned.store(false);

    // Now signal all the waiters so they wake up and stop
    // waiting. Note that this may do more `signal()` than necessary
    // but since no future threads will wait that doesn't matter (it
    // would only matter if we cared about the value of the semaphore
    // which in the current implementation we don't).
    for (size_t i = waiters.load(); i > 0; i--) {
      signal();
    }
  }

  bool decomissioned() const
  {
    return !comissioned.load();
  }

  size_t capacity() const
  {
    // The semaphore probably doesn't actually support this many but
    // who knows how to get this value otherwise.
    return SIZE_MAX;
  }

private:
  std::atomic<bool> comissioned = ATOMIC_VAR_INIT(true);
  std::atomic<size_t> waiters = ATOMIC_VAR_INIT(0);
};


// Empirical evidence (see SVG's attached at
// https://issues.apache.org/jira/browse/MESOS-7798) has shown that
// the semaphore implementation on Linux has some performance
// issues. The two biggest issues we saw:
//
//   (1) When there are many threads contending on the same semaphore
//       but there are not very many "units of resource" available
//       then the threads will spin in the kernel spinlock associated
//       with the futex.
//
//   (2) After a thread is signaled but before the thread wakes up
//       other signaling threads may attempt to wake up that thread
//       again. This appears to be because in the Linux/glibc
//       implementation only the waiting thread decrements the count
//       of waiters.
//
// The `DecomissionableLastInFirstOutFixedSizeSemaphore`
// optimizes both of the above issues. For (1) we give every thread
// their own thread-local semaphore and have them wait on that. That
// way there is effectively no contention on the kernel spinlock. For
// (2) we have the signaler decrement the number of waiters rather
// than the waiter do the decrement after it actually wakes up.
//
// Two other optimizations we introduce here:
//
//   (1) We store the threads in a last-in-first-out (LIFO) order
//       rather first-in-first-out (FIFO) ordering. The rational here
//       is that we may get better cache locality if the kernel starts
//       the thread on the same CPU and the thread works on the same
//       resource(s). This would be more pronounced if the threads
//       were pinned to cores. FIFO doesn't have any possible
//       performance wins (that we could think of) so there is nothing
//       but upside to doing LIFO instead.
//
//   (2) We use a fixed size array to store each thread's
//       semaphore. This ensures we won't need to do any memory
//       allocation or keeps us from having to do fancier lock-free
//       code to deal with growing (or shrinking) the storage for the
//       thread-local semaphores.
//
// As mentioned above, every thread get's its own semaphore that is
// used to wait on the actual semaphore. Because a thread can only be
// waiting on a single semaphore at a time it's safe for each thread
// to only have one.
thread_local KernelSemaphore* __semaphore__ = nullptr;

// Using Clang we weren't able to initialize `__semaphore__` likely
// because it is declared `thread_local` so instead we dereference the
// semaphore on every read.
#define _semaphore_                                                 \
  (__semaphore__ == nullptr ? __semaphore__ = new KernelSemaphore() \
                            : __semaphore__)

class DecomissionableLastInFirstOutFixedSizeSemaphore
{
public:
  // TODO(benh): enable specifying the number of threads that will use
  // this semaphore. Currently this is difficult because we construct
  // the `RunQueue` and later this class before we've determined the
  // number of worker threads we'll create.
  DecomissionableLastInFirstOutFixedSizeSemaphore()
  {
    for (size_t i = 0; i < semaphores.size(); i++) {
      semaphores[i] = nullptr;
    }
  }

  void signal()
  {
    // NOTE: we _always_ increment `count` which means that even if we
    // try and signal a thread another thread might have come in and
    // decremented `count` already. This is deliberate, but it would
    // be interesting to also investigate the performance where we
    // always signal a new thread.
    count.fetch_add(1);

    while (waiters.load() > 0 && count.load() > 0) {
      for (size_t i = 0; i < semaphores.size(); i++) {
        // Don't bother finding a semaphore to signal if there isn't
        // anybody to signal (`waiters` == 0) or anything to do
        // (`count` == 0).
        if (waiters.load() == 0 || count.load() == 0) {
          return;
        }

        // Try and find and then signal a waiter.
        //
        // TODO(benh): we `load()` first and then do a
        // compare-and-swap because the read shouldn't require a lock
        // instruction or synchronizing the bus. In addition, we
        // should be able to optimize the loads in the future to a
        // weaker memory ordering. That being said, if we don't see
        // performance wins when trying that we should consider just
        // doing a `std::atomic::exchange()` instead.
        KernelSemaphore* semaphore = semaphores[i].load();
        if (semaphore != nullptr) {
          if (!semaphores[i].compare_exchange_strong(semaphore, nullptr)) {
            continue;
          }

          // NOTE: we decrement `waiters` _here_ rather than in `wait`
          // so that future signalers won't bother looping here
          // (potentially for a long time) trying to find a waiter
          // that might have already been signaled but just hasn't
          // woken up yet. We even go as far as decrementing `waiters`
          // _before_ we signal to really keep a thread from having to
          // loop.
          waiters.fetch_sub(1);

          semaphore->signal();

          return;
        }
      }
    }
  }

  void wait()
  {
    do {
      size_t old = count.load();
      while (old > 0) {
      CAS:
        if (!count.compare_exchange_strong(old, old - 1)) {
          continue;
        }
        return;
      }

      // Need to actually wait (slow path).
      waiters.fetch_add(1);

      // NOTE: we must check `commissioned` AFTER we have
      // incremented `waiters` otherwise we might race with
      // `decomission()` and fail to properly get signaled.
      if (!comissioned.load()) {
        waiters.fetch_sub(1);
        return;
      }

      bool done = false;
      while (!done) {
        for (size_t i = 0; i < semaphores.size(); i++) {
          // NOTE: see TODO in `signal()` above for why we do the
          // `load()` first rather than trying to compare-and-swap
          // immediately.
          KernelSemaphore* semaphore = semaphores[i].load();
          if (semaphore == nullptr) {
            // NOTE: we _must_ check one last time if we should really
            // wait because there is a race that `signal()` was
            // completely executed in between when we checked `count`
            // and when we incremented `waiters` and hence we could
            // wait forever. We delay this check until the 11th hour
            // so that we can also benefit from the possibility that
            // more things have been enqueued while we were looking
            // for a slot in the array.
            if ((old = count.load()) > 0) {
              waiters.fetch_sub(1);
              goto CAS;
            }
            if (semaphores[i].compare_exchange_strong(semaphore, _semaphore_)) {
              done = true;
              break;
            }
          }
        }
      }

      // TODO(benh): To make this be wait-free for the signalers we
      // need to enqueue semaphore before we increment `waiters`. The
      // reason we can't do that right now is because we don't know
      // how to remove ourselves from `semaphores` if, after checking
      // `count` (which we need to do due to the race between
      // signaling and waiting) we determine that we don't need to
      // wait (because then we have our semaphore stuck in the
      // queue). A solution here could be to have a fixed size queue
      // that we can just remove ourselves from, but then note that
      // we'll need to set the semaphore back to zero in the event
      // that it got signaled so the next time we don't _not_ wait.

      _semaphore_->wait();
    } while (true);
  }

  void decomission()
  {
    comissioned.store(false);

    // Now signal all the waiters so they wake up and stop
    // waiting. Note that this may do more `signal()` than necessary
    // but since no future threads will wait that doesn't matter (it
    // would only matter if we cared about the value of the semaphore
    // which in the current implementation we don't).
    for (size_t i = waiters.load(); i > 0; i--) {
      signal();
    }
  }

  bool decomissioned() const
  {
    return !comissioned.load();
  }

  size_t capacity() const
  {
    return semaphores.size();
  }

private:
  // Maximum number of threads that could ever wait on this semaphore.
  static constexpr size_t THREADS = 128;

  // Indicates whether or not this semaphore has been decomissioned.
  std::atomic<bool> comissioned = ATOMIC_VAR_INIT(true);

  // Count of currently available "units of resource" represented by
  // this semaphore.
  std::atomic<size_t> count = ATOMIC_VAR_INIT(0);

  // Number of threads waiting for an available "unit of resource".
  std::atomic<size_t> waiters = ATOMIC_VAR_INIT(0);

  // Fixed array holding thread-local semaphores used for waiting and
  // signaling threads.
  std::array<std::atomic<KernelSemaphore*>, THREADS> semaphores;
};

} // namespace process
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_SYNC_H_