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

// mesos/3rdparty/libprocess/src/libev.hpp
// mesos/3rdparty/libprocess/src/libev.cpp
// mesos/3rdparty/libprocess/src/libev_poll.cpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_LIBEV_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_LIBEV_H_

#include <memory>
#include <mutex>
#include <queue>

#include "utils/mesos_libprocess_event_loop.h"
#include "utils/mesos_libprocess_owned.h"
#include "utils/mesos_stout_duration.h"
#include "utils/mesos_stout_lambda.h"
#include "utils/mesos_stout_some.h"
#include "utils/mesos_stout_synchronized.h"
#include "utils/mesos_libprocess_future.h"

#include <ev.h>

namespace bubblefs {
namespace mymesos {
namespace process {

// Event loop.
extern struct ev_loop* loop;

// Asynchronous watcher for interrupting loop to specifically deal
// with IO watchers and functions (via run_in_event_loop).
extern ev_async async_watcher;

// Queue of I/O watchers to be asynchronously added to the event loop
// (protected by 'watchers' below).
// TODO(benh): Replace this queue with functions that we put in
// 'functions' below that perform the ev_io_start themselves.
extern std::queue<ev_io*>* watchers;
extern std::mutex* watchers_mutex;

// Queue of functions to be invoked asynchronously within the vent
// loop (protected by 'watchers' above).
extern std::queue<lambda::function<void()>>* functions;

// Per thread bool pointer. We use a pointer to lazily construct the
// actual bool.
extern thread_local bool* _in_event_loop_;

#define __in_event_loop__ *(_in_event_loop_ == nullptr ?                \
  _in_event_loop_ = new bool(false) : _in_event_loop_)


// Wrapper around function we want to run in the event loop.
template <typename T>
void _run_in_event_loop(
    const lambda::function<Future<T>()>& f,
    const Owned<Promise<T>>& promise)
{
  // Don't bother running the function if the future has been discarded.
  if (promise->future().hasDiscard()) {
    promise->discard();
  } else {
    promise->set(f());
  }
}


// Helper for running a function in the event loop.
template <typename T>
Future<T> run_in_event_loop(const lambda::function<Future<T>()>& f)
{
  // If this is already the event loop then just run the function.
  if (__in_event_loop__) {
    return f();
  }

  Owned<Promise<T>> promise(new Promise<T>());

  Future<T> future = promise->future();

  // Enqueue the function.
  synchronized (watchers_mutex) {
    functions->push(lambda::bind(&_run_in_event_loop<T>, f, promise));
  }

  // Interrupt the loop.
  ev_async_send(loop, &async_watcher);

  return future;
}

ev_async async_watcher;
// We need an asynchronous watcher to receive the request to shutdown.
ev_async shutdown_watcher;

// Define the initial values for all of the declarations made in
// libev.hpp (since these need to live in the static data space).
struct ev_loop* loop = nullptr;

std::queue<ev_io*>* watchers = new std::queue<ev_io*>();

std::mutex* watchers_mutex = new std::mutex();

std::queue<lambda::function<void()>>* functions =
  new std::queue<lambda::function<void()>>();

thread_local bool* _in_event_loop_ = nullptr;


void handle_async(struct ev_loop* loop, ev_async* _, int revents)
{
  std::queue<lambda::function<void()>> run_functions;
  synchronized (watchers_mutex) {
    // Start all the new I/O watchers.
    while (!watchers->empty()) {
      ev_io* watcher = watchers->front();
      watchers->pop();
      ev_io_start(loop, watcher);
    }

    // Swap the functions into a temporary queue so that we can invoke
    // them outside of the mutex.
    std::swap(run_functions, *functions);
  }

  // Running the functions outside of the mutex reduces locking
  // contention as these are arbitrary functions that can take a long
  // time to execute. Doing this also avoids a deadlock scenario where
  // (A) mutexes are acquired before calling `run_in_event_loop`,
  // followed by locking (B) `watchers_mutex`. If we executed the
  // functions inside the mutex, then the locking order violation
  // would be this function acquiring the (B) `watchers_mutex`
  // followed by the arbitrary function acquiring the (A) mutexes.
  while (!run_functions.empty()) {
    (run_functions.front())();
    run_functions.pop();
  }
}


void handle_shutdown(struct ev_loop* loop, ev_async* _, int revents)
{
  ev_unloop(loop, EVUNLOOP_ALL);
}


void EventLoop::initialize()
{
  loop = ev_default_loop(EVFLAG_AUTO);

  ev_async_init(&async_watcher, handle_async);
  ev_async_init(&shutdown_watcher, handle_shutdown);

  ev_async_start(loop, &async_watcher);
  ev_async_start(loop, &shutdown_watcher);
}


namespace internal {

void handle_delay(struct ev_loop* loop, ev_timer* timer, int revents)
{
  lambda::function<void()>* function =
    reinterpret_cast<lambda::function<void()>*>(timer->data);
  (*function)();
  delete function;
  ev_timer_stop(loop, timer);
  delete timer;
}


Future<Nothing> delay(
    const Duration& duration,
    const lambda::function<void()>& function)
{
  ev_timer* timer = new ev_timer();
  timer->data = reinterpret_cast<void*>(new lambda::function<void()>(function));

  // Determine the 'after' parameter to pass to libev and set it to 0
  // in the event that it's negative so that we always make sure to
  // invoke 'function' even if libev doesn't support negative 'after'
  // values.
  double after = duration.secs();

  if (after < 0) {
    after = 0;
  }

  const double repeat = 0.0;

  ev_timer_init(timer, handle_delay, after, repeat);
  ev_timer_start(loop, timer);

  return Nothing();
}

} // namespace internal {


void EventLoop::delay(
    const Duration& duration,
    const lambda::function<void()>& function)
{
  run_in_event_loop<Nothing>(
      lambda::bind(&internal::delay, duration, function));
}


double EventLoop::time()
{
  // TODO(benh): Versus ev_now()?
  return ev_time();
}


void EventLoop::run()
{
  __in_event_loop__ = true;

  ev_loop(loop, 0);

  __in_event_loop__ = false;
}


void EventLoop::stop()
{
  ev_async_send(loop, &shutdown_watcher);
}

// Data necessary for polling so we can discard polling and actually
// stop it in the event loop.
struct Poll
{
  Poll()
  {
    // Need to explicitly instantiate the watchers.
    watcher.io.reset(new ev_io());
    watcher.async.reset(new ev_async());
  }

  // An I/O watcher for checking for readability or writeability and
  // an async watcher for being able to discard the polling.
  struct {
    std::shared_ptr<ev_io> io;
    std::shared_ptr<ev_async> async;
  } watcher;

  Promise<short> promise;
};


// Event loop callback when I/O is ready on polling file descriptor.
void polled(struct ev_loop* loop, ev_io* watcher, int revents)
{
  Poll* poll = (Poll*) watcher->data;

  ev_io_stop(loop, poll->watcher.io.get());

  // Stop the async watcher (also clears if pending so 'discard_poll'
  // will not get invoked and we can delete 'poll' here).
  ev_async_stop(loop, poll->watcher.async.get());

  poll->promise.set(revents);

  delete poll;
}


// Event loop callback when future associated with polling file
// descriptor has been discarded.
void discard_poll(struct ev_loop* loop, ev_async* watcher, int revents)
{
  Poll* poll = (Poll*) watcher->data;

  // Check and see if we have a pending 'polled' callback and if so
  // let it "win".
  if (ev_is_pending(poll->watcher.io.get())) {
    return;
  }

  ev_async_stop(loop, poll->watcher.async.get());

  // Stop the I/O watcher (but note we check if pending above) so it
  // won't get invoked and we can delete 'poll' here.
  ev_io_stop(loop, poll->watcher.io.get());

  poll->promise.discard();

  delete poll;
}


namespace io {
  
namespace internal {

// Helper/continuation of 'poll' on future discard.
void _poll(const std::shared_ptr<ev_async>& async)
{
  ev_async_send(loop, async.get());
}


Future<short> poll(int_fd fd, short events)
{
  Poll* poll = new Poll();

  // Have the watchers data point back to the struct.
  poll->watcher.async->data = poll;
  poll->watcher.io->data = poll;

  // Get a copy of the future to avoid any races with the event loop.
  Future<short> future = poll->promise.future();

  // Initialize and start the async watcher.
  ev_async_init(poll->watcher.async.get(), discard_poll);
  ev_async_start(loop, poll->watcher.async.get());

  // Make sure we stop polling if a discard occurs on our future.
  // Note that it's possible that we'll invoke '_poll' when someone
  // does a discard even after the polling has already completed, but
  // in this case while we will interrupt the event loop since the
  // async watcher has already been stopped we won't cause
  // 'discard_poll' to get invoked.
  future.onDiscard(lambda::bind(&_poll, poll->watcher.async));

  // Initialize and start the I/O watcher.
  ev_io_init(poll->watcher.io.get(), polled, fd, events);
  ev_io_start(loop, poll->watcher.io.get());

  return future;
}

} // namespace internal {


Future<short> poll(int_fd fd, short events)
{
  process::initialize();

  // TODO(benh): Check if the file descriptor is non-blocking?

  return run_in_event_loop<short>(lambda::bind(&internal::poll, fd, events));
}

} // namespace io

} // namespace process 
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_LIBEV_H_