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

// mesos/3rdparty/libprocess/src/libevent.hpp
// mesos/3rdparty/libprocess/src/libevent.cpp
// mesos/3rdparty/libprocess/src/libevent_poll.cpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_LIBEVENT_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_LIBEVENT_H_

#include <unistd.h>
#include <memory>
#include <mutex>
#include <queue>

#include "utils/mesos_stout_lambda.h"
#include "utils/mesos_stout_synchronized.h"
#include "utils/mesos_libprocess_once.h"
#include "utils/mesos_libprocess_event_loop.h"

#include <event2/event.h>
#include <event2/thread.h>
#include <event2/util.h>

namespace bubblefs {
namespace mymesos {
namespace process {

// Event loop.
extern event_base* base;


// Per thread bool pointer. We use a pointer to lazily construct the
// actual bool.
extern thread_local bool* _in_event_loop_;


#define __in_event_loop__ *(_in_event_loop_ == nullptr ?                \
  _in_event_loop_ = new bool(false) : _in_event_loop_)


enum EventLoopLogicFlow
{
  ALLOW_SHORT_CIRCUIT,
  DISALLOW_SHORT_CIRCUIT
};


void run_in_event_loop(
    const lambda::function<void()>& f,
    EventLoopLogicFlow event_loop_logic_flow = ALLOW_SHORT_CIRCUIT);

event_base* base = nullptr;

static std::mutex* functions_mutex = new std::mutex();
std::queue<lambda::function<void()>>* functions =
  new std::queue<lambda::function<void()>>();


thread_local bool* _in_event_loop_ = nullptr;


void async_function(evutil_socket_t socket, short which, void* arg)
{
  event* ev = reinterpret_cast<event*>(arg);
  event_free(ev);

  std::queue<lambda::function<void()>> q;

  synchronized (functions_mutex) {
    std::swap(q, *functions);
  }

  while (!q.empty()) {
    q.front()();
    q.pop();
  }
}


void run_in_event_loop(
    const lambda::function<void()>& f,
    EventLoopLogicFlow event_loop_logic_flow)
{
  if (__in_event_loop__ && event_loop_logic_flow == ALLOW_SHORT_CIRCUIT) {
    f();
    return;
  }

  synchronized (functions_mutex) {
    functions->push(f);

    // Add an event and activate it to interrupt the event loop.
    // TODO(jmlvanre): after libevent v 2.1 we can use
    // event_self_cbarg instead of re-assigning the event. For now we
    // manually re-assign the event to pass in the pointer to the
    // event itself as the callback argument.
    event* ev = evtimer_new(base, async_function, nullptr);

    // 'event_assign' is only valid on non-pending AND non-active
    // events. This means we have to assign the callback before
    // calling 'event_active'.
    if (evtimer_assign(ev, base, async_function, ev) < 0) {
      LOG(FATAL) << "Failed to assign callback on event";
    }

    event_active(ev, EV_TIMEOUT, 0);
  }
}


void EventLoop::run()
{
  __in_event_loop__ = true;

  do {
    int result = event_base_loop(base, EVLOOP_ONCE);
    if (result < 0) {
      LOG(FATAL) << "Failed to run event loop";
    } else if (result > 0) {
      // All events are handled, continue event loop.
      continue;
    } else {
      CHECK_EQ(0, result);
      if (event_base_got_break(base)) {
        break;
      } else if (event_base_got_exit(base)) {
        break;
      }
    }
  } while (true);

  __in_event_loop__ = false;
}


void EventLoop::stop()
{
  event_base_loopexit(base, nullptr);
}


namespace internal {

struct Delay
{
  lambda::function<void()> function;
  event* timer;
};

void handle_delay(evutil_socket_t, short, void* arg)
{
  Delay* delay = reinterpret_cast<Delay*>(arg);
  delay->function();
  event_free(delay->timer);
  delete delay;
}

}  // namespace internal {


void EventLoop::delay(
    const Duration& duration,
    const lambda::function<void()>& function)
{
  internal::Delay* delay = new internal::Delay();
  delay->timer = evtimer_new(base, &internal::handle_delay, delay);
  if (delay->timer == nullptr) {
    LOG(FATAL) << "Failed to delay, evtimer_new";
  }

  delay->function = function;

  timeval t{0, 0};
  if (duration > Seconds(0)) {
    t = duration.timeval();
  }

  evtimer_add(delay->timer, &t);
}


double EventLoop::time()
{
  // We explicitly call `evutil_gettimeofday()` for now to avoid any
  // issues that may be introduced by using the cached value provided
  // by `event_base_gettimeofday_cached()`. Since a lot of logic in
  // libprocess depends on time math, we want to log fatal rather than
  // cause logic errors if the time fails.
  timeval t;
  if (evutil_gettimeofday(&t, nullptr) < 0) {
    LOG(FATAL) << "Failed to get time, evutil_gettimeofday";
  }

  return Duration(t).secs();
}


void EventLoop::initialize()
{
  static Once* initialized = new Once();

  if (initialized->once()) {
    return;
  }

  // We need to initialize Libevent differently depending on the
  // operating system threading support.
#if defined(EVTHREAD_USE_PTHREADS_IMPLEMENTED)
  if (evthread_use_pthreads() < 0) {
    LOG(FATAL) << "Failed to initialize, evthread_use_pthreads";
  }
#elif defined(EVTHREAD_USE_WINDOWS_THREADS_IMPLEMENTED)
  if (evthread_use_windows_threads() < 0) {
    LOG(FATAL) << "Failed to initialize, evthread_use_windows_threads";
  }
#else
#error "Libevent must be compiled with either pthread or Windows thread support"
#endif

  // This enables debugging of libevent calls. We can remove this
  // when the implementation settles and after we gain confidence.
  event_enable_debug_mode();

  // TODO(jmlvanre): Allow support for 'epoll' once SSL related
  // issues are resolved.
  struct event_config* config = event_config_new();
  event_config_avoid_method(config, "epoll");

  base = event_base_new_with_config(config);

  if (base == nullptr) {
    LOG(FATAL) << "Failed to initialize, event_base_new";
  }

  initialized->done();
}

namespace io {
namespace internal {

struct Poll
{
  Promise<short> promise;
  std::shared_ptr<event> ev;
};


void pollCallback(evutil_socket_t, short what, void* arg)
{
  Poll* poll = reinterpret_cast<Poll*>(arg);

  if (poll->promise.future().hasDiscard()) {
    poll->promise.discard();
  } else {
    // Convert libevent specific EV_READ / EV_WRITE to io::* specific
    // values of these enumerations.
    short events =
      ((what & EV_READ) ? io::READ : 0) | ((what & EV_WRITE) ? io::WRITE : 0);

    poll->promise.set(events);
  }

  // Deleting the `poll` also destructs `ev` and hence triggers `event_free`,
  // which makes the event non-pending.
  delete poll;
}


void pollDiscard(const std::weak_ptr<event>& ev, short events)
{
  // Discarding inside the event loop prevents `pollCallback()` from being
  // called twice if the future is discarded.
  run_in_event_loop([=]() {
    std::shared_ptr<event> shared = ev.lock();
    // If `ev` cannot be locked `pollCallback` already ran. If it was locked
    // but not pending, `pollCallback` is scheduled to be executed.
    if (static_cast<bool>(shared) &&
        event_pending(shared.get(), events, nullptr)) {
      // `event_active` will trigger the `pollCallback` to be executed.
      event_active(shared.get(), EV_READ, 0);
    }
  });
}

} // namespace internal {


Future<short> poll(int_fd fd, short events)
{
  process::initialize();

  internal::Poll* poll = new internal::Poll();

  Future<short> future = poll->promise.future();

  // Convert io::READ / io::WRITE to libevent specific values of these
  // enumerations.
  short what =
    ((events & io::READ) ? EV_READ : 0) | ((events & io::WRITE) ? EV_WRITE : 0);

  // Bind `event_free` to the destructor of the `ev` shared pointer
  // guaranteeing that the event will be freed only once.
  poll->ev.reset(
      event_new(base, fd, what, &internal::pollCallback, poll),
      event_free);

  if (poll->ev == nullptr) {
    LOG(FATAL) << "Failed to poll, event_new";
  }

  // Using a `weak_ptr` prevents `ev` to become a dangling pointer if
  // the returned future is discarded after the event is triggered.
  // The `weak_ptr` needs to be created before `event_add` in case
  // the event is ready and the callback is executed before creating
  // `ev`.
  std::weak_ptr<event> ev(poll->ev);

  event_add(poll->ev.get(), nullptr);

  return future
    .onDiscard(lambda::bind(&internal::pollDiscard, ev, what));
}

} // namespace io

} // namespace process
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_LIBEVENT_H_