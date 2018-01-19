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

// mesos/3rdparty/libprocess/include/process/clock.hpp
// mesos/3rdparty/libprocess/src/clock.cpp
// mesos/3rdparty/libprocess/include/process/delay.hpp
// mesos/3rdparty/libprocess/include/process/after.hpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_CLOCK_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_CLOCK_H_

#include <list>

#include <process/time.hpp>

// TODO(benh): We should really be including <process/timer.hpp> but
// there are currently too many circular dependencies in header files
// that would need to get moved to translation units first.

#include <stout/duration.hpp>
#include <stout/lambda.hpp>

// Clock Management and Timeouts
// Asynchronous programs often use timeouts, e.g., 
// because a process that initiates an asynchronous operation wants to take action 
// if the operation hasn't completed within a certain time bound. 
// To facilitate this, libprocess provides a set of abstractions 
// that simplify writing timeout logic. Importantly, 
// test code has the ability to manipulate the clock, in order to ensure 
// that timeout logic is exercised (without needing to block the test program 
// until the appropriate amount of system time has elapsed).
// To invoke a function after a certain amount of time has elapsed, use delay:
// delay(Seconds(5), process.self(), &DelayedProcess::action, "Neil");
// This invokes the action function after (at least) five seconds of time have elapsed.
// When writing unit tests for this code, blocking the test for five seconds is undesirable. 
// To avoid this, we can use Clock::advance:
// delay(Seconds(5), process.self(), &DelayedProcess::action, "Neil");
// Clock::advance(Seconds(5));

namespace bubblefs {
namespace mymesos {
namespace process {

// Forward declarations (to avoid circular dependencies).
class ProcessBase;
class Time;
class Timer;

/**
 * Provides timers.
 */
class Clock
{
public:
  /**
   * Initialize the clock with the specified callback that will be
   * invoked whenever a batch of timers has expired.
   */
  // TODO(benh): Introduce a "channel" or listener pattern for getting
  // the expired Timers rather than passing in a callback. This might
  // mean we don't need 'initialize' or 'shutdown'.
  static void initialize(
      lambda::function<void(const std::list<Timer>&)>&& callback);

  /**
   * Clears all timers without executing them.
   *
   * The process manager must be properly finalized before the clock is
   * finalized.  This will eliminate the need for timers to activate and
   * prevent further timers from being added after finalization.
   *
   * Also, the Clock must not be paused when finalizing.
   */
  static void finalize();

  /**
   * The current clock time for either the current process that makes
   * this call or the global clock time if not invoked from a process.
   *
   * @return This process' current clock time or the global clock time
   *         if not invoked from a process.
   */
  static Time now();

  static Time now(ProcessBase* process);

  static Timer timer(
      const Duration& duration,
      const lambda::function<void()>& thunk);

  static bool cancel(const Timer& timer);

  /**
   * Pauses the clock e.g. for testing purposes.
   */
  static void pause();

  /**
   * Check whether clock is currently running.
   */
  static bool paused();

  static void resume();

  static void advance(const Duration& duration);
  static void advance(ProcessBase* process, const Duration& duration);

  static void update(const Time& time);

  // When updating the time of a particular process you can specify
  // whether or not you want to override the existing value even if
  // you're going backwards in time! SAFE means don't update the
  // previous Clock for a process if going backwards in time, where as
  // FORCE forces this change.
  enum Update
  {
    SAFE,  /*!< Don't update Clock for a process if going backwards in time. */
    FORCE, /*!< Update Clock even if going backwards in time. */
  };

  static void update(
      ProcessBase* process,
      const Time& time,
      Update update = SAFE);

  static void order(ProcessBase* from, ProcessBase* to);

  // When the clock is paused this returns only after
  //   (1) all expired timers are executed,
  //   (2) no processes are running, and
  //   (3) no processes are ready to run.
  //
  // In other words, this function blocks synchronously until no other
  // execution on any processes will occur unless the clock is
  // advanced.
  //
  // TODO(benh): Move this function elsewhere, for instance, to a
  // top-level function in the 'process' namespace since it deals with
  // both processes and the clock.
  static void settle();

  // When the clock is paused this returns true if all timers that
  // expire before the paused time have executed, otherwise false.
  // Note that if the clock continually gets advanced concurrently
  // this function may never return true because the "paused" time
  // will continue to get pushed out farther in the future making more
  // timers candidates for execution.
  static bool settled();
};

// We store the timers in a map of lists indexed by the timeout of the
// timer so that we can have two timers that have the same timeout. We
// exploit that the map is SORTED!
static std::map<Time, std::list<Timer>>* timers = new std::map<Time, list<Timer>>();
static recursive_mutex* timers_mutex = new recursive_mutex();


// We namespace the clock related variables to keep them well
// named. In the future we'll probably want to associate a clock with
// a specific ProcessManager/SocketManager instance pair, so this will
// likely change.
namespace clock {

std::map<ProcessBase*, Time>* currents = new map<ProcessBase*, Time>();

Time* initial = new Time(Time::epoch());
Time* current = new Time(Time::epoch());

Duration* advanced = new Duration(Duration::zero());

bool paused = false;

// For supporting Clock::settled(), false if we're not currently
// settling (or we're not paused), true if we're currently attempting
// to settle (and we're paused).
bool settling = false;

// Lambda function to invoke when timers have expired.
lambda::function<void(const list<Timer>&)>* callback =
    new lambda::function<void(const list<Timer>&)>();

// Keep track of 'ticks' that have been scheduled. To reduce the
// number of outstanding delays on the EventLoop system, we only
// schedule a _new_ 'tick' when it's earlier than all currently
// scheduled 'ticks'.
set<Time>* ticks = new set<Time>();


// Helper for determining the time when the next timer elapses,
// or None if no timers are pending, or the clock is paused and no
// timers are expired. Note that we don't manipulate 'timers' directly
// so that it's clear from the callsite that the use of 'timers' is
// within a 'synchronized' block.
//
// TODO(benh): Create a generic 'Timers' abstraction which hides this
// and more away (i.e., all manipulations of 'timers' below).
Option<Time> next(const map<Time, list<Timer>>& timers)
{
  if (!timers.empty()) {
    Time first = timers.begin()->first;

    // If the clock is paused and no timers are expired, the
    // timers cannot fire until the clock is advanced, so we
    // return None() here. Note that we pass nullptr to ensure
    // that this looks at the global clock, since this can be
    // called from a Process context through Clock::timer.
    if (Clock::paused() && first > Clock::now(nullptr)) {
      return None();
    }

    return first;
  }

  return None();
}


// Forward declaration for scheduleTick.
void tick(const Time& time);


// Helper for scheduling the next clock tick, if applicable. Note
// that we don't manipulate 'timers' or 'ticks' directly so that
// it's clear from the callsite that this needs to be called within
// a 'synchronized' block.
// TODO(bmahler): Consider taking an optional 'now' to avoid
// excessive syscalls via Clock::now(nullptr).
void scheduleTick(const map<Time, list<Timer>>& timers, set<Time>* ticks)
{
  // Determine when the next 'tick' should fire.
  const Option<Time> next = clock::next(timers);

  if (next.isSome()) {
    // Don't schedule a 'tick' if there is a 'tick' scheduled for
    // an earlier time, to avoid excessive pending timers.
    if (ticks->empty() || next.get() < (*ticks->begin())) {
      ticks->insert(next.get());

      // The delay can be negative if the timer is expired, this
      // is expected will result in a 'tick' firing immediately.
      const Duration delay = next.get() - Clock::now(nullptr);
      EventLoop::delay(delay, lambda::bind(tick, next.get()));
    }
  }
}


// NOTE: This method must remain robust to arbitrary invocations.
// i.e. `tick` should not make any assumptions of what is held in `timers`,
// which can be empty or have timers that trigger later than the current time.
void tick(const Time& time)
{
  std::list<Timer> timedout;

  synchronized (timers_mutex) {
    // We pass nullptr to be explicit about the fact that we want the
    // global clock time, even though it's unnecessary ('tick' is
    // called from the event loop, not a Process context).
    Time now = Clock::now(nullptr);

    //VLOG(3) << "Handling timers up to " << now;

    foreachkey (const Time& timeout, *timers) {
      if (timeout > now) {
        break;
      }

      //VLOG(3) << "Have timeout(s) at " << timeout;

      // Need to toggle 'settling' so that we don't prematurely say
      // we're settled until after the timers are executed below,
      // outside of the critical section.
      if (clock::paused) {
        clock::settling = true;
      }

      timedout.splice(timedout.end(), (*timers)[timeout]);
    }

    // Now erase the range of timers that timed out.
    timers->erase(timers->begin(), timers->upper_bound(now));

    // Okay, so the timeout for the next timer should not have fired.
    //CHECK(timers->empty() || (timers->begin()->first > now));

    // Remove this tick from the scheduled 'ticks', it may have
    // been removed already if the clock was paused / manipulated
    // in the interim.
    ticks->erase(time);

    // Schedule another "tick" if necessary.
    scheduleTick(*timers, ticks);
  }

  (*clock::callback)(timedout);

  timedout.clear();

  // Mark 'settling' as false since there are not any more timers
  // that will expire before the paused time and we've finished
  // executing expired timers.
  synchronized (timers_mutex) {
    if (clock::paused &&
        (timers->size() == 0 ||
         timers->begin()->first > *clock::current)) {
      VLOG(3) << "Clock has settled";
      clock::settling = false;
    }
  }
}

} // namespace clock

void Clock::initialize(lambda::function<void(const list<Timer>&)>&& callback)
{
  (*clock::callback) = callback;
}


void Clock::finalize()
{
  // We keep track of state differently when the clock is paused.
  // Finalization only handles cleanup of a running clock.
  //CHECK(!clock::paused) << "Clock must not be paused when finalizing";

  synchronized (timers_mutex) {
    // NOTE: `currents` is only non-empty when the clock is paused.

    // This, along with the `timers_mutex`, is all that is required to clean
    // up any pending timers.  Timers are triggered via "ticks".  However,
    // we do not need to clear `ticks` because a "tick" with an empty `timers`
    // map will effectively be a no-op.
    timers->clear();
  }
}


Time Clock::now()
{
  return now(__process__);
}


Time Clock::now(ProcessBase* process)
{
  synchronized (timers_mutex) {
    if (Clock::paused()) {
      if (process != nullptr) {
        if (clock::currents->count(process) != 0) {
          return (*clock::currents)[process];
        } else {
          return (*clock::currents)[process] = *clock::initial;
        }
      } else {
        return *clock::current;
      }
    }
  }

  double d = EventLoop::time();
  Try<Time> time = Time::create(d); // Compensates for clock::advanced.

  // TODO(xujyan): Move CHECK_SOME to libprocess and add CHECK_SOME
  // here.
  if (time.isError()) {
    LOG(FATAL) << "Failed to create a Time from " << d << ": "
               << time.error();
  }
  return time.get();
}


Timer Clock::timer(
    const Duration& duration,
    const lambda::function<void()>& thunk)
{
  // Start at 1 since Timer() instances use id 0.
  static std::atomic<uint64_t> id(1);

  // Assumes Clock::now() does Clock::now(__process__).
  Timeout timeout = Timeout::in(duration);

  UPID pid = __process__ != nullptr ? __process__->self() : UPID();

  Timer timer(id.fetch_add(1), timeout, pid, thunk);

  VLOG(3) << "Created a timer for " << pid << " in " << stringify(duration)
          << " in the future (" << timeout.time() << ")";

  // Add the timer.
  synchronized (timers_mutex) {
    if (timers->size() == 0 ||
        timer.timeout().time() < timers->begin()->first) {
      // Need to interrupt the loop to update/set timer repeat.
      (*timers)[timer.timeout().time()].push_back(timer);

      // Schedule another "tick" if necessary.
      clock::scheduleTick(*timers, clock::ticks);
    } else {
      // Timer repeat is adequate, just add the timeout.
      CHECK(timers->size() >= 1);
      (*timers)[timer.timeout().time()].push_back(timer);
    }
  }

  return timer;
}


bool Clock::cancel(const Timer& timer)
{
  bool canceled = false;
  synchronized (timers_mutex) {
    // Check if the timeout is still pending, and if so, erase it. In
    // addition, erase an empty list if we just removed the last
    // timeout.
    Time time = timer.timeout().time();
    if (timers->count(time) > 0) {
      canceled = true;
      (*timers)[time].remove(timer);
      if ((*timers)[time].empty()) {
        timers->erase(time);
      }
    }
  }

  return canceled;
}


void Clock::pause()
{
  process::initialize(); // To make sure the event loop is ready.

  synchronized (timers_mutex) {
    if (!clock::paused) {
      *clock::initial = *clock::current = now();
      clock::paused = true;
      VLOG(2) << "Clock paused at " << *clock::initial;

      // When the clock is paused, we clear the scheduled 'ticks'
      // since they no longer accurately represent when a 'tick'
      // will fire (our notion of "time" is now moving differently
      // from that of the event loop). Note that only 'ticks'
      // that fire immediately will be scheduled while the clock
      // is paused.
      clock::ticks->clear();
    }
  }

  // Note that after pausing the clock, the existing scheduled
  // 'ticks' might still fire, but since 'paused' == true no "time"
  // will actually have passed, so no timer will actually fire.
}


bool Clock::paused()
{
  return clock::paused;
}


void Clock::resume()
{
  process::initialize(); // To make sure the event loop is ready.

  synchronized (timers_mutex) {
    if (clock::paused) {
      VLOG(2) << "Clock resumed at " << *clock::current;

      clock::paused = false;
      clock::settling = false;
      clock::currents->clear();

      // Schedule another "tick" if necessary.
      clock::scheduleTick(*timers, clock::ticks);
    }
  }
}


void Clock::advance(const Duration& duration)
{
  synchronized (timers_mutex) {
    if (clock::paused) {
      *clock::advanced += duration;
      *clock::current += duration;

      VLOG(2) << "Clock advanced (" << duration << ") to " << *clock::current;

      // Schedule another "tick" if necessary. Only "ticks" that
      // fire immediately will be scheduled here, since the clock
      // is paused.
      clock::scheduleTick(*timers, clock::ticks);
    }
  }
}


void Clock::advance(ProcessBase* process, const Duration& duration)
{
  synchronized (timers_mutex) {
    if (clock::paused) {
      Time current = now(process);
      current += duration;
      (*clock::currents)[process] = current;
      VLOG(2) << "Clock of " << process->self() << " advanced (" << duration
              << ") to " << current;

      // When the clock is advanced for a specific process, we do not
      // need to schedule another "tick", as done in the global
      // advance() above. This is because the clock ticks are based
      // on global time, not per-Process time.
    }
  }
}


void Clock::update(const Time& time)
{
  synchronized (timers_mutex) {
    if (clock::paused) {
      if (*clock::current < time) {
        *clock::advanced += (time - *clock::current);
        *clock::current = Time(time);
        VLOG(2) << "Clock updated to " << *clock::current;

        // Schedule another "tick" if necessary. Only "ticks" that
        // fire immediately will be scheduled here, since the clock
        // is paused.
        clock::scheduleTick(*timers, clock::ticks);
      }
    }
  }
}


void Clock::update(ProcessBase* process, const Time& time, Update update)
{
  synchronized (timers_mutex) {
    if (clock::paused) {
      if (now(process) < time || update == Clock::FORCE) {
        VLOG(2) << "Clock of " << process->self() << " updated to " << time;
        (*clock::currents)[process] = Time(time);

        // When the clock is updated for a specific process, we do not
        // need to schedule another "tick", as done in the global
        // update() above. This is because the clock ticks are based
        // on global time, not per-Process time.
      }
    }
  }
}


void Clock::order(ProcessBase* from, ProcessBase* to)
{
  VLOG(2) << "Clock of " << to->self() << " being updated to " << from->self();
  update(to, now(from));
}


bool Clock::settled()
{
  synchronized (timers_mutex) {
    CHECK(clock::paused);

    if (clock::settling) {
      VLOG(3) << "Clock still not settled";
      return false;
    } else if (timers->size() == 0 ||
               timers->begin()->first > *clock::current) {
      VLOG(3) << "Clock is settled";
      return true;
    }

    VLOG(3) << "Clock is not settled";
    return false;
  }

  UNREACHABLE();
}


// TODO(benh): Introduce a Clock::time(seconds) that replaces this
// function for getting a 'Time' value.
Try<Time> Time::create(double seconds)
{
  Try<Duration> duration = Duration::create(seconds);
  if (duration.isSome()) {
    // In production code, clock::advanced will always be zero!
    return Time(duration.get() + *clock::advanced);
  } else {
    return Error("Argument too large for Time: " + duration.error());
  }
}

// The 'delay' mechanism enables you to delay a dispatch to a process
// for some specified number of seconds. Returns a Timer instance that
// can be cancelled (but it might have already executed or be
// executing concurrently).

template <typename T>
Timer delay(const Duration& duration,
            const PID<T>& pid,
            void (T::*method)())
{
  return Clock::timer(duration, [=]() {
    dispatch(pid, method);
  });
}


template <typename T>
Timer delay(const Duration& duration,
            const Process<T>& process,
            void (T::*method)())
{
  return delay(duration, process.self(), method);
}


template <typename T>
Timer delay(const Duration& duration,
            const Process<T>* process,
            void (T::*method)())
{
  return delay(duration, process->self(), method);
}


#define TEMPLATE(Z, N, DATA)                                            \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Timer delay(const Duration& duration,                                 \
              const PID<T>& pid,                                        \
              void (T::*method)(ENUM_PARAMS(N, P)),                     \
              ENUM_BINARY_PARAMS(N, A, a))                              \
  {                                                                     \
    return Clock::timer(duration, [=]() {                               \
      dispatch(pid, method, ENUM_PARAMS(N, a));                         \
    });                                                                 \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Timer delay(const Duration& duration,                                 \
              const Process<T>& process,                                \
              void (T::*method)(ENUM_PARAMS(N, P)),                     \
              ENUM_BINARY_PARAMS(N, A, a))                              \
  {                                                                     \
    return delay(duration, process.self(), method, ENUM_PARAMS(N, a));  \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Timer delay(const Duration& duration,                                 \
              const Process<T>* process,                                \
              void (T::*method)(ENUM_PARAMS(N, P)),                     \
              ENUM_BINARY_PARAMS(N, A, a))                              \
  {                                                                     \
    return delay(duration, process->self(), method, ENUM_PARAMS(N, a)); \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE
  
// Provides an abstraction over `Timer` and `Clock::timer` that
// completes a future after some duration and lets you attempt to
// discard that future.
//
// This can be used along with `loop` to create "waiting loops", for
// example:
//
//   // Wait until a file exists, check for it ever every second.
//   loop(None(),
//        []() {
//          return after(Seconds(1));
//        },
//        [=]() {
//          if (os::exists(file)) -> ControlFlow<Nothing> {
//            return Break();
//          }
//          return Continue();
//        });
inline Future<Nothing> after(const Duration& duration)
{
  std::shared_ptr<Promise<Nothing>> promise(new Promise<Nothing>());

  Timer timer = Clock::timer(duration, [=]() {
    promise->set(Nothing());
  });

  // Attempt to discard the promise if the future is discarded.
  //
  // NOTE: while the future holds a reference to the promise there is
  // no cicular reference here because even if there are no references
  // to the Future the timer will eventually fire and we'll set the
  // promise which will clear the `onDiscard` callback and delete the
  // reference to Promise.
  promise->future()
    .onDiscard([=]() {
      if (Clock::cancel(timer)) {
        promise->discard();
      }
    });

  return promise->future();
}



} // namespace process
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_CLOCK_H_