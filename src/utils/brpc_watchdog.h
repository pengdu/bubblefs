// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/watchdog.h

// The Watchdog class creates a second thread that can Alarm if a specific
// duration of time passes without proper attention.  The duration of time is
// specified at construction time.  The Watchdog may be used many times by
// simply calling Arm() (to start timing) and Disarm() (to reset the timer).
// The Watchdog is typically used under a debugger, where the stack traces on
// other threads can be examined if/when the Watchdog alarms.

// Some watchdogs will be enabled or disabled via command line switches. To
// facilitate such code, an "enabled" argument for the constuctor can be used
// to permanently disable the watchdog.  Disabled watchdogs don't even spawn
// a second thread, and their methods call (Arm() and Disarm()) return very
// quickly.

#ifndef BUBBLEFS_UTILS_BRPC_WATCHDOG_H_
#define BUBBLEFS_UTILS_BRPC_WATCHDOG_H_

#include <string>
#include "platform/base_export.h"
#include "platform//compiler_specific.h"
#include "platform/brpc_condition_variable.h"
#include "platform/brpc_platform_thread.h"
#include "platform/time.h"

namespace bubblefs {
namespace brpc {
  
class BASE_EXPORT Watchdog {
 public:
  // Constructor specifies how long the Watchdog will wait before alarming.
  Watchdog(const int64_t duration,
           const std::string& thread_watched_name,
           bool enabled);
  virtual ~Watchdog();

  // Notify watchdog thread to finish up. Sets the state_ to SHUTDOWN.
  void Cleanup();

  // Returns true if we state_ is JOINABLE (which indicates that Watchdog has
  // exited).
  bool IsJoinable();

  // Start timing, and alarm when time expires (unless we're disarm()ed.)
  void Arm();  // Arm  starting now.
  void ArmSomeTimeDeltaAgo(const int64_t time_delta);
  void ArmAtStartTime(const int64_t start_time);

  // Reset time, and do not set off the alarm.
  void Disarm();

  // Alarm is called if the time expires after an Arm() without someone calling
  // Disarm().  This method can be overridden to create testable classes.
  virtual void Alarm();

  // Reset static data to initial state. Useful for tests, to ensure
  // they are independent.
  static void ResetStaticData();

 private:
  class ThreadDelegate : public PlatformThread::Delegate {
   public:
    explicit ThreadDelegate(Watchdog* watchdog) : watchdog_(watchdog) {
    }
    virtual void ThreadMain() OVERRIDE;
   private:
    void SetThreadName() const;

    Watchdog* watchdog_;
  };

  enum State {ARMED, DISARMED, SHUTDOWN, JOINABLE };

  bool enabled_;

  Lock lock_;  // Mutex for state_.
  ConditionVariable condition_variable_;
  State state_;
  const int64_t duration_;  // How long after start_time_ do we alarm?
  const std::string thread_watched_name_;
  PlatformThreadHandle handle_;
  ThreadDelegate delegate_;  // Store it, because it must outlive the thread.

  int64_t start_time_;  // Start of epoch, and alarm after duration_.

  DISALLOW_COPY_AND_ASSIGN(Watchdog);
};

}  // namespace brpc
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BRPC_WATCHDOG_H_