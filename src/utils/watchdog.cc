// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/watchdog.cc

#include "utils/watchdog.h"
#include "platform/compiler_specific.h"
#include "platform/logging.h"
#include "platform/time.h"
#include "utils/lazy_instance.h"

namespace bubblefs {
namespace core {

namespace {

// When the debugger breaks (when we alarm), all the other alarms that are
// armed will expire (also alarm).  To diminish this effect, we track any
// delay due to debugger breaks, and we *try* to adjust the effective start
// time of other alarms to step past the debugging break.
// Without this safety net, any alarm will typically trigger a host of follow
// on alarms from callers that specify old times.
  
using base::PlatformThread;
using concurrent::Lock;
using concurrent::AutoLock;
using concurrent::AutoUnlock;
using concurrent::ConditionVariable;

struct StaticData {
  // Lock for access of static data...
  Lock lock;

  // When did we last alarm and get stuck (for a while) in a debugger?
  int64_t last_debugged_alarm_time;

  // How long did we sit on a break in the debugger?
  int64_t last_debugged_alarm_delay;
};

base::LazyInstance<StaticData>::Leaky g_static_data = LAZY_INSTANCE_INITIALIZER;

}  // namespace

// Start thread running in a Disarmed state.
Watchdog::Watchdog(const int64_t duration,
                   const std::string& thread_watched_name,
                   bool enabled)
  : enabled_(enabled),
    lock_(),
    condition_variable_(&lock_),
    state_(DISARMED),
    duration_(duration),
    thread_watched_name_(thread_watched_name),
    delegate_(this) {
  if (!enabled_)
    return;  // Don't start thread, or doing anything really.
  enabled_ = PlatformThread::Create(0,  // Default stack size.
                                    &delegate_,
                                    &handle_);
  DCHECK(enabled_);
}

// Notify watchdog thread, and wait for it to finish up.
Watchdog::~Watchdog() {
  if (!enabled_)
    return;
  if (!IsJoinable())
    Cleanup();
  condition_variable_.Signal();
  PlatformThread::Join(handle_);
}

void Watchdog::Cleanup() {
  if (!enabled_)
    return;
  {
    AutoLock lock(lock_);
    state_ = SHUTDOWN;
  }
  condition_variable_.Signal();
}

bool Watchdog::IsJoinable() {
  if (!enabled_)
    return true;
  AutoLock lock(lock_);
  return (state_ == JOINABLE);
}

void Watchdog::Arm() {
  ArmAtStartTime(timeutil::get_micros());
}

void Watchdog::ArmSomeTimeDeltaAgo(const int64_t time_delta) {
  ArmAtStartTime(timeutil::get_micros() - time_delta);
}

// Start clock for watchdog.
void Watchdog::ArmAtStartTime(const int64_t start_time) {
  {
    AutoLock lock(lock_);
    start_time_ = start_time;
    state_ = ARMED;
  }
  // Force watchdog to wake up, and go to sleep with the timer ticking with the
  // proper duration.
  condition_variable_.Signal();
}

// Disable watchdog so that it won't do anything when time expires.
void Watchdog::Disarm() {
  AutoLock lock(lock_);
  state_ = DISARMED;
  // We don't need to signal, as the watchdog will eventually wake up, and it
  // will check its state and time, and act accordingly.
}

void Watchdog::Alarm() {
  DVLOG(1) << "Watchdog alarmed for " << thread_watched_name_;
}

//------------------------------------------------------------------------------
// Internal private methods that the watchdog thread uses.

void Watchdog::ThreadDelegate::ThreadMain() {
  SetThreadName();
  int64_t remaining_duration;
  StaticData* static_data = g_static_data.Pointer();
  while (1) {
    AutoLock lock(watchdog_->lock_);
    while (DISARMED == watchdog_->state_)
      watchdog_->condition_variable_.Wait();
    if (SHUTDOWN == watchdog_->state_) {
      watchdog_->state_ = JOINABLE;
      return;
    }
    DCHECK(ARMED == watchdog_->state_);
    remaining_duration = watchdog_->duration_ -
        (timeutil::get_micros() - watchdog_->start_time_);
    if (remaining_duration > 0) {
      // Spurios wake?  Timer drifts?  Go back to sleep for remaining time.
      watchdog_->condition_variable_.TimedWait(remaining_duration);
      continue;
    }
    // We overslept, so this seems like a real alarm.
    // Watch out for a user that stopped the debugger on a different alarm!
    {
      AutoLock static_lock(static_data->lock);
      if (static_data->last_debugged_alarm_time > watchdog_->start_time_) {
        // False alarm: we started our clock before the debugger break (last
        // alarm time).
        watchdog_->start_time_ += static_data->last_debugged_alarm_delay;
        if (static_data->last_debugged_alarm_time > watchdog_->start_time_)
          // Too many alarms must have taken place.
          watchdog_->state_ = DISARMED;
        continue;
      }
    }
    watchdog_->state_ = DISARMED;  // Only alarm at most once.
    int64_t last_alarm_time = timeutil::get_micros();
    {
      AutoUnlock lock(watchdog_->lock_);
      watchdog_->Alarm();  // Set a break point here to debug on alarms.
    }
    int64_t last_alarm_delay = timeutil::get_micros() - last_alarm_time;
    if (last_alarm_delay <= 2000) // TimeDelta::FromMilliseconds(2)
      continue;
    // Ignore race of two alarms/breaks going off at roughly the same time.
    AutoLock static_lock(static_data->lock);
    // This was a real debugger break.
    static_data->last_debugged_alarm_time = last_alarm_time;
    static_data->last_debugged_alarm_delay = last_alarm_delay;
  }
}

void Watchdog::ThreadDelegate::SetThreadName() const {
  std::string name = watchdog_->thread_watched_name_ + " Watchdog";
  PlatformThread::SetName(name.c_str());
  DVLOG(1) << "Watchdog active: " << name;
}

// static
void Watchdog::ResetStaticData() {
  StaticData* static_data = g_static_data.Pointer();
  AutoLock lock(static_data->lock);
  static_data->last_debugged_alarm_time = timeutil::get_micros();
  static_data->last_debugged_alarm_delay = timeutil::get_micros();
}

}  // namespace core
}  // namespace bubblefs