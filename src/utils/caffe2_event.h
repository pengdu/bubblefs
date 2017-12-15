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

// caffe2/caffe2/core/event.h
// caffe2/caffe2/core/event.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_EVENT_H_
#define BUBBLEFS_UTILS_CAFFE2_EVENT_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/caffe2_proto_caffe2.h"

namespace bubblefs {
namespace mycaffe2 {

constexpr int MaxDeviceTypes = DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
class Event;

enum EventStatus {
  EVENT_INITIALIZED = 0,
  EVENT_SCHEDULED = 1,
  EVENT_SUCCESS = 2,
  EVENT_FAILED = 3,
};

// For the following functions, void* shall be interpreted as the corresponding
// context object corresponding to the device type associated with the
// functions.

// Initializes event
typedef void (*EventCreateFunction)(const DeviceOption& option, Event*);

// Called on event to signal that CPU part of operation is finished,
// Optionally accepts error message from CPU part.
// Should be called no more than once per event
typedef void (*EventRecordFunction)(Event*, const void*, const char*);

// Waits and returns as soon as possible in order schedule next operation,
// e.g. for CUDA->CUDA waits only for CPU part of CUDA op,
// for CUDA->CPU waits till the CUDA op is fully completed.
// Prepares context to synchronize device part of operation.
// Can be called concurrently from multiple threads
typedef void (*EventWaitFunction)(const Event*, void*);

// Waits till operation is fully finished,
// can be called concurrently from multiple threads
typedef void (*EventFinishFunction)(const Event*);

// Queries current status of operation,
// can be called concurrently from multiple threads
typedef EventStatus (*EventQueryFunction)(const Event*);
typedef const std::string& (*EventErrorMessageFunction)(const Event*);
typedef void (*EventSetFinishedFunction)(const Event*, const char*);
typedef void (*EventResetFunction)(Event*);

class Event {
 public:
  explicit Event(const DeviceOption& option)
      : event_(), type_(option.device_type()), option_(option) {
    PANIC_ENFORCE_LT(type_, MaxDeviceTypes);
    PANIC_ENFORCE(event_creator_[type_], "event_creator_[type] is NULL");
    event_creator_[type_](option, this);
  }

  // Nothing needs to be done in the destructor, as the event creator should
  // set the proper destruction process for the unique_ptr.
  ~Event() {}

  void Record(
      int recorder_type,
      const void* context,
      const char* err_msg = nullptr) {
    PANIC_ENFORCE_EQ(recorder_type, type_) // "You are trying to record with a wrong device type."
    PANIC_ENFORCE(event_recorder_[recorder_type]);
    event_recorder_[recorder_type](this, context, err_msg);
  }

  void Wait(int waiter_type, void* context) const {
    PANIC_ENFORCE(event_waiter_[waiter_type][type_], "event_waiter_[waiter_type][type_] is NULL");
    event_waiter_[waiter_type][type_](this, context);
  }

  void Finish() const {
    PANIC_ENFORCE(event_finisher_[type_], "event_finisher_[type_] is NULL");
    event_finisher_[type_](this);
  }

  EventStatus Query() const {
    PANIC_ENFORCE(event_querier_[type_]);
    return event_querier_[type_](this);
  }

  const std::string& ErrorMessage() const {
    PANIC_ENFORCE(event_err_msg_getter_[type_], "event_err_msg_getter_[type_] is NULL");
    return event_err_msg_getter_[type_](this);
  }

  void Reset() {
    PANIC_ENFORCE(event_resetter_[type_], "event_resetter_[type_] is NULL");
    event_resetter_[type_](this);
  }

  const DeviceOption& GetDeviceOption() const {
    return option_;
  }

  bool IsScheduled() const {
    return Query() == EventStatus::EVENT_SCHEDULED;
  }

  bool IsFinished() const {
    auto status = Query();
    return status == EventStatus::EVENT_SUCCESS ||
        status == EventStatus::EVENT_FAILED;
  }

  void SetFinished(const char* err_msg = nullptr) {
    PANIC_ENFORCE(event_finished_setter_[type_], "event_finished_setter_[type_] is NULL");
    return event_finished_setter_[type_](this, err_msg);
  }

  // If parent op has succeeded, then we can run any child op;
  // If parent op is in scheduled state, we need to check that:
  //  - child op supports async scheduling
  //  - there's a way to setup synchronization between async parent and
  //    child - both child and parent should use the same type of device,
  //    non-blocking synchronization between different device types is not
  //    supported
  // If parent op is in another state (initialized or failed) then scheduling
  // is not possible
  bool CanSchedule(const Event& child_event, bool supports_async) const {
    return CanSchedule(type_, Query(), child_event.GetType(), supports_async);
  }

  static bool CanSchedule(
      int parent_type,
      EventStatus parent_status,
      int child_type,
      bool child_supports_async) {
    if (parent_status == EventStatus::EVENT_SUCCESS) {
      return true;
    }
    if (parent_status == EventStatus::EVENT_SCHEDULED) {
      return (parent_type == child_type) && child_supports_async;
    }
    return false;
  }

  int GetType() const {
    return type_;
  }

  // event_ is going to be accessed by the EventCreate/Record/Wait/Finish
  // functions, but one should not use it outside the own Event functionalities.
  // In the future we may move it to a private member.
  std::shared_ptr<void> event_;

 private:
  int type_;
  DeviceOption option_;

  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];

  static EventQueryFunction event_querier_[MaxDeviceTypes];
  static EventErrorMessageFunction event_err_msg_getter_[MaxDeviceTypes];
  static EventSetFinishedFunction event_finished_setter_[MaxDeviceTypes];
  static EventResetFunction event_resetter_[MaxDeviceTypes];

  template <int d>
  friend struct EventCreateFunctionRegisterer;
  template <int d>
  friend struct EventRecordFunctionRegisterer;
  template <int w, int d>
  friend struct EventWaitFunctionRegisterer;
  template <int d>
  friend struct EventFinishFunctionRegisterer;

  template <int d>
  friend struct EventQueryFunctionRegisterer;
  template <int d>
  friend struct EventErrorMessageFunctionRegisterer;
  template <int d>
  friend struct EventSetFinishedFunctionRegisterer;
  template <int d>
  friend struct EventResetFunctionRegisterer;
};

template <int d>
struct EventCreateFunctionRegisterer {
  explicit EventCreateFunctionRegisterer(EventCreateFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_creator_[d] = f;
  }
};
#define REGISTER_EVENT_CREATE_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventCreateFunctionRegisterer<d> g_event_create_##d(f); \
  }

template <int d>
struct EventRecordFunctionRegisterer {
  explicit EventRecordFunctionRegisterer(EventRecordFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_recorder_[d] = f;
  }
};
#define REGISTER_EVENT_RECORD_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventRecordFunctionRegisterer<d> g_event_record_##d(f); \
  }

template <int waiter_type, int event_type>
struct EventWaitFunctionRegisterer {
  explicit EventWaitFunctionRegisterer(EventWaitFunction f) {
    static_assert(waiter_type < MaxDeviceTypes, "");
    static_assert(event_type < MaxDeviceTypes, "");
    Event::event_waiter_[waiter_type][event_type] = f;
  }
};
#define REGISTER_EVENT_WAIT_FUNCTION(w, d, f)                         \
  namespace {                                                         \
  static EventWaitFunctionRegisterer<w, d> g_event_wait_##w##_##d(f); \
  }

template <int d>
struct EventQueryFunctionRegisterer {
  explicit EventQueryFunctionRegisterer(EventQueryFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_querier_[d] = f;
  }
};
#define REGISTER_EVENT_QUERY_FUNCTION(d, f)                    \
  namespace {                                                  \
  static EventQueryFunctionRegisterer<d> g_event_query_##d(f); \
  }

template <int d>
struct EventErrorMessageFunctionRegisterer {
  explicit EventErrorMessageFunctionRegisterer(EventErrorMessageFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_err_msg_getter_[d] = f;
  }
};
#define REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(d, f)                     \
  namespace {                                                           \
  static EventErrorMessageFunctionRegisterer<d> g_event_err_msg_##d(f); \
  }

template <int d>
struct EventSetFinishedFunctionRegisterer {
  explicit EventSetFinishedFunctionRegisterer(EventSetFinishedFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_finished_setter_[d] = f;
  }
};
#define REGISTER_EVENT_SET_FINISHED_FUNCTION(d, f)                          \
  namespace {                                                               \
  static EventSetFinishedFunctionRegisterer<d> g_event_set_finished_##d(f); \
  }

template <int d>
struct EventFinishFunctionRegisterer {
  explicit EventFinishFunctionRegisterer(EventFinishFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_finisher_[d] = f;
  }
};
#define REGISTER_EVENT_FINISH_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventFinishFunctionRegisterer<d> g_event_finish_##d(f); \
  }

template <int d>
struct EventResetFunctionRegisterer {
  explicit EventResetFunctionRegisterer(EventResetFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_resetter_[d] = f;
  }
};
#define REGISTER_EVENT_RESET_FUNCTION(d, f)                    \
  namespace {                                                  \
  static EventResetFunctionRegisterer<d> g_event_reset_##d(f); \
  }

EventCreateFunction Event::event_creator_[MaxDeviceTypes];
EventRecordFunction Event::event_recorder_[MaxDeviceTypes];
EventWaitFunction Event::event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
EventFinishFunction Event::event_finisher_[MaxDeviceTypes];

EventQueryFunction Event::event_querier_[MaxDeviceTypes];
EventErrorMessageFunction Event::event_err_msg_getter_[MaxDeviceTypes];
EventSetFinishedFunction Event::event_finished_setter_[MaxDeviceTypes];
EventResetFunction Event::event_resetter_[MaxDeviceTypes];

namespace {
const std::string kNoError = "No error";
} // namespace

struct CPUEventWrapper {
  explicit CPUEventWrapper(const DeviceOption& option)
      : status_(EventStatus::EVENT_INITIALIZED) {
    PANIC_ENFORCE(
        option.device_type() == CPU || option.device_type() == MKLDNN,
        "Expected CPU/MKLDNN device type");
  }
  ~CPUEventWrapper() {}

  std::mutex mutex_;
  std::condition_variable cv_completed_;
  std::atomic<int> status_;
  std::string err_msg_;
};

void EventCreateCPU(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<CPUEventWrapper>(option);
}

void EventRecordCPU(
    Event* event,
    const void* /* unused */,
    const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  // Possible state changes:
  //  INITIALIZED -> SCHEDULED or SUCCESS/FAILED
  //  SCHEDULED -> SUCCESS/FAILED
  //  SUCCESS/FAILED - terminal, no further changes to status_/err_msg_

  PANIC_ENFORCE(
      wrapper->status_ == EventStatus::EVENT_INITIALIZED,
      "Calling Record multiple times");

  if (!err_msg) {
    wrapper->status_ = EventStatus::EVENT_SCHEDULED;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::EVENT_FAILED;
    wrapper->cv_completed_.notify_all();
  }
}

void EventFinishCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  while (wrapper->status_ != EventStatus::EVENT_SUCCESS &&
         wrapper->status_ != EventStatus::EVENT_FAILED) {
    wrapper->cv_completed_.wait(lock);
  }
}

void EventWaitCPUCPU(const Event* event, void* /* context */) {
  EventFinishCPU(event);
}

EventStatus EventQueryCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::EVENT_FAILED) {
    // Failed is a terminal state, not synchronizing,
    // err_msg_ should not be changed anymore
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedCPU(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  PANIC_ENFORCE(
      wrapper->status_ == EventStatus::EVENT_INITIALIZED ||
          wrapper->status_ == EventStatus::EVENT_SCHEDULED,
      "Calling SetFinished on finished event");

  if (!err_msg) {
    wrapper->status_ = EventStatus::EVENT_SUCCESS;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::EVENT_FAILED;
  }
  wrapper->cv_completed_.notify_all();
}

void EventResetCPU(Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  wrapper->status_ = EventStatus::EVENT_INITIALIZED;
  wrapper->err_msg_ = "";
}

REGISTER_EVENT_CREATE_FUNCTION(CPU, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(CPU, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, CPU, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(CPU, EventFinishCPU);

REGISTER_EVENT_QUERY_FUNCTION(CPU, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(CPU, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(CPU, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(CPU, EventResetCPU);
  
} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_EVENT_H_