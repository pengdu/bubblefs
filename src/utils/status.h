/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/lib/core/status.h

#ifndef BUBBLEFS_UTILS_STATUS_H_
#define BUBBLEFS_UTILS_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include "platform/logging.h"
#include "platform/macros.h"
#include "utils/error_codes.h"
#include "utils/stringpiece.h"

namespace bubblefs {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(error::Code code, StringPiece msg);
  Status(error::Code code, error::SubCode subcode = error::NONE);
  Status(error::Code code, error::SubCode subcode, StringPiece msg, StringPiece msg2);
  
  Status(Code _code, const StringPiecece& msg, const StringPiece& msg2)
      : Status(_code, error::NONE, msg, msg2) {}

  /// Copy the specified status.
  Status(const Status& s);
  void operator=(const Status& s);

  // Return a success status.
  static Status OK() { return Status(); }
  
    // Return error status of an appropriate type.
  static Status NotFound(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NOT_FOUND, msg, msg2);
  }
  // Fast path for not found without malloc;
  static Status NotFound(error::SubCode msg = error::NONE) { return Status(error::NOT_FOUND, msg); }
  
  static Status InvalidArgument(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INVALID_ARGUMENT, msg, msg2);
  }
  static Status InvalidArgument(error::SubCode msg = error::NONE) {
    return Status(error::INVALID_ARGUMENT, msg);
  }
  
  static Status Aborted(error::SubCode msg = error::NONE) { return Status(error::ABORTED, msg); }
  static Status Aborted(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ABORTED, msg, msg2);
  }

  static Status Corruption(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::CORRUPTION, msg, msg2);
  }
  static Status Corruption(error::SubCode msg = error::NONE) {
    return Status(error::CORRUPTION, msg);
  }

  static Status IOError(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::IOERROR, msg, msg2);
  }
  static Status IOError(error::SubCode msg =  error::NONE) { return Status(error::IOERROR, msg); }

  static Status Incomplete(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INCOMPLETE, msg, msg2);
  }
  static Status Incomplete(error::SubCode msg = error::NONE) {
    return Status(error::INCOMPLETE, msg);
  }

  static Status ShutdownInProgress(error::SubCode msg = error::NONE) {
    return Status(error::SHUTDOWN_IN_PROGRESS, msg);
  }
  static Status ShutdownInProgress(const StringPiece& msg,
                                   const StringPiece& msg2 = StringPiece()) {
    return Status(error::SHUTDOWN_IN_PROGRESS, msg, msg2);
  }

  static Status Expired(error::SubCode msg = error::NONE) { return Status(error::EXPIRED, msg); }
  static Status Expired(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::EXPIRED, msg, msg2);
  }

  static Status TryAgain(error::SubCode msg = error::NONE) { return Status(error::TRY_AGAIN, msg); }
  static Status TryAgain(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::TRY_AGAIN, msg, msg2);
  }
  
  static Status NotSupported(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NOT_SUPPORTED, msg, msg2);
  }
  static Status NotSupported(error::SubCode msg = error::NONE) {
    return Status(error::NOT_SUPPORTED, msg);
  }
  
  static Status TimedOut(error::SubCode msg = error::NONE) { return Status(error::TIMEDOUT, msg); }
  static Status TimedOut(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::TIMEDOUT, msg, msg2);
  }
  
  static Status Busy(error::SubCode msg = error::NONE) { return Status(error::BUSY, msg); }
  static Status Busy(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::BUSY, msg, msg2);
  }

  static Status NoSpace() { return Status(error::IOERROR, error::NOSPACE); }
  static Status NoSpace(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::IOERROR, error::NOSPACE, msg, msg2);
  }

  static Status MemoryLimit() { return Status(error::ABORTED, error::MEMORY_LIMIT); }
  static Status MemoryLimit(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ABORTED, error::MEMORY_LIMIT, msg, msg2);
  }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }
  
   // Returns true iff the status indicates a NotFound error.
  bool IsNotFound() const { return code() == error::NOT_FOUND; }

  // Returns true iff the status indicates a Corruption error.
  bool IsCorruption() const { return code() == error::CORRUPTION; }

  // Returns true iff the status indicates a NotSupported error.
  bool IsNotSupported() const { return code() == error::NOT_SUPPORTED; }

  // Returns true iff the status indicates an InvalidArgument error.
  bool IsInvalidArgument() const { return code() == error::INVALID_ARGUMENT; }

  // Returns true iff the status indicates an IOError.
  bool IsIOError() const { return code() == error::IOERROR; }

  // Returns true iff the status indicates Incomplete
  bool IsIncomplete() const { return code() == error::INCOMPLETE; }

  // Returns true iff the status indicates Shutdown In progress
  bool IsShutdownInProgress() const { return code() == error::SHUTDOWN_IN_PROGRESS; }

  bool IsTimedOut() const { return code() == error::TIMEDOUT; }

  bool IsAborted() const { return code() == error::ABORTED; }

  bool IsLockLimit() const {
    return code() == error::ABORTED && subcode() == error::LOCK_LIMIT;
  }

  // Returns true iff the status indicates that a resource is Busy and
  // temporarily could not be acquired.
  bool IsBusy() const { return code() == error::BUSY; }

  bool IsDeadlock() const { return code() == error::BUSY && subcode() == error::DEAD_LOCK; }

  // Returns true iff the status indicated that the operation has Expired.
  bool IsExpired() const { return code() == error::EXPIRED; }

  // Returns true iff the status indicates a TryAgain error.
  // This usually means that the operation failed, but may succeed if
  // re-attempted.
  bool IsTryAgain() const { return code() == error::TRY_AGAIN; }

  // Returns true iff the status indicates a NoSpace error
  // This is caused by an I/O error returning the specific "out of space"
  // error condition. Stricto sensu, an NoSpace error is an I/O error
  // with a specific subcode, enabling users to take the appropriate action
  // if needed
  bool IsNoSpace() const {
    return (code() == error::IOERROR) && (subcode() == error::NOSPACE);
  }

  // Returns true iff the status indicates a memory limit error.  There may be
  // cases where we limit the memory used in certain operations (eg. the size
  // of a write batch) in order to avoid out of memory exceptions.
  bool IsMemoryLimit() const {
    return (code() == error::ABORTED) && (subcode() == error::MEMORY_LIMIT);
  }

  error::Code code() const {
    return ok() ? error::OK : state_->code;
  }

  const string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }
  
  error::SubCode subcode() const {
    return ok() ? error::NONE : state_->subcode;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  static const string& empty_string();
  struct State {
    error::Code code;
    string msg;
    error::SubCode subcode;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

inline void Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
}

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern string* TfCheckOpHelperOutOfLine(
    const Status& v, const char* msg);

inline string* TfCheckOpHelper(Status v,
                               const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                  \
  while (auto _result = TfCheckOpHelper(val, #val)) \
    LOG(level) << *(_result)

#define TF_CHECK_OK(val)  TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::bubblefs::Status::OK() == (val))) LOG(FATAL)
#endif

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_STATUS_H_