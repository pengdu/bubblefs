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
class WARN_UNUSED_RESULT Status;
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
  
  Status(error::Code code, const StringPiece& msg, const StringPiece& msg2)
      : Status(code, error::NONE, msg, msg2) {}

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
  static Status NotFound(error::SubCode subcode = error::NONE) { return Status(error::NOT_FOUND, subcode); }
  
  static Status InvalidArgument(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INVALID_ARGUMENT, msg, msg2);
  }
  static Status InvalidArgument(error::SubCode subcode = error::NONE) {
    return Status(error::INVALID_ARGUMENT, subcode);
  }
  
  static Status Aborted(error::SubCode subcode = error::NONE) { return Status(error::ABORTED, subcode); }
  static Status Aborted(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ABORTED, msg, msg2);
  }

  static Status IOError(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::IOERROR, msg, msg2);
  }
  static Status IOError(error::SubCode subcode =  error::NONE) { return Status(error::IOERROR, subcode); }
  
  static Status NotSupported(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NOT_SUPPORTED, msg, msg2);
  }
  static Status NotSupported(error::SubCode subcode = error::NONE) {
    return Status(error::NOT_SUPPORTED, subcode);
  }
  
  static Status Incomplete(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INCOMPLETE, msg, msg2);
  }
  static Status Incomplete(error::SubCode subcode = error::NONE) {
    return Status(error::INCOMPLETE, subcode);
  }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }
  
  // Returns true iff the status indicates a NotFound error.
  bool IsNotFound() const { return code() == error::NOT_FOUND; }
  
  // Returns true iff the status indicates a NotSupported error.
  bool IsNotSupported() const { return code() == error::NOT_SUPPORTED; }

  // Returns true iff the status indicates an InvalidArgument error.
  bool IsInvalidArgument() const { return code() == error::INVALID_ARGUMENT; }
  
  // Returns true iff the status indicates an IOError.
  bool IsIOError() const { return code() == error::IOERROR; }

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