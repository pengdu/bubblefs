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
  Status() : state_(nullptr) {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(error::Code code, StringPiece msg);
  Status(error::Code code, int64_t subcode = error::NONE);
  Status(error::Code code, int64_t subcode, StringPiece msg);
  Status(error::Code code, int64_t subcode, StringPiece msg, StringPiece msg2);
  
  Status(error::Code code, const StringPiece msg, const StringPiece msg2)
      : Status(code, error::NONE, msg, msg2) {}

  /// Copy the specified status.
  Status(const Status& s);
  void operator=(const Status& s);

  // Return a success status.
  static Status OK() { return Status(); }
  
  static Status Aborted(int64_t subcode = error::NONE) { return Status(error::ABORTED, subcode); }
  static Status Aborted(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ABORTED, msg, msg2);
  }
  
  static Status AlreadyExists(int64_t subcode = error::NONE) { return Status(error::ALREADY_EXISTS, subcode); }
  static Status AlreadyExists(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ALREADY_EXISTS, msg, msg2);
  }
  
  static Status Complete(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::COMPLETE, msg, msg2);
  }
  static Status Complete(int64_t subcode = error::NONE) {
    return Status(error::COMPLETE, subcode);
  }
  
  static Status Corruption(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::CORRUPTION, msg, msg2);
  }
  static Status Corruption(int64_t subcode = error::NONE) {
    return Status(error::CORRUPTION, subcode);
  }
  
  static Status EndFile(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::ENDFILE, msg, msg2);
  }
  static Status EndFile(int64_t subcode = error::NONE) {
    return Status(error::ENDFILE, subcode);
  }
  
  static Status Incomplete(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INCOMPLETE, msg, msg2);
  }
  static Status Incomplete(int64_t subcode = error::NONE) {
    return Status(error::INCOMPLETE, subcode);
  }
  
  static Status InternalError(StringPiece message = StringPiece()) {
    return Status(error::INTERNAL, message);
  }
  
  static Status InvalidArgument(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::INVALID_ARGUMENT, msg, msg2);
  }
  static Status InvalidArgument(int64_t subcode = error::NONE) {
    return Status(error::INVALID_ARGUMENT, subcode);
  }
  
  static Status IOError(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::IOERROR, msg, msg2);
  }
  static Status IOError(int64_t subcode = error::NONE) { return Status(error::IOERROR, subcode); }
  
  static Status NetworkError(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NETWORK_ERROR, msg, msg2);
  }
  static Status NetworkError(int64_t subcode = error::NONE) { return Status(error::NETWORK_ERROR, subcode); }
  
  static Status NotFound(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NOT_FOUND, msg, msg2);
  }
  // Fast path for not found without malloc;
  static Status NotFound(int64_t subcode = error::NONE) { return Status(error::NOT_FOUND, subcode); }
  
  static Status NotSupported(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::NOT_SUPPORTED, msg, msg2);
  }
  static Status NotSupported(int64_t subcode = error::NONE) {
    return Status(error::NOT_SUPPORTED, subcode);
  }
  
  static Status Timeout(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::TIMEDOUT, msg, msg2);
  }
  static Status Timeout(int64_t subcode = error::NONE) {
    return Status(error::TIMEDOUT, subcode);
  }
  
  static Status Uuauthenticated(const StringPiece& msg, const StringPiece& msg2 = StringPiece()) {
    return Status(error::UNAUTHENTICATED, msg, msg2);
  }
  static Status Uuauthenticated(int64_t subcode = error::NONE) {
    return Status(error::UNAUTHENTICATED, subcode);
  }
  
  // Return error status of an appropriate type.
  static Status UnimplementedError(StringPiece message = StringPiece()) {
    return Status(error::UNIMPLEMENTED, message);
  }  
  
  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }
  
  // bool Is##Err() const { return code() == k##Err; }
  
  bool IsAlreadyExists() const { return code() == error::ALREADY_EXISTS; }
  
  // Returns true if the status is complete.
  bool IsComplete() const { return code() == error::COMPLETE; }
  
  // Returns true if the status indicates a Corruption error.
  bool IsCorruption() const { return code() == error::CORRUPTION; }
  
  // Returns true if the status indicates an EOF.
  bool IsEndFile() const { return code() == error::ENDFILE; }
  
  // Returns true iff the status indicates Incomplete
  bool IsIncomplete() const { return code() == error::INCOMPLETE; }
  
  // Returns true iff the status indicates an InvalidArgument error.
  bool IsInvalidArgument() const { return code() == error::INVALID_ARGUMENT; }
  
  // Returns true iff the status indicates an IOError.
  bool IsIOError() const { return code() == error::IOERROR; }
  
  bool IsNetworkError() const { return code() == error::NETWORK_ERROR; }
  
  // Returns true iff the status indicates a NotFound error.
  bool IsNotFound() const { return code() == error::NOT_FOUND; }
  
  // Returns true iff the status indicates a NotSupported error.
  bool IsNotSupported() const { return code() == error::NOT_SUPPORTED; }

  // Returns true iff the status indicates Shutdown In progress
  bool IsShutdownInProgress() const { return code() == error::SHUTDOWN_IN_PROGRESS; }
  
  // Returns true if the status is Timeout
  bool IsTimeout() const { return code() == error::TIMEDOUT; }
  
  // Returns true if the status is AuthFailed
  bool IsUnauthenticated() const { return code() == error::UNAUTHENTICATED; }

  error::Code code() const {
    return ok() ? error::OK : state_->code;
  }

  const string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }
  
  int64_t subcode() const {
    return ok() ? error::NONE : state_->subcode;
  }
  
  int err_code() const { return static_cast<int>(code()); }
  
  static Status FromCode(int err_code) {
    assert(err_code > 0 && err_code <= error::MAX_CODE);
    return Status(static_cast<error::Code>(err_code));
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
    int64_t subcode;
    string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;
  
  static const char* msgs[static_cast<int64_t>(error::MAX_INTERNAL_SUB_CODE)];

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