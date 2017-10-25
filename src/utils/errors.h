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

// tensorflow/tensorflow/core/lib/core/errors.h

#ifndef BUBBLEFS_UTILS_ERRORS_H_
#define BUBBLEFS_UTILS_ERRORS_H_

#include <stdarg.h>
#include "platform/logging.h"
#include "platform/macros.h"
#include "utils/status.h"
#include "utils/strcat.h"

namespace bubblefs {
namespace errors {
  
typedef ::bubblefs::error::Code Code;

// Append some context to an error message.  Each time we append
// context put it on a new line, since it is possible for there
// to be several layers of additional context.
template <typename... Args>
void AppendToMessage(Status* status, Args... args) {
  *status = Status(
      status->code(),
      strings::StrCat(status->error_message(), "\n\t", args...));
}

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(expr)                         \
  do {                                                   \
    const ::bubblefs::Status _status = (expr);         \
    if (PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define TF_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)                  \
  do {                                                              \
    ::bubblefs::Status _status = (expr);                          \
    if (PREDICT_FALSE(!_status.ok())) {                          \
      ::bubblefs::errors::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                               \
    }                                                               \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CONST)                                       \
  template <typename... Args>                                            \
  Status FUNC(Args... args) {                              \
    return Status(error::CONST, strings::StrCat(args...)); \
  }                                                                      \
  inline bool Is##FUNC(const Status& status) {             \
    return status.code() == error::CONST;                  \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Corruption, CORRUPTION)
DECLARE_ERROR(IOError, IOERROR)
DECLARE_ERROR(InComplete, INCOMPLETE)
DECLARE_ERROR(ShutDownInProgress, SHUTDOWN_IN_PROGRESS)
DECLARE_ERROR(Expired, EXPIRED)
DECLARE_ERROR(OpTryAgain, OP_TRY_AGAIN)
DECLARE_ERROR(NotSupported, NOT_SUPPORTED)
DECLARE_ERROR(TimedOut, TIMEDOUT)
DECLARE_ERROR(Busy, BUSY)
DECLARE_ERROR(Unanthenticated, UNAUTHENTICATED)
DECLARE_ERROR(NetworkError, NETWORK_ERROR)
DECLARE_ERROR(EndFile, ENDFILE)
DECLARE_ERROR(Complete, COMPLETE)
// custom
DECLARE_ERROR(UserError, USER_ERROR)

#undef DECLARE_ERROR

// The CanonicalCode() for non-errors.
using error::OK;

}  // namespace errors
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_ERRORS_H_