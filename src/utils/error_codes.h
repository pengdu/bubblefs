// tensorflow/tensorflow/core/lib/core/error_codes.proto

// Sometimes multiple error codes may apply.  Services should return
// the most specific error code that applies.  For example, prefer
// OUT_OF_RANGE over FAILED_PRECONDITION if both codes apply.
// Similarly prefer NOT_FOUND or ALREADY_EXISTS over FAILED_PRECONDITION.

#ifndef BUBBLEFS_UTILS_ERROR_CODES_H_
#define BUBBLEFS_UTILS_ERROR_CODES_H_

namespace bubblefs {
namespace error {

enum Code {
  // Not an error; returned on success
  OK = 0,

  // The operation was cancelled (typically by the caller).
  CANCELLED = 1,

  // Unknown error.  An example of where this error may be returned is
  // if a Status value received from another address space belongs to
  // an error-space that is not known in this address space.  Also
  // errors raised by APIs that do not return enough error information
  // may be converted to this error.
  UNKNOWN = 2,

  // Client specified an invalid argument.  Note that this differs
  // from FAILED_PRECONDITION.  INVALID_ARGUMENT indicates arguments
  // that are problematic regardless of the state of the system
  // (e.g., a malformed file name).
  INVALID_ARGUMENT = 3,

  // Deadline expired before operation could complete.  For operations
  // that change the state of the system, this error may be returned
  // even if the operation has completed successfully.  For example, a
  // successful response from a server could have been delayed long
  // enough for the deadline to expire.
  DEADLINE_EXCEEDED = 4,

  // Some requested entity (e.g., file or directory) was not found.
  // For privacy reasons, this code *may* be returned when the client
  // does not have the access right to the entity.
  NOT_FOUND = 5,

  // Some entity that we attempted to create (e.g., file or directory)
  // already exists.
  ALREADY_EXISTS = 6,

  // The caller does not have permission to execute the specified
  // operation.  PERMISSION_DENIED must not be used for rejections
  // caused by exhausting some resource (use RESOURCE_EXHAUSTED
  // instead for those errors).  PERMISSION_DENIED must not be
  // used if the caller can not be identified (use UNAUTHENTICATED
  // instead for those errors).
  PERMISSION_DENIED = 7,

  // Some resource has been exhausted, perhaps a per-user quota, or
  // perhaps the entire file system is out of space.
  RESOURCE_EXHAUSTED = 8,

  // Operation was rejected because the system is not in a state
  // required for the operation's execution.  For example, directory
  // to be deleted may be non-empty, an rmdir operation is applied to
  // a non-directory, etc.
  //
  // A litmus test that may help a service implementor in deciding
  // between FAILED_PRECONDITION, ABORTED, and UNAVAILABLE:
  //  (a) Use UNAVAILABLE if the client can retry just the failing call.
  //  (b) Use ABORTED if the client should retry at a higher-level
  //      (e.g., restarting a read-modify-write sequence).
  //  (c) Use FAILED_PRECONDITION if the client should not retry until
  //      the system state has been explicitly fixed.  E.g., if an "rmdir"
  //      fails because the directory is non-empty, FAILED_PRECONDITION
  //      should be returned since the client should not retry unless
  //      they have first fixed up the directory by deleting files from it.
  //  (d) Use FAILED_PRECONDITION if the client performs conditional
  //      REST Get/Update/Delete on a resource and the resource on the
  //      server does not match the condition. E.g., conflicting
  //      read-modify-write on the same resource.
  FAILED_PRECONDITION = 9,

  // The operation was aborted, typically due to a concurrency issue
  // like sequencer check failures, transaction aborts, etc.
  //
  // See litmus test above for deciding between FAILED_PRECONDITION,
  // ABORTED, and UNAVAILABLE.
  ABORTED = 10,

  // Operation tried to iterate past the valid input range.  E.g., seeking or
  // reading past end of file.
  //
  // Unlike INVALID_ARGUMENT, this error indicates a problem that may
  // be fixed if the system state changes. For example, a 32-bit file
  // system will generate INVALID_ARGUMENT if asked to read at an
  // offset that is not in the range [0,2^32-1], but it will generate
  // OUT_OF_RANGE if asked to read from an offset past the current
  // file size.
  //
  // There is a fair bit of overlap between FAILED_PRECONDITION and
  // OUT_OF_RANGE.  We recommend using OUT_OF_RANGE (the more specific
  // error) when it applies so that callers who are iterating through
  // a space can easily look for an OUT_OF_RANGE error to detect when
  // they are done.
  OUT_OF_RANGE = 11,

  // Operation is not implemented or not supported/enabled in this service.
  UNIMPLEMENTED = 12,

  // Internal errors.  Means some invariants expected by underlying
  // system has been broken.  If you see one of these errors,
  // something is very broken.
  INTERNAL = 13,

  // The service is currently unavailable.  This is a most likely a
  // transient condition and may be corrected by retrying with
  // a backoff.
  //
  // See litmus test above for deciding between FAILED_PRECONDITION,
  // ABORTED, and UNAVAILABLE.
  UNAVAILABLE = 14,

  // Unrecoverable data loss or corruption.
  DATA_LOSS = 15,
  
  CORRUPTION = 16,
  
  IOERROR = 17,
  
  INCOMPLETE = 18,
  
  SHUTDOWN_IN_PROGRESS = 19,
  
  EXPIRED = 20,
  
  TRY_AGAIN = 21,
  
  NOT_SUPPORTED = 22,
  
  TIMEDOUT = 23,
  
  BUSY = 24,
  
  // The request does not have valid authentication credentials for the
  // operation.
  UNAUTHENTICATED = 25,

  // An extra enum entry to prevent people from writing code that
  // fails to compile when a new code is added.
  //
  // Nobody should ever reference this enumeration entry. In particular,
  // if you write C++ code that switches on this enumeration, add a default:
  // case instead of a case that mentions this enumeration entry.
  //
  // Nobody should rely on the value (currently 20) listed here.  It
  // may change in the future.
  DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 30
}; // enum Code

enum SubCode {
  NONE = 0,
  MUTEX_TIMEOUT = 1,
  LOCK_TIMEOUT = 2,
  LOCK_LIMIT = 3,
  NOSPACE = 4,
  DEAD_LOCK = 5,
  STALE_FILE = 6,
  MEMORY_LIMIT = 7,
  
  MAX_SUB_CODE
}; // enum SubCode

} // namespace error
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_ERROR_CODES_H_