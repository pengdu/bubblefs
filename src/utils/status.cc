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

// tensorflow/tensorflow/core/lib/core/status.cc

#include "utils/status.h"
#include <assert.h>
#include <stdio.h>

namespace bubblefs {

Status::Status(error::Code code, StringPiece msg) {
  assert(code != error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = msg.ToString();
  state_->subcode = error::NONE;
}

Status::Status(error::Code code, error::SubCode subcode) {
  assert(code != error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->subcode = subcode;
}

Status::Status(error::Code code, error::SubCode subcode, StringPiece msg, StringPiece msg2) {
  assert(code != error::OK);
  assert(subcode != error::MAX_SUB_CODE);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->subcode = error::NONE;
  const size_t len1 = msg.size();
  const size_t len2 = msg2.size();
  const size_t size = len1 + (len2 ? (2 + len2) : 0);
  char* const result = new char[size + 1];  // +1 for null terminator
  memcpy(result, msg.data(), len1);
  if (len2) {
    result[len1] = ':';
    result[len1 + 1] = ' ';
    memcpy(result + len1 + 2, msg2.data(), len2);
  }
  result[size] = '\0';  // null terminator for C style string
  state_ = result;
  state_->subcode = error::NONE;
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case error::CANCELLED:
        type = "Cancelled";
        break;
      case error::UNKNOWN:
        type = "Unknown";
        break;
      case error::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case error::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case error::NOT_FOUND:
        type = "Not found";
        break;
      case error::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case error::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case error::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case error::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case error::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case error::ABORTED:
        type = "Aborted";
        break;
      case error::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case error::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case error::INTERNAL:
        type = "Internal";
        break;
      case error::UNAVAILABLE:
        type = "Unavailable";
        break;
      case error::DATA_LOSS:
        type = "Data loss";
        break;
      case error::CORRUPTION:
        type = "Corruption";
        break;
      case error::IOERROR:
        type = "IO error";
        break;
      case error::INCOMPLETE:
        type = "Result incomplete";
        break;  
      case error::SHUTDOWN_IN_PROGRESS:
        type = "Shutdown in progress";
        break;
      case error::EXPIRED:
        type = "Operation expired";
        break;
      case error::TRY_AGAIN:
        type = "Operation try again";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    string result(type);
    result += ": ";
    error::SubCode subcode = error::NONE;
    if (subcode != error::NONE) {
      memset(tmp, 0, sizeof(tmp));
      snprintf(tmp, sizeof(tmp), "Subode(%d)",
                 static_cast<int>(subcode));
      result.append(tmp);
    }
    result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

string* TfCheckOpHelperOutOfLine(const Status& v,
                                 const char* msg) {
  string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new string(r);
}

}  // namespace bubblefs
