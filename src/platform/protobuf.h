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

// tensorflow/tensorflow/core/platform/protobuf.h
// tensorflow/tensorflow/core/platform/default/protobuf.h

#ifndef BUBBLEFS_PLATFORM_PROTOBUF_H_
#define BUBBLEFS_PLATFORM_PROTOBUF_H_

#include "platform/platform.h"
#include "platform/types.h"

// Import whatever namespace protobuf comes from into the
// ::bubblefs::protobuf namespace.
//
// Code should use the ::bubblefs::protobuf namespace to
// refer to all protobuf APIs.

#include "google/protobuf/arena.h"
#include "google/protobuf/compiler/importer.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"

namespace bubblefs {

namespace protobuf = ::google::protobuf;
using protobuf_int64 = ::google::protobuf::int64;
using protobuf_uint64 = ::google::protobuf::uint64;
extern const char* kProtobufInt64Typename;
extern const char* kProtobufUint64Typename;
  
// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                         const string& serialized);
bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size);

// Returns the string value for the value of a string or bytes protobuf field.
inline const string& ProtobufStringToString(const string& s) { return s; }

// Set <dest> to <src>. Swapping is allowed, as <src> does not need to be
// preserved.
inline void SetProtobufStringSwapAllowed(string* src, string* dest) {
  dest->swap(*src);
}

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_PROTOBUF_H_