// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/util/base64/base64.h

#ifndef BUBBLEFS_UTILS_VOYAGER_BASE64_H_
#define BUBBLEFS_UTILS_VOYAGER_BASE64_H_

#include <string>

namespace bubblefs {
namespace myvoyager {

// Encodes the input string in base64. Returns true if successful and false
// otherwise. The output string is only modified if successful.
bool Base64Encode(const std::string& input, std::string* output);

// Decodes the base64 input string. Returns true if successful and false
// otherwise. The output string is only modified if successful.
bool Base64Decode(const std::string& input, std::string* output);

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_BASE64_H_