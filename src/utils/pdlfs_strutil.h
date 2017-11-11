/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/strutil.h

#ifndef BUBBLEFS_UTILS_PDLFS_STRUTIL_H_
#define BUBBLEFS_UTILS_PDLFS_STRUTIL_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "utils/stringpiece.h"

namespace bubblefs {
namespace mypdlfs {

using Slice = StringPiece;

// Append a human-readable printout of "num" to *str
extern void AppendSignedNumberTo(std::string* str,
                                 int64_t num);               // Signed number
extern void AppendNumberTo(std::string* str, uint64_t num);  // Unsigned number

// Append a human-readable printout of "value" to *str.
// Escapes any non-printable characters found in "value".
extern void AppendEscapedStringTo(std::string* str, const Slice& value);

// Return a human-readable printout of "num"
extern std::string NumberToString(uint64_t num);

// Return a human-readable version of "value".
// Escapes any non-printable characters found in "value".
extern std::string EscapeString(const Slice& value);

// Parse a human-readable number from "*in" into *value.  On success,
// advances "*in" past the consumed number and sets "*val" to the
// numeric value.  Otherwise, returns false and leaves *in in an
// unspecified state.
extern bool ConsumeDecimalNumber(Slice* in, uint64_t* val);

// Split a string into a list of substrings using a specified delimiter.
// If max_splits is non-negative, the number of splits won't exceed it.
// Return the size of the resulting array.
extern size_t SplitString(std::vector<std::string>* result, const char* value,
                          char delim = ';', int max_splits = -1);

// Parse a human-readable text to a long int value.
// Return true if successfully parsed and false otherwise.
extern bool ParsePrettyNumber(const Slice& value, uint64_t* val);

// Parse a human-readable text to a boolean value.
// Return true if successfully parsed and false otherwise.
extern bool ParsePrettyBool(const Slice& value, bool* val);

// Convert a potentially large size number to a human-readable text.
extern std::string PrettySize(uint64_t num);

}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PDLFS_STRUTIL_H_