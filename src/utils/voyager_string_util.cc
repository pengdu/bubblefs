// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/util/string_util.cc
// voyager/voyager/util/stringprintf.cc

#include "utils/voyager_string_util.h"
#include <assert.h>
#include <string.h>
#include <iterator>

namespace bubblefs {
namespace myvoyager {

void StripString(std::string* s, const char* remove, char replacewith) {
  const char* str_start = s->c_str();
  const char* str = str_start;
  for (str = strpbrk(str, remove); str != nullptr;
       str = strpbrk(str + 1, remove)) {
    (*s)[str - str_start] = replacewith;
  }
}

void StripWhitespace(std::string* str) {
  int str_length = static_cast<int>(str->length());

  // Strip off leading whitespace.
  int first = 0;
  while (first < str_length && ascii_isspace(str->at(first))) {
    ++first;
  }
  // If entire string is white space.
  if (first == str_length) {
    str->clear();
    return;
  }
  if (first > 0) {
    str->erase(0, first);
    str_length -= first;
  }

  // Strip off trailing whitespace.
  int last = str_length - 1;
  while (last >= 0 && ascii_isspace(str->at(last))) {
    --last;
  }
  if (last != (str_length - 1) && last >= 0) {
    str->erase(last + 1, std::string::npos);
  }
}

void StringReplace(const std::string& s, const std::string& oldsub,
                   const std::string& newsub, bool replace_all,
                   std::string* res) {
  if (oldsub.empty()) {
    res->append(s);  // if empty, append the given string.
    return;
  }
  std::string::size_type start_pos = 0;
  std::string::size_type pos;
  do {
    pos = s.find(oldsub, start_pos);
    if (pos == std::string::npos) {
      break;
    }
    res->append(s, start_pos, pos - start_pos);
    res->append(newsub);
    start_pos = pos + oldsub.size();  // start searching again after the "old"
  } while (replace_all);
  res->append(s, start_pos, s.length() - start_pos);
}

std::string StringReplace(const std::string& s, const std::string& oldsub,
                          const std::string& newsub, bool replace_all) {
  std::string result;
  StringReplace(s, oldsub, newsub, replace_all, &result);
  return result;
}

template <typename ITR>
static inline void SplitStringToIteratorUsing(const std::string& full,
                                              const char* delim, ITR& result) {
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char* p = full.data();
    const char* end = p + full.size();
    while (p != end) {
      if (*p == c) {
        ++p;
      } else {
        const char* start = p;
        while (++p != end && *p != c) {
        }
        *result++ = std::string(start, p - start);
      }
    }
    return;
  }

  std::string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != std::string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

void SplitStringUsing(const std::string& full, const char* delim,
                      std::vector<std::string>* result) {
  std::back_insert_iterator<std::vector<std::string> > it(*result);
  SplitStringToIteratorUsing(full, delim, it);
}

template <typename StringType, typename ITR>
static inline void SplitStringToIteratorAllowEmpty(const StringType& full,
                                                   const char* delim,
                                                   int pieces, ITR& result) {
  std::string::size_type begin_index, end_index;
  begin_index = 0;
  for (int i = 0; (i < pieces - 1) || (pieces == 0); ++i) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = end_index + 1;
  }
  *result++ = full.substr(begin_index);
}

void SplitStringAllowEmpty(const std::string& full, const char* delim,
                           std::vector<std::string>* result) {
  std::back_insert_iterator<std::vector<std::string> > it(*result);
  SplitStringToIteratorAllowEmpty(full, delim, 0, it);
}

template <class ITERATOR>
static void JoinStringsIterator(const ITERATOR& start, const ITERATOR& end,
                                const char* delim, std::string* result) {
  assert(result != nullptr);
  result->clear();
  size_t delim_length = strlen(delim);

  // Precompute resulting length so we can reserve() memory in one shot.
  size_t length = 0;
  for (ITERATOR iter = start; iter != end; ++iter) {
    if (iter != start) {
      length += delim_length;
    }
    length += iter->size();
  }
  result->reserve(length);

  // Now combine everything.
  for (ITERATOR iter = start; iter != end; ++iter) {
    if (iter != start) {
      result->append(delim, delim_length);
    }
    result->append(iter->data(), iter->size());
  }
}

void JoinStrings(const std::vector<std::string>& components, const char* delim,
                 std::string* result) {
  JoinStringsIterator(components.begin(), components.end(), delim, result);
}

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  static const int kSpaceLength = 128;
  char space[kSpaceLength];

  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, kSpaceLength, format, backup_ap);
  va_end(backup_ap);

  if (result < kSpaceLength) {
    if (result >= 0) {
      dst->append(space, static_cast<size_t>(result));
      return;
    }
    if (result < 0) {
      // Error
      return;
    }
  }

  int length = result + 1;
  char* buf = new char[length];

  va_copy(backup_ap, ap);
  result = vsnprintf(buf, static_cast<size_t>(length), format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) {
    dst->append(buf, static_cast<size_t>(result));
  }
  delete[] buf;
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

const std::string& SStringPrintf(std::string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  dst->clear();
  StringAppendV(dst, format, ap);
  va_end(ap);
  return *dst;
}

void StringAppendF(std::string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  StringAppendV(dst, format, ap);
  va_end(ap);
}

}  // namespace myvoyager
}  // namespace bubblefs