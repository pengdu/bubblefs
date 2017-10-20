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

// tensorflow/tensorflow/core/lib/io/path.h

#ifndef BUBBLEFS_UTILS_PATH_H_
#define BUBBLEFS_UTILS_PATH_H_

#include <vector>
#include "utils/status.h"
#include "utils/stringpiece.h"

namespace bubblefs {
namespace io {
  
static const uint32_t MAX_PATH_LENGTH = 10240;  
  
namespace internal {
string JoinPathImpl(std::initializer_list<StringPiece> paths);
} // namespace internal

inline bool SplitPath(const std::string& path,
                      std::vector<std::string>* element,
                      bool* isdir = nullptr) {
    if (path.empty() || path[0] != '/' || path.size() > io::MAX_PATH_LENGTH) {
        return false;
    }
    element->clear();
    size_t last_pos = 0;
    for (size_t i = 1; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (last_pos + 1 < i) {
                element->push_back(path.substr(last_pos + 1, i - last_pos - 1));
            }
            last_pos = i;
        }
    }
    if (isdir) {
        *isdir = (path[path.size() - 1] == '/');
    }
    return true;
}

// Utility routines for processing filenames

#ifndef SWIG  // variadic templates
// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = io::JoinPath("/mydir", filename);
// string path = io::JoinPath(FLAGS_test_srcdir, filename);
// string path = io::JoinPath("/full", "path", "to", "filename);
template <typename... T>
string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}
#endif /* SWIG */

// Return true if path is absolute.
bool IsAbsolutePath(StringPiece path);

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input.
StringPiece Dirname(StringPiece path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
StringPiece Basename(StringPiece path);

// Returns the part of the basename of path after the final ".".  If
// there is no "." in the basename, the result is empty.
StringPiece Extension(StringPiece path);

// Collapse duplicate "/"s, resolve ".." and "." path elements, remove
// trailing "/".
//
// NOTE: This respects relative vs. absolute paths, but does not
// invoke any system calls (getcwd(2)) in order to resolve relative
// paths with respect to the actual working directory.  That is, this is purely
// string manipulation, completely independent of process state.
string CleanPath(StringPiece path);

// Populates the scheme, host, and path from a URI. scheme, host, and path are
// guaranteed by this function to point into the contents of uri, even if
// empty.
//
// Corner cases:
// - If the URI is invalid, scheme and host are set to empty strings and the
//   passed string is assumed to be a path
// - If the URI omits the path (e.g. file://host), then the path is left empty.
void ParseURI(StringPiece uri, StringPiece* scheme, StringPiece* host,
              StringPiece* path);

// Creates a URI from a scheme, host, and path. If the scheme is empty, we just
// return the path.
string CreateURI(StringPiece scheme, StringPiece host, StringPiece path);

}  // namespace io
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PATH_H_