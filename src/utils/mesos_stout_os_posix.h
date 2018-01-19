// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mesos/3rdparty/stout/include/stout/os/constants.hpp
// mesos/3rdparty/stout/include/stout/posix/dynamiclibrary.hpp
// mesos/3rdparty/stout/include/stout/uri.hpp
// mesos/3rdparty/stout/include/stout/path.hpp
// mesos/3rdparty/stout/include/stout/os/sysctl.hpp
// mesos/3rdparty/stout/include/stout/os/posix/bootid.hpp
// mesos/3rdparty/stout/include/stout/os/posix/chdir.hpp
// mesos/3rdparty/stout/include/stout/os/raw/argv.hpp
// mesos/3rdparty/stout/include/stout/os/posix/shell.hpp
// mesos/3rdparty/stout/include/stout/os/posix/stat.hpp
// mesos/3rdparty/stout/include/stout/os/posix/chown.hpp
// mesos/3rdparty/stout/include/stout/os/posix/chroot.hpp
// mesos/3rdparty/stout/include/stout/os/posix/close.hpp
// mesos/3rdparty/stout/include/stout/os/posix/copyfile.hpp
// mesos/3rdparty/stout/include/stout/os/posix/dup.hpp
// mesos/3rdparty/stout/include/stout/os/posix/exists.hpp
// mesos/3rdparty/stout/include/stout/os/posix/fcntl.hpp
// mesos/3rdparty/stout/include/stout/os/posix/fsync.hpp
// mesos/3rdparty/stout/include/stout/os/posix/ftruncate.hpp
// mesos/3rdparty/stout/include/stout/os/posix/getcwd.hpp
// mesos/3rdparty/stout/include/stout/os/posix/getenv.hpp
// mesos/3rdparty/stout/include/stout/os/posix/kill.hpp
// mesos/3rdparty/stout/include/stout/os/posix/ls.hpp
// mesos/3rdparty/stout/include/stout/os/posix/mkdir.hpp
// mesos/3rdparty/stout/include/stout/os/posix/temp.hpp
// mesos/3rdparty/stout/include/stout/os/posix/mkdtemp.hpp
// mesos/3rdparty/stout/include/stout/os/posix/mktemp.hpp
// mesos/3rdparty/stout/include/stout/os/posix/pagesize.hpp
// mesos/3rdparty/stout/include/stout/os/posix/pipe.hpp
// mesos/3rdparty/stout/include/stout/os/posix/realpath.hpp
// mesos/3rdparty/stout/include/stout/os/posix/rename.hpp
// mesos/3rdparty/stout/include/stout/os/posix/rm.hpp
// mesos/3rdparty/stout/include/stout/os/posix/rmdir.hpp
// mesos/3rdparty/stout/include/stout/os/posix/signals.hpp
// mesos/3rdparty/stout/include/stout/os/posix/sendfile.hpp
// mesos/3rdparty/stout/include/stout/os/posix/xattr.hpp
// mesos/3rdparty/stout/include/stout/os/posix/su.hpp
// mesos/3rdparty/stout/include/stout/posix/fs.hpp
// mesos/3rdparty/stout/include/stout/posix/net.hpp
// mesos/3rdparty/stout/include/stout/os/posix/fork.hpp
// mesos/3rdparty/stout/include/stout/os/pstree.hpp
// mesos/3rdparty/stout/include/stout/os/posix/killtree.hpp

#ifndef BUBBLEFS_UTILS_MESOS_STOUT_OS_POSIX_H_
#define BUBBLEFS_UTILS_MESOS_STOUT_OS_POSIX_H_

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <fts.h>
#include <grp.h>
#include <limits.h>
#include <pthread.h>
#include <pwd.h>
#include <signal.h>
#include <stdarg.h> // For va_list, va_start, etc.
#include <stdio.h> // For ferror, fgets, FILE, pclose, popen.
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h> // For waitpid
#include <sys/xattr.h>
#include <sys/sendfile.h> // __linux__

#include <atomic>
#include <array>
#include <list>
#include <queue>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/mesos_stout_error.h"
#include "utils/mesos_stout_foreach.h"
#include "utils/mesos_stout_format.h"
#include "utils/mesos_stout_option.h"
#include "utils/mesos_stout_process.h"
#include "utils/mesos_stout_some.h"
#include "utils/mesos_stout_stringify.h"
#include "utils/mesos_stout_strings.h"
#include "utils/mesos_stout_try.h"
#include "utils/mesos_stout_process.h"

namespace bubblefs {
namespace mymesos {
namespace os {

constexpr char WINDOWS_PATH_SEPARATOR = '\\';
constexpr char POSIX_PATH_SEPARATOR = '/';

#ifndef __WINDOWS__
constexpr char PATH_SEPARATOR = POSIX_PATH_SEPARATOR;
#else
constexpr char PATH_SEPARATOR = WINDOWS_PATH_SEPARATOR;
#endif // __WINDOWS__

#ifndef __WINDOWS__
constexpr char DEV_NULL[] = "/dev/null";
#else
constexpr char DEV_NULL[] = "NUL";
#endif // __WINDOWS__

#ifdef __WINDOWS__
// This prefix is prepended to absolute paths on Windows to indicate the path
// may be greater than 255 characters.
//
// NOTE: We do not use a R"raw string" here because syntax highlighters do not
// handle mismatched backslashes well.
constexpr char LONGPATH_PREFIX[] = "\\\\?\\";
#endif // __WINDOWS__  

/**
 * DynamicLibrary is a very simple wrapper around the programming interface
 * to the dynamic linking loader.
 */
class DynamicLibrary
{
public:
  DynamicLibrary() : handle_(nullptr) { }

  // Since this class manages a naked handle it cannot be copy- or
  // move-constructed.
  // TODO(bbannier): Allow for move-construction.
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary(DynamicLibrary&&) = delete;

  virtual ~DynamicLibrary()
  {
    if (handle_ != nullptr) {
      close();
    }
  }

  Try<Nothing> open(const std::string& path)
  {
    // Check if we've already opened a library.
    if (handle_ != nullptr) {
      return Error("Library already opened");
    }

    handle_ = dlopen(path.c_str(), RTLD_NOW);

    if (handle_ == nullptr) {
      return Error("Could not load library '" + path + "': " + dlerror());
    }

    path_ = path;

    return Nothing();
  }

  Try<Nothing> close()
  {
    if (handle_ == nullptr) {
      return Error("Could not close library; handle was already `nullptr`");
    }

    if (dlclose(handle_) != 0) {
      return Error(
          "Could not close library '" +
          (path_.isSome() ? path_.get() : "") + "': " + dlerror());
    }

    handle_ = nullptr;
    path_ = None();

    return Nothing();
  }

  Try<void*> loadSymbol(const std::string& name)
  {
    if (handle_ == nullptr) {
      return Error(
          "Could not get symbol '" + name + "'; library handle was `nullptr`");
    }

    void* symbol = dlsym(handle_, name.c_str());

    if (symbol == nullptr) {
      return Error(
          "Error looking up symbol '" + name + "' in '" +
          (path_.isSome() ? path_.get() : "") + "' : " + dlerror());
    }

    return symbol;
  }

private:
  void* handle_;
  Option<std::string> path_;
};

// Returns a valid URI containing a filename.
//
// On Windows, the / character is replaced with \ since that's the path
// separator. Note that / will often work, but will absolutely not work if the
// path is a long path.

const std::string FILE_PREFIX = "file://";

inline std::string from_path(const std::string& filepath)
{
#ifdef __WINDOWS__
  return FILE_PREFIX + strings::replace(filepath, "\\", "/");
#else
  return FILE_PREFIX + filepath;
#endif // __WINDOWS__
}

namespace path {

// Converts a fully formed URI to a filename for the platform.
//
// On all platforms, the optional "file://" prefix is removed if it
// exists.
//
// On Windows, this also converts "/" characters to "\" characters.
// The Windows file system APIs don't work with "/" in the filename
// when using long paths (although they do work fine if the file
// path happens to be short).
//
// NOTE: Currently, Mesos uses URIs and files somewhat interchangably.
// For compatibility, lack of "file://" prefix is not considered an
// error.
inline std::string from_uri(const std::string& uri)
{
  // Remove the optional "file://" if it exists.
  // TODO(coffler): Remove the `hostname` component.
  const std::string path = strings::remove(uri, "file://", strings::PREFIX);

#ifndef __WINDOWS__
  return path;
#else
  return strings::replace(path, "/", "\\");
#endif // __WINDOWS__
}


// Base case.
inline std::string join(
    const std::string& path1,
    const std::string& path2,
    const char _separator = os::PATH_SEPARATOR)
{
  const std::string separator = stringify(_separator);
  return strings::remove(path1, separator, strings::SUFFIX) +
         separator +
         strings::remove(path2, separator, strings::PREFIX);
}


template <typename... Paths>
inline std::string join(
    const std::string& path1,
    const std::string& path2,
    Paths&&... paths)
{
  return join(path1, join(path2, std::forward<Paths>(paths)...));
}


inline std::string join(const std::vector<std::string>& paths)
{
  if (paths.empty()) {
    return "";
  }

  std::string result = paths[0];
  for (size_t i = 1; i < paths.size(); ++i) {
    result = join(result, paths[i]);
  }
  return result;
}


/**
 * Returns whether the given path is an absolute path.
 * If an invalid path is given, the return result is also invalid.
 */
inline bool absolute(const std::string& path)
{
#ifndef __WINDOWS__
  return strings::startsWith(path, os::PATH_SEPARATOR);
#else
  // NOTE: We do not use `PathIsRelative` Windows utility function
  // here because it does not support long paths.
  //
  // See https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx
  // for details on paths. In short, an absolute path for files on Windows
  // looks like one of the following:
  //   * "[A-Za-z]:\"
  //   * "[A-Za-z]:/"
  //   * "\\?\..."
  //   * "\\server\..." where "server" is a network host.
  //
  // NOLINT(whitespace/line_length)

  // A uniform naming convention (UNC) name of any format,
  // always starts with two backslash characters.
  if (strings::startsWith(path, "\\\\")) {
    return true;
  }

  // A disk designator with a slash, for example "C:\" or "d:/".
  if (path.length() < 3) {
    return false;
  }

  const char letter = path[0];
  if (!((letter >= 'A' && letter <= 'Z') ||
        (letter >= 'a' && letter <= 'z'))) {
    return false;
  }

  std::string colon = path.substr(1, 2);
  return colon == ":\\" || colon == ":/";
#endif // __WINDOWS__
}

} // namespace path

/**
 * Represents a POSIX or Windows file system path and offers common path
 * manipulations. When reading the comments below, keep in mind that '/' refers
 * to the path separator character, so read it as "'/' or '\', depending on
 * platform".
 */
class Path
{
public:
  Path() : value() {}

  explicit Path(const std::string& path)
    : value(strings::remove(path, "file://", strings::PREFIX)) {}

  // TODO(cmaloney): Add more useful operations such as 'directoryname()',
  // 'filename()', etc.

  /**
   * Extracts the component following the final '/'. Trailing '/'
   * characters are not counted as part of the pathname.
   *
   * Like the standard '::basename()' except it is thread safe.
   *
   * The following list of examples (taken from SUSv2) shows the
   * strings returned by basename() for different paths:
   *
   * path        | basename
   * ----------- | -----------
   * "/usr/lib"  | "lib"
   * "/usr/"     | "usr"
   * "usr"       | "usr"
   * "/"         | "/"
   * "."         | "."
   * ".."        | ".."
   *
   * @return The component following the final '/'. If Path does not
   *   contain a '/', this returns a copy of Path. If Path is the
   *   string "/", then this returns the string "/". If Path is an
   *   empty string, then it returns the string ".".
   */
  inline std::string basename() const
  {
    if (value.empty()) {
      return std::string(".");
    }

    size_t end = value.size() - 1;

    // Remove trailing slashes.
    if (value[end] == os::PATH_SEPARATOR) {
      end = value.find_last_not_of(os::PATH_SEPARATOR, end);

      // Paths containing only slashes result into "/".
      if (end == std::string::npos) {
        return stringify(os::PATH_SEPARATOR);
      }
    }

    // 'start' should point towards the character after the last slash
    // that is non trailing.
    size_t start = value.find_last_of(os::PATH_SEPARATOR, end);

    if (start == std::string::npos) {
      start = 0;
    } else {
      start++;
    }

    return value.substr(start, end + 1 - start);
  }

  // TODO(hausdorff) Make sure this works on Windows for very short path names,
  // such as "C:\Temp". There is a distinction between "C:" and "C:\", the
  // former means "current directory of the C drive", while the latter means
  // "The root of the C drive". Also make sure that UNC paths are handled.
  // Will probably need to use the Windows path functions for that.
  /**
   * Extracts the component up to, but not including, the final '/'.
   * Trailing '/' characters are not counted as part of the pathname.
   *
   * Like the standard '::dirname()' except it is thread safe.
   *
   * The following list of examples (taken from SUSv2) shows the
   * strings returned by dirname() for different paths:
   *
   * path        | dirname
   * ----------- | -----------
   * "/usr/lib"  | "/usr"
   * "/usr/"     | "/"
   * "usr"       | "."
   * "/"         | "/"
   * "."         | "."
   * ".."        | "."
   *
   * @return The component up to, but not including, the final '/'. If
   *   Path does not contain a '/', then this returns the string ".".
   *   If Path is the string "/", then this returns the string "/".
   *   If Path is an empty string, then this returns the string ".".
   */
  inline std::string dirname() const
  {
    if (value.empty()) {
      return std::string(".");
    }

    size_t end = value.size() - 1;

    // Remove trailing slashes.
    if (value[end] == os::PATH_SEPARATOR) {
      end = value.find_last_not_of(os::PATH_SEPARATOR, end);
    }

    // Remove anything trailing the last slash.
    end = value.find_last_of(os::PATH_SEPARATOR, end);

    // Paths containing no slashes result in ".".
    if (end == std::string::npos) {
      return std::string(".");
    }

    // Paths containing only slashes result in "/".
    if (end == 0) {
      return stringify(os::PATH_SEPARATOR);
    }

    // 'end' should point towards the last non slash character
    // preceding the last slash.
    end = value.find_last_not_of(os::PATH_SEPARATOR, end);

    // Paths containing no non slash characters result in "/".
    if (end == std::string::npos) {
      return stringify(os::PATH_SEPARATOR);
    }

    return value.substr(0, end + 1);
  }

  /**
   * Returns the file extension of the path, including the dot.
   *
   * Returns None if the basename contains no dots, or consists
   * entirely of dots (i.e. '.', '..').
   *
   * Examples:
   *
   *   path         | extension
   *   ----------   | -----------
   *   "a.txt"      |  ".txt"
   *   "a.tar.gz"   |  ".gz"
   *   ".bashrc"    |  ".bashrc"
   *   "a"          |  None
   *   "."          |  None
   *   ".."         |  None
   */
  inline Option<std::string> extension() const
  {
    std::string _basename = basename();
    size_t index = _basename.rfind('.');

    if (_basename == "." || _basename == ".." || index == std::string::npos) {
      return None();
    }

    return _basename.substr(index);
  }

  // Checks whether the path is absolute.
  inline bool absolute() const
  {
    return path::absolute(value);
  }

  // Implicit conversion from Path to string.
  operator std::string() const
  {
    return value;
  }

  const std::string& string() const
  {
    return value;
  }

private:
  std::string value;
};


inline bool operator==(const Path& left, const Path& right)
{
  return left.string() == right.string();
}


inline bool operator!=(const Path& left, const Path& right)
{
  return !(left == right);
}


inline bool operator<(const Path& left, const Path& right)
{
  return left.string() < right.string();
}


inline bool operator>(const Path& left, const Path& right)
{
  return right < left;
}


inline bool operator<=(const Path& left, const Path& right)
{
  return !(left > right);
}


inline bool operator>=(const Path& left, const Path& right)
{
  return !(left < right);
}


inline std::ostream& operator<<(
    std::ostream& stream,
    const Path& path)
{
  return stream << path.string();
}
  
// Provides an abstraction for getting system information via the
// underlying 'sysctl' system call. You describe the sysctl
// "Management Information Base" (MIB) name via the constructor, for
// example, to describe "maximum number of processes allowed in the
// system" you would do:
//
//   os::sysctl(CTL_KERN, KERN_MAXPROC)
//
// To _retrieve_ the value you need to use one of the 'integer',
// 'string', 'table', or 'time' methods to indicate the type of the
// value being retrieved. For example:
//
//   Try<int> maxproc = os::sysctl(CTL_KERN, KERN_MAXPROC).integer();
//
// Note that the 'table' method requires specifying a length. If you
// would like the length to be looked up dynamically you can just pass
// None. Here's an example using 'table' that builds on above:
//
//   Try<vector<kinfo_proc>> processes =
//     os::sysctl(CTL_KERN, KERN_PROC, KERN_PROC_ALL).table(maxprox.get());
//
// TODO(benh): Provide an 'integer(i)', 'string(s)', and 'table(t)' to
// enable setting system information.
struct sysctl
{
  // Note that we create a constructor for each number of levels
  // because we can't pick a suitable default for unused levels (in
  // order to distinguish no value from some value) and while Option
  // would solve that it could also cause people to use None which
  // we'd need to later handle as an error.
  explicit sysctl(int level1);
  sysctl(int level1, int level2);
  sysctl(int level1, int level2, int level3);
  sysctl(int level1, int level2, int level3, int level4);
  sysctl(int level1, int level2, int level3, int level4, int level5);
  ~sysctl();

  // Get system information as an integer.
private: struct Integer; // Forward declaration.
public:
  Integer integer() const;

  // Get system information as a string.
  Try<std::string> string() const;

  // Get system information as a timeval.
  Try<timeval> time() const;

  // Get system information as a table, optionally specifying a
  // length. Note that this function is lazy and will not actually
  // perform the syscall until you cast (implicitly or explicitly) a
  // 'Table' to a std::vector<T>. For example, to get the first 10
  // processes in the process table you can do:
  //
  //     Try<std::vector<kinfo_proc>> processes =
  //       os::sysctl(CTL_KERN, KERN_PROC, KERN_PROC_ALL).table(10);
  //
private: struct Table; // Forward declaration.
public:
  Table table(const Option<size_t>& length = None()) const;

private:
  struct Integer
  {
    Integer(int _levels, int* _name);

    template <typename T>
    operator Try<T>();

    const int levels;
    int* name;
  };

  struct Table
  {
    Table(int _levels, int* _name, const Option<size_t>& _length);

    template <typename T>
    operator Try<std::vector<T>>();

    const int levels;
    int* name;
    Option<size_t> length;
  };

  const int levels;
  int* name;
};


inline sysctl::sysctl(int level1)
  : levels(1), name(new int[levels])
{
  name[0] = level1;
}


inline sysctl::sysctl(int level1, int level2)
  : levels(2), name(new int[levels])
{
  name[0] = level1;
  name[1] = level2;
}


inline sysctl::sysctl(int level1, int level2, int level3)
  : levels(3), name(new int[levels])
{
  name[0] = level1;
  name[1] = level2;
  name[2] = level3;
}


inline sysctl::sysctl(int level1, int level2, int level3, int level4)
  : levels(4), name(new int[levels])
{
  name[0] = level1;
  name[1] = level2;
  name[2] = level3;
  name[3] = level4;
}


inline sysctl::sysctl(
    int level1,
    int level2,
    int level3,
    int level4,
    int level5)
  : levels(5), name(new int[levels])
{
  name[0] = level1;
  name[1] = level2;
  name[2] = level3;
  name[3] = level4;
  name[4] = level5;
}


inline sysctl::~sysctl()
{
  delete[] name;
}


inline sysctl::Integer sysctl::integer() const
{
  return Integer(levels, name);
}


inline Try<std::string> sysctl::string() const
{
  // First determine the size of the string.
  size_t size = 0;
  if (::sysctl(name, levels, nullptr, &size, nullptr, 0) == -1) {
    return ErrnoError();
  }

  // Now read it.
  size_t length = size / sizeof(char);
  char* temp = new char[length];
  if (::sysctl(name, levels, temp, &size, nullptr, 0) == -1) {
    Error error = ErrnoError();
    delete[] temp;
    return error;
  }

  // TODO(benh): It's possible that the value has changed since we
  // determined it's length above. We should really check that we
  // get back the same length and if not throw an error.

  // The "string" in 'temp' might include null bytes, so to get all of
  // the data we need to create a string with 'size' (but we exclude
  // the last null byte via 'size - 1').
  std::string result(temp, size - 1);
  delete[] temp;
  return result;
}


inline Try<timeval> sysctl::time() const
{
  timeval result;
  size_t size = sizeof(result);
  if (::sysctl(name, levels, &result, &size, nullptr, 0) == -1) {
    return ErrnoError();
  }
  return result;
}


inline sysctl::Table sysctl::table(const Option<size_t>& length) const
{
  return Table(levels, name, length);
}


inline sysctl::Integer::Integer(
    int _levels,
    int* _name)
  : levels(_levels),
    name(_name)
{}


template <typename T>
sysctl::Integer::operator Try<T>()
{
  T i;
  size_t size = sizeof(i);
  if (::sysctl(name, levels, &i, &size, nullptr, 0) == -1) {
    return ErrnoError();
  }
  return i;
}


inline sysctl::Table::Table(
    int _levels,
    int* _name,
    const Option<size_t>& _length)
  : levels(_levels),
    name(_name),
    length(_length)
{}


template <typename T>
sysctl::Table::operator Try<std::vector<T>>()
{
  size_t size = 0;
  if (length.isNone()) {
    if (::sysctl(name, levels, nullptr, &size, nullptr, 0) == -1) {
      return ErrnoError();
    }
    if (size % sizeof(T) != 0) {
      return Error("Failed to determine the length of result, "
                   "amount of available data is not a multiple "
                   "of the table type");
    }
    length = Option<size_t>(size / sizeof(T));
  }

  T* ts = new T[length.get()];
  size = length.get() * sizeof(T);
  if (::sysctl(name, levels, ts, &size, nullptr, 0) == -1) {
    Error error = ErrnoError();
    delete[] ts;
    return error;
  }

  // TODO(benh): It's possible that the value has changed since we
  // determined it's length above (or from what was specified). We
  // should really check that we get back the same length and if not
  // throw an error.

  length = size / sizeof(T);

  std::vector<T> results;
  for (size_t i = 0; i < length.get(); i++) {
    results.push_back(ts[i]);
  }
  delete[] ts;
  return results;
}  
  
inline Try<std::string> bootId()
{
  Try<std::string> read = ::read("/proc/sys/kernel/random/boot_id");
  if (read.isError()) {
    return read;
  }
  return strings::trim(read.get());
}

inline Try<Nothing> chdir(const std::string& directory)
{
  if (::chdir(directory.c_str()) < 0) {
    return ErrnoError();
  }

  return Nothing();
}

namespace raw {

/**
 * Represent the argument list expected by `execv` routines. The
 * argument list is an array of pointers that point to null-terminated
 * strings. The array of pointers must be terminated by a nullptr. To
 * use this abstraction, see the following example:
 *
 *   vector<string> args = {"arg0", "arg1"};
 *   os::raw::Argv argv(args);
 *   execvp("my_binary", argv);
 */
class Argv
{
public:
  Argv(const Argv&) = delete;
  Argv& operator=(const Argv&) = delete;

  template <typename Iterable>
  explicit Argv(const Iterable& iterable)
  {
    foreach (const std::string& arg, iterable) {
      args.emplace_back(arg);
    }

    argv = new char*[args.size() + 1];
    for (size_t i = 0; i < args.size(); i++) {
      argv[i] = const_cast<char*>(args[i].c_str());
    }

    argv[args.size()] = nullptr;
  }

  ~Argv()
  {
    delete[] argv;
  }

  operator char**() const
  {
    return argv;
  }

  operator std::vector<std::string>() const
  {
    return args;
  }

private:
  std::vector<std::string> args;

  // NOTE: This points to strings in the vector `args`.
  char** argv;
};

} // namespace raw 

namespace Shell {

// Canonical constants used as platform-dependent args to `exec`
// calls. `name` is the command name, `arg0` is the first argument
// received by the callee, usually the command name and `arg1` is the
// second command argument received by the callee.

constexpr const char* name = "sh";
constexpr const char* arg0 = "sh";
constexpr const char* arg1 = "-c";

} // namespace Shell {

/**
 * Runs a shell command with optional arguments.
 *
 * This assumes that a successful execution will result in the exit code
 * for the command to be `EXIT_SUCCESS`; in this case, the contents
 * of the `Try` will be the contents of `stdout`.
 *
 * If the exit code is non-zero or the process was signaled, we will
 * return an appropriate error message; but *not* `stderr`.
 *
 * If the caller needs to examine the contents of `stderr` it should
 * be redirected to `stdout` (using, e.g., "2>&1 || true" in the command
 * string).  The `|| true` is required to obtain a success exit
 * code in case of errors, and still obtain `stderr`, as piped to
 * `stdout`.
 *
 * @param fmt the formatting string that contains the command to execute
 *   in the underlying shell.
 * @param t optional arguments for `fmt`.
 *
 * @return the output from running the specified command with the shell; or
 *   an error message if the command's exit code is non-zero.
 */
template <typename... T>
Try<std::string> shell(const std::string& fmt, const T&... t)
{
  const Try<std::string> command = strings::internal::format(fmt, t...);
  if (command.isError()) {
    return Error(command.error());
  }

  FILE* file;
  std::ostringstream stdout;

  if ((file = popen(command.get().c_str(), "r")) == nullptr) {
    return Error("Failed to run '" + command.get() + "'");
  }

  char line[1024];
  // NOTE(vinod): Ideally the if and while loops should be interchanged. But
  // we get a broken pipe error if we don't read the output and simply close.
  while (fgets(line, sizeof(line), file) != nullptr) {
    stdout << line;
  }

  if (ferror(file) != 0) {
    pclose(file); // Ignoring result since we already have an error.
    return Error("Error reading output of '" + command.get() + "'");
  }

  int status;
  if ((status = pclose(file)) == -1) {
    return Error("Failed to get status of '" + command.get() + "'");
  }

  if (WIFSIGNALED(status)) {
    return Error(
        "Running '" + command.get() + "' was interrupted by signal '" +
        strsignal(WTERMSIG(status)) + "'");
  } else if ((WEXITSTATUS(status) != EXIT_SUCCESS)) {
    LOG(ERROR) << "Command '" << command.get()
               << "' failed; this is the output:\n" << stdout.str();
    return Error(
        "Failed to execute '" + command.get() + "'; the command was either "
        "not found or exited with a non-zero exit status: " +
        stringify(WEXITSTATUS(status)));
  }

  return stdout.str();
}


// Executes a command by calling "/bin/sh -c <command>", and returns
// after the command has been completed. Returns 0 if succeeds, and
// return -1 on error (e.g., fork/exec/waitpid failed). This function
// is async signal safe. We return int instead of returning a Try
// because Try involves 'new', which is not async signal safe.
//
// Note: Be cautious about shell injection
// (https://en.wikipedia.org/wiki/Code_injection#Shell_injection)
// when using this method and use proper validation and sanitization
// on the `command`. For this reason in general `os::spawn` is
// preferred if a shell is not required.
inline int system(const std::string& command)
{
  pid_t pid = ::fork();

  if (pid == -1) {
    return -1;
  } else if (pid == 0) {
    // In child process.
    ::execlp(
        Shell::name, Shell::arg0, Shell::arg1, command.c_str(), (char*)nullptr);
    ::exit(127);
  } else {
    // In parent process.
    int status;
    while (::waitpid(pid, &status, 0) == -1) {
      if (errno != EINTR) {
        return -1;
      }
    }

    return status;
  }
}

// Executes a command by calling "<command> <arguments...>", and
// returns after the command has been completed. Returns 0 if
// succeeds, and -1 on error (e.g., fork/exec/waitpid failed). This
// function is async signal safe. We return int instead of returning a
// Try because Try involves 'new', which is not async signal safe.
inline int spawn(
    const std::string& command,
    const std::vector<std::string>& arguments)
{
  pid_t pid = ::fork();

  if (pid == -1) {
    return -1;
  } else if (pid == 0) {
    // In child process.
    ::execvp(command.c_str(), os::raw::Argv(arguments));
    ::exit(127);
  } else {
    // In parent process.
    int status;
    while (::waitpid(pid, &status, 0) == -1) {
      if (errno != EINTR) {
        return -1;
      }
    }

    return status;
  }
}


template<typename... T>
inline int execlp(const char* file, T... t)
{
  return ::execlp(file, t...);
}


inline int execvp(const char* file, char* const argv[])
{
  return ::execvp(file, argv);
}

namespace stat {

// Specify whether symlink path arguments should be followed or
// not. APIs in the os::stat family that take a FollowSymlink
// argument all provide FollowSymlink::FOLLOW_SYMLINK as the default value,
// so they will follow symlinks unless otherwise specified.
enum class FollowSymlink
{
  DO_NOT_FOLLOW_SYMLINK,
  FOLLOW_SYMLINK
};

namespace internal {

inline Try<struct ::stat> stat(
    const std::string& path,
    const FollowSymlink follow)
{
  struct ::stat s;

  switch (follow) {
    case FollowSymlink::DO_NOT_FOLLOW_SYMLINK:
      if (::lstat(path.c_str(), &s) < 0) {
        return ErrnoError("Failed to lstat '" + path + "'");
      }
      return s;
    case FollowSymlink::FOLLOW_SYMLINK:
      if (::stat(path.c_str(), &s) < 0) {
        return ErrnoError("Failed to stat '" + path + "'");
      }
      return s;
  }
  abort();
}

} // namespace internal {

inline bool islink(const std::string& path)
{
  // By definition, you don't follow symlinks when trying
  // to find whether a path is a link. If you followed it,
  // it wouldn't ever be a link.
  Try<struct ::stat> s = internal::stat(
      path, FollowSymlink::DO_NOT_FOLLOW_SYMLINK);
  return s.isSome() && S_ISLNK(s->st_mode);
}


inline bool isdir(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  return s.isSome() && S_ISDIR(s->st_mode);
}


inline bool isfile(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  return s.isSome() && S_ISREG(s->st_mode);
}

// Returns the size in Bytes of a given file system entry. When
// applied to a symbolic link with `follow` set to
// `DO_NOT_FOLLOW_SYMLINK`, this will return the length of the entry
// name (strlen).
inline long size(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return (long)(s->st_size);
}


inline Try<long> mtime(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return s->st_mtime;
}


inline Try<mode_t> mode(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return s->st_mode;
}


inline Try<dev_t> dev(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return s->st_dev;
}


inline Try<dev_t> rdev(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  if (!S_ISCHR(s->st_mode) && !S_ISBLK(s->st_mode)) {
    return Error("Not a special file: " + path);
  }

  return s->st_rdev;
}


inline Try<ino_t> inode(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return s->st_ino;
}


inline Try<uid_t> uid(
    const std::string& path,
    const FollowSymlink follow = FollowSymlink::FOLLOW_SYMLINK)
{
  Try<struct ::stat> s = internal::stat(path, follow);
  if (s.isError()) {
    return Error(s.error());
  }

  return s->st_uid;
}

} // namespace stat

// Set the ownership for a path. This function never follows any symlinks.
inline Try<Nothing> chown(
    uid_t uid,
    gid_t gid,
    const std::string& path,
    bool recursive)
{
  char* path_[] = {const_cast<char*>(path.c_str()), nullptr};

  FTS* tree = ::fts_open(
      path_, FTS_NOCHDIR | FTS_PHYSICAL, nullptr);

  if (tree == nullptr) {
    return ErrnoError();
  }

  FTSENT *node;
  while ((node = ::fts_read(tree)) != nullptr) {
    switch (node->fts_info) {
      // Preorder directory.
      case FTS_D:
      // Regular file.
      case FTS_F:
      // Symbolic link.
      case FTS_SL:
      // Symbolic link without target.
      case FTS_SLNONE: {
        if (::lchown(node->fts_path, uid, gid) < 0) {
          Error error = ErrnoError();
          ::fts_close(tree);
          return error;
        }

        break;
      }

      // Unreadable directory.
      case FTS_DNR:
      // Error; errno is set.
      case FTS_ERR:
      // Directory that causes cycles.
      case FTS_DC:
      // `stat(2)` failed.
      case FTS_NS: {
        Error error = ErrnoError();
        ::fts_close(tree);
        return error;
      }

      default:
        break;
    }

    if (node->fts_level == FTS_ROOTLEVEL && !recursive) {
      break;
    }
  }

  ::fts_close(tree);
  return Nothing();
}

// Changes the specified path's user and group ownership to that of
// the specified user.
inline Try<Nothing> chown(
    const std::string& user,
    const std::string& path,
    bool recursive = true)
{
  passwd* passwd;

  errno = 0;

  if ((passwd = ::getpwnam(user.c_str())) == nullptr) {
    return errno
      ? ErrnoError("Failed to get user information for '" + user + "'")
      : Error("No such user '" + user + "'");
  }

  return chown(passwd->pw_uid, passwd->pw_gid, path, recursive);
}

inline Try<Nothing> chroot(const std::string& directory)
{
  if (::chroot(directory.c_str()) < 0) {
    return ErrnoError();
  }

  return Nothing();
}

inline Try<Nothing> close(int fd)
{
  if (::close(fd) != 0) {
    return ErrnoError();
  }

  return Nothing();
}

// This implementation works by running the `cp` command with some
// additional conditions to ensure we copy a single file only,
// from an absolute file path to another absolute file path.
//
// Directories are not supported as a destination path for two reasons:
// 1. No callers depended on that behavior,
// 2. Consistency with Windows implementation.
//
// Relative paths are not allowed, as these are resolved based on
// the current working directory and may be inconsistent.
inline Try<Nothing> copyfile(
    const std::string& source, const std::string& destination)
{
  // NOTE: We check the form of the path too in case it does not exist, and to
  // prevent user error.
  if (stat::isdir(source) || source.back() == '/') {
    return Error("`source` was a directory");
  }

  if (stat::isdir(destination) || destination.back() == '/') {
    return Error("`destination` was a directory");
  }

  if (!path::absolute(source)) {
    return Error("`source` was a relative path");
  }

  if (!path::absolute(destination)) {
    return Error("`destination` was a relative path");
  }

  const int status = os::spawn("cp", {"cp", source, destination});

  if (status == -1) {
    return ErrnoError("os::spawn failed");
  }

  if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
    return Error("cp failed with status: " + stringify(status));
  }

  return Nothing();
}

inline Try<int> dup(int fd)
{
  int result = ::dup(fd);
  if (result < 0) {
    return ErrnoError();
  }
  return result;
}

inline bool exists(const std::string& path)
{
  struct stat s;

  if (::lstat(path.c_str(), &s) < 0) {
    return false;
  }
  return true;
}

// Determine if the process identified by pid exists.
// NOTE: Zombie processes have a pid and therefore exist. See os::process(pid)
// to get details of a process.
inline bool exists(pid_t pid)
{
  // The special signal 0 is used to check if the process exists; see kill(2).
  // If the current user does not have permission to signal pid, but it does
  // exist, then ::kill will return -1 and set errno == EPERM.
  if (::kill(pid, 0) == 0 || errno == EPERM) {
    return true;
  }

  return false;
}

inline Try<Nothing> cloexec(int fd)
{
  int flags = ::fcntl(fd, F_GETFD);

  if (flags == -1) {
    return ErrnoError();
  }

  if (::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
    return ErrnoError();
  }

  return Nothing();
}


inline Try<Nothing> unsetCloexec(int fd)
{
  int flags = ::fcntl(fd, F_GETFD);

  if (flags == -1) {
    return ErrnoError();
  }

  if (::fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC) == -1) {
    return ErrnoError();
  }

  return Nothing();
}


inline Try<bool> isCloexec(int fd)
{
  int flags = ::fcntl(fd, F_GETFD);

  if (flags == -1) {
    return ErrnoError();
  }

  return (flags & FD_CLOEXEC) != 0;
}


inline Try<Nothing> nonblock(int fd)
{
  int flags = ::fcntl(fd, F_GETFL);

  if (flags == -1) {
    return ErrnoError();
  }

  if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    return ErrnoError();
  }

  return Nothing();
}


inline Try<bool> isNonblock(int fd)
{
  int flags = ::fcntl(fd, F_GETFL);

  if (flags == -1) {
    return ErrnoError();
  }

  return (flags & O_NONBLOCK) != 0;
}

inline Try<Nothing> fsync(int fd)
{
  if (::fsync(fd) == -1) {
    return ErrnoError();
  }

  return Nothing();
}

inline Try<Nothing> ftruncate(int fd, off_t length)
{
  if (::ftruncate(fd, length) != 0) {
    return ErrnoError(
      "Failed to truncate file at file descriptor '" + stringify(fd) + "' to " +
      stringify(length) + " bytes.");
  }

  return Nothing();
}


inline std::string getcwd()
{
  size_t size = 100;

  while (true) {
    char* temp = new char[size];
    if (::getcwd(temp, size) == temp) {
      std::string result(temp);
      delete[] temp;
      return result;
    } else {
      if (errno != ERANGE) {
        delete[] temp;
        return std::string();
      }
      size *= 2;
      delete[] temp;
    }
  }

  return std::string();
}


// Looks in the environment variables for the specified key and
// returns a string representation of its value. If no environment
// variable matching key is found, None() is returned.
inline Option<std::string> getenv(const std::string& key)
{
  char* value = ::getenv(key.c_str());

  if (value == nullptr) {
    return None();
  }

  return std::string(value);
}

inline int kill(pid_t pid, int sig)
{
  return ::kill(pid, sig);
}

inline Try<std::list<std::string>> ls(const std::string& directory)
{
  DIR* dir = opendir(directory.c_str());

  if (dir == nullptr) {
    return ErrnoError("Failed to opendir '" + directory + "'");
  }

  std::list<std::string> result;
  struct dirent* entry;

  // Zero `errno` before starting to call `readdir`. This is necessary
  // to allow us to determine when `readdir` returns an error.
  errno = 0;

  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    result.push_back(entry->d_name);
  }

  if (errno != 0) {
    // Preserve `readdir` error.
    Error error = ErrnoError("Failed to read directory");
    closedir(dir);
    return error;
  }

  if (closedir(dir) == -1) {
    return ErrnoError("Failed to close directory");
  }

  return result;
}

inline Try<Nothing> mkdir(const std::string& directory, bool recursive = true)
{
  if (!recursive) {
    if (::mkdir(directory.c_str(), 0755) < 0) {
      return ErrnoError();
    }
  } else {
    std::vector<std::string> tokens =
      strings::tokenize(directory, stringify(os::PATH_SEPARATOR));

    std::string path;

    // We got an absolute path, so keep the leading slash.
    if (directory.find_first_of(stringify(os::PATH_SEPARATOR)) == 0) {
      path = os::PATH_SEPARATOR;
    }

    foreach (const std::string& token, tokens) {
      path += token;
      if (::mkdir(path.c_str(), 0755) < 0 && errno != EEXIST) {
        return ErrnoError();
      }

      path += os::PATH_SEPARATOR;
    }
  }

  return Nothing();
}

// Attempts to resolve the system-designated temporary directory before
// back on a sensible default. On POSIX platforms, this involves checking
// the POSIX-standard `TMPDIR` environment variable before falling
// back to `/tmp`.
inline std::string temp()
{
  Option<std::string> tmpdir = os::getenv("TMPDIR");

  return tmpdir.getOrElse("/tmp");
}

// Creates a temporary directory using the specified path
// template. The template may be any path with _6_ `Xs' appended to
// it, for example /tmp/temp.XXXXXX. The trailing `Xs' are replaced
// with a unique alphanumeric combination.
inline Try<std::string> mkdtemp(
    const std::string& path = path::join(os::temp(), "XXXXXX"))
{
  char* temp = new char[path.size() + 1];
  ::memcpy(temp, path.c_str(), path.size() + 1);

  if (::mkdtemp(temp) != nullptr) {
    std::string result(temp);
    delete[] temp;
    return result;
  } else {
    delete[] temp;
    return ErrnoError();
  }
}

// Creates a temporary file using the specified path template. The
// template may be any path with _6_ `Xs' appended to it, for example
// /tmp/temp.XXXXXX. The trailing `Xs' are replaced with a unique
// alphanumeric combination.
inline Try<std::string> mktemp(
    const std::string& path = path::join(os::temp(), "XXXXXX"))
{
  char* temp = new char[path.size() + 1];
  ::memcpy(temp, path.c_str(), path.size() + 1);

  int fd = ::mkstemp(temp);
  if (fd < 0) {
    delete[] temp;
    return ErrnoError();
  }

  // We ignore the return value of close(). This is because users
  // calling this function are interested in the return value of
  // mkstemp(). Also an unsuccessful close() doesn't affect the file.
  os::close(fd);

  std::string result(temp);
  delete[] temp;
  return result;
}

// The alternative `getpagesize()` is not defined by POSIX.
inline size_t pagesize()
{
  // We assume that `sysconf` will not fail in practice.
  long result = ::sysconf(_SC_PAGESIZE);
  CHECK(result >= 0);
  return result;
}

// Create pipes for interprocess communication.
inline Try<std::array<int, 2>> pipe()
{
  std::array<int, 2> result;
  if (::pipe(result.data()) < 0) {
    return ErrnoError();
  }
  return result;
}

inline Result<std::string> realpath(const std::string& path)
{
  char temp[PATH_MAX];
  if (::realpath(path.c_str(), temp) == nullptr) {
    if (errno == ENOENT || errno == ENOTDIR) {
      return None();
    }

    return ErrnoError();
  }

  return std::string(temp);
}

inline Try<Nothing> rename(const std::string& from, const std::string& to)
{
  if (::rename(from.c_str(), to.c_str()) != 0) {
    return ErrnoError();
  }

  return Nothing();
}

inline Try<Nothing> rm(const std::string& path)
{
  if (::remove(path.c_str()) != 0) {
    return ErrnoError();
  }

  return Nothing();
}

// By default, recursively deletes a directory akin to: 'rm -r'. If
// `recursive` is false, it deletes a directory akin to: 'rmdir'. In
// recursive mode, `removeRoot` can be set to false to enable removing
// all the files and directories beneath the given root directory, but
// not the root directory itself.
// Note that this function expects an absolute path.
// By default rmdir aborts when an error occurs during the deletion of
// any file but if 'continueOnError' is set to true, rmdir logs the error
// and continues with the next file.

inline Try<Nothing> rmdir(
    const std::string& directory,
    bool recursive = true,
    bool removeRoot = true,
    bool continueOnError = false)
{
  unsigned int errorCount = 0;

  if (!recursive) {
    if (::rmdir(directory.c_str()) < 0) {
      return ErrnoError();
    }
  } else {
    // NOTE: `fts_open` will not always return `nullptr` if the path does not
    // exist. We manually induce an error here to indicate that we can't remove
    // a directory that does not exist.
    if (!os::exists(directory)) {
      return ErrnoError(ENOENT);
    }

    char* paths[] = {const_cast<char*>(directory.c_str()), nullptr};

    // Using `FTS_PHYSICAL` here because we need `FTSENT` for the
    // symbolic link in the directory and not the target it links to.
    FTS* tree = fts_open(paths, (FTS_NOCHDIR | FTS_PHYSICAL), nullptr);
    if (tree == nullptr) {
      return ErrnoError();
    }

    FTSENT* node;
    while ((node = fts_read(tree)) != nullptr) {
      switch (node->fts_info) {
        case FTS_DP:
          // Don't remove the root of the traversal of `removeRoot`
          // is false.
          if (!removeRoot && node->fts_level == FTS_ROOTLEVEL) {
            continue;
          }

          if (::rmdir(node->fts_path) < 0 && errno != ENOENT) {
            if (continueOnError) {
              LOG(ERROR) << "Failed to delete directory "
                         << path::join(directory, node->fts_path)
                         << ": " << os::strerror(errno);
              ++errorCount;
            } else {
              Error error = ErrnoError();
              fts_close(tree);
              return error;
            }
          }
          break;
        // `FTS_DEFAULT` would include any file type which is not
        // explicitly described by any of the other `fts_info` values.
        case FTS_DEFAULT:
        case FTS_F:
        case FTS_SL:
        // `FTS_SLNONE` should never be the case as we don't set
        // `FTS_COMFOLLOW` or `FTS_LOGICAL`. Adding here for completion.
        case FTS_SLNONE:
          if (::unlink(node->fts_path) < 0 && errno != ENOENT) {
            if (continueOnError) {
              LOG(ERROR) << "Failed to delete path "
                         << path::join(directory, node->fts_path)
                         << ": " << os::strerror(errno);
              ++errorCount;
            } else {
              Error error = ErrnoError();
              fts_close(tree);
              return error;
            }
          }
          break;
        default:
          break;
      }
    }

    if (errno != 0) {
      Error error = ErrnoError("fts_read failed");
      fts_close(tree);
      return error;
    }

    if (fts_close(tree) < 0) {
      return ErrnoError();
    }
  }

  if (errorCount > 0) {
    return Error("Failed to delete " + stringify(errorCount) + " paths");
  }

  return Nothing();
}


namespace signals {

// Installs the given signal handler.
inline int install(int signal, void(*handler)(int))
{
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  sigemptyset(&action.sa_mask);
  action.sa_handler = handler;
  return sigaction(signal, &action, nullptr);
}


// Resets the signal handler to the default handler of the signal.
inline int reset(int signal)
{
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  sigemptyset(&action.sa_mask);
  action.sa_handler = SIG_DFL;
  return sigaction(signal, &action, nullptr);
}


// Returns true iff the signal is pending.
inline bool pending(int signal)
{
  sigset_t set;
  sigemptyset(&set);
  sigpending(&set);
  return sigismember(&set, signal);
}


// Returns true if the signal has been blocked, or false if the
// signal was already blocked.
inline bool block(int signal)
{
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, signal);

  sigset_t oldset;
  sigemptyset(&oldset);

  // We ignore errors here as the only documented one is
  // EINVAL due to a bad value of the SIG_* argument.
  pthread_sigmask(SIG_BLOCK, &set, &oldset);

  return !sigismember(&oldset, signal);
}


// Returns true if the signal has been unblocked, or false if the
// signal was not previously blocked.
inline bool unblock(int signal)
{
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, signal);

  sigset_t oldset;
  sigemptyset(&oldset);

  pthread_sigmask(SIG_UNBLOCK, &set, &oldset);

  return sigismember(&oldset, signal);
}

namespace internal {

// Suppresses a signal on the current thread for the lifetime of
// the Suppressor. The signal *must* be synchronous and delivered
// per-thread. The suppression occurs only on the thread of
// execution of the Suppressor.
struct Suppressor
{
  Suppressor(int _signal)
    : signal(_signal), pending(false), unblock(false)
  {
    // Check to see if the signal is already reported as pending.
    // If pending, it means the thread already blocks the signal!
    // Therefore, any new instances of the signal will also be
    // blocked and merged with the pending one since there is no
    // queuing for signals.
    pending = signals::pending(signal);

    if (!pending) {
      // Block the signal for this thread only. If already blocked,
      // there's no need to unblock it.
      unblock = signals::block(signal);
    }
  }

  ~Suppressor()
  {
    // We want to preserve errno when the Suppressor drops out of
    // scope. Otherwise, one needs to potentially store errno when
    // using the suppress() macro.
    int _errno = errno;

    // If the signal has become pending after we blocked it, we
    // need to clear it before unblocking it.
    if (!pending && signals::pending(signal)) {
      // It is possible that in between having observed the pending
      // signal with sigpending() and clearing it with sigwait(),
      // the signal was delivered to another thread before we were
      // able to clear it here. This can happen if the signal was
      // generated for the whole process (e.g. a kill was issued).
      // See 2.4.1 here:
      // http://pubs.opengroup.org/onlinepubs/009695399/functions/xsh_chap02_04.html
      // To handle the above scenario, one can either:
      //   1. Use sigtimedwait() with a timeout of 0, to ensure we
      //      don't block forever. However, this only works on Linux
      //      and we may still swallow the signal intended for the
      //      process.
      //   2. After seeing the pending signal, signal ourselves with
      //      pthread_kill prior to calling sigwait(). This can still
      //      swallow the signal intended for the process.
      // We chose to use the latter technique as it works on all
      // POSIX systems and is less likely to swallow process signals,
      // provided the thread signal and process signal are not merged.

      // Delivering on this thread an extra time will require an extra sigwait
      // call on FreeBSD, so we skip it.
#ifndef __FreeBSD__
      pthread_kill(pthread_self(), signal);
#endif

      sigset_t mask;
      sigemptyset(&mask);
      sigaddset(&mask, signal);

      int result;
      do {
        int _ignored;
        result = sigwait(&mask, &_ignored);
      } while (result == -1 && errno == EINTR);
    }

    // Unblock the signal (only if we were the ones to block it).
    if (unblock) {
      signals::unblock(signal);
    }

    // Restore errno.
    errno = _errno;
  }

  // Needed for the suppress() macro.
  operator bool() { return true; }
private:
  const int signal;
  bool pending; // Whether the signal is already pending.
  bool unblock; // Whether to unblock the signal on destruction.
};

} // namespace internal

#define SUPPRESS(signal) \
  if (os::signals::internal::Suppressor suppressor ## signal = \
      os::signals::internal::Suppressor(signal))

} // namespace signals

// Returns the amount of bytes written from the input file
// descriptor to the output socket. On error,
// `Try<ssize_t, SocketError>` contains the error.
// NOTE: The following limitations exist because of the OS X
// implementation of sendfile:
//   1. s must be a stream oriented socket descriptor.
//   2. fd must be a regular file descriptor.
inline Try<ssize_t, SocketError> sendfile(
    int s, int fd, off_t offset, size_t length)
{
  SUPPRESS (SIGPIPE) {
    // This will set errno to EPIPE if a SIGPIPE occurs.
    ssize_t sent = ::sendfile(s, fd, &offset, length);
    if (sent < 0) {
      return SocketError();
    }

    return sent;
  }
  abort();
}

inline Try<Nothing> setxattr(
    const std::string& path,
    const std::string& name,
    const std::string& value,
    int flags)
{
#ifdef __APPLE__
  if (::setxattr(
      path.c_str(),
      name.c_str(),
      value.c_str(),
      value.length(),
      0,
      flags) < 0) {
#elif __FreeBSD__
  if (::extattr_set_file(
        path.c_str(),
        EXTATTR_NAMESPACE_USER,
        name.c_str(),
        value.c_str(),
        value.length()) < 0) {
#else
  if (::setxattr(
        path.c_str(),
        name.c_str(),
        value.c_str(),
        value.length(),
        flags) < 0) {
#endif
    return ErrnoError();
  }

  return Nothing();
}


inline Try<std::string> getxattr(
    const std::string& path,
    const std::string& name)
{
  // Get the current size of the attribute.
#ifdef __APPLE__
  ssize_t size = ::getxattr(path.c_str(), name.c_str(), nullptr, 0, 0, 0);
#elif __FreeBSD__
  ssize_t size = ::extattr_get_file(path.c_str(),
                                    EXTATTR_NAMESPACE_USER,
                                    name.c_str(),
                                    nullptr,
                                    0);
#else
  ssize_t size = ::getxattr(path.c_str(), name.c_str(), nullptr, 0);
#endif
  if (size < 0) {
    return ErrnoError();
  }

  char* temp = new char[size + 1];
  ::memset(temp, 0, (size_t)size + 1);

#ifdef __APPLE__
  if (::getxattr(path.c_str(), name.c_str(), temp, (size_t)size, 0, 0) < 0) {
#elif __FreeBSD__
  if (::extattr_get_file(
              path.c_str(),
              EXTATTR_NAMESPACE_USER,
              name.c_str(),
              temp,
              (size_t)size) < 0) {
#else
  if (::getxattr(path.c_str(), name.c_str(), temp, (size_t)size) < 0) {
#endif
    delete[] temp;
    return ErrnoError();
  }

  std::string result(temp);
  delete[] temp;

  return result;
}


inline Try<Nothing> removexattr(
    const std::string& path,
    const std::string& name)
{
#ifdef __APPLE__
  if (::removexattr(path.c_str(), name.c_str(), 0) < 0) {
#elif __FreeBSD__
  if (::extattr_delete_file(path.c_str(),
                            EXTATTR_NAMESPACE_USER,
                            name.c_str())) {
#else
  if (::removexattr(path.c_str(), name.c_str()) < 0) {
#endif
    return ErrnoError();
  }

  return Nothing();
}

inline Result<uid_t> getuid(const Option<std::string>& user = None())
{
  if (user.isNone()) {
    return ::getuid();
  }

  int size = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (size == -1) {
    // Initial value for buffer size.
    size = 1024;
  }

  while (true) {
    struct passwd pwd;
    struct passwd* result;
    char* buffer = new char[size];

    if (getpwnam_r(user->c_str(), &pwd, buffer, size, &result) == 0) {
      // Per POSIX, if the user name is not found, `getpwnam_r` returns
      // zero and sets `result` to the null pointer. (Linux behaves
      // differently for invalid user names; see below).
      if (result == nullptr) {
        delete[] buffer;
        return None();
      }

      // Entry found.
      uid_t uid = pwd.pw_uid;
      delete[] buffer;
      return uid;
    } else {
      delete[] buffer;

      if (errno == ERANGE) {
        // Buffer too small; enlarge it and retry.
        size *= 2;
        continue;
      }

      // According to POSIX, a non-zero return value from `getpwnam_r`
      // indicates an error. However, some versions of glibc return
      // non-zero and set errno to ENOENT, ESRCH, EBADF, EPERM,
      // EINVAL, or other values if the user name was invalid and/or
      // not found. POSIX and Linux manpages also list certain errno
      // values (e.g., EIO, EMFILE) as definitely indicating an error.
      //
      // Hence, we check for those specific error values and return an
      // error to the caller; for any errno value not in that list, we
      // assume the user name wasn't found.
      //
      // TODO(neilc): Consider retrying on EINTR.
      if (errno != EIO &&
          errno != EINTR &&
          errno != EMFILE &&
          errno != ENFILE &&
          errno != ENOMEM) {
        return None();
      }

      return ErrnoError("Failed to get username information");
    }
  }

  abort();
}


inline Try<Nothing> setuid(uid_t uid)
{
  if (::setuid(uid) == -1) {
    return ErrnoError();
  }

  return Nothing();
}


inline Result<gid_t> getgid(const Option<std::string>& user = None())
{
  if (user.isNone()) {
    return ::getgid();
  }

  int size = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (size == -1) {
    // Initial value for buffer size.
    size = 1024;
  }

  while (true) {
    struct passwd pwd;
    struct passwd* result;
    char* buffer = new char[size];

    if (getpwnam_r(user->c_str(), &pwd, buffer, size, &result) == 0) {
      // Per POSIX, if the user name is not found, `getpwnam_r` returns
      // zero and sets `result` to the null pointer. (Linux behaves
      // differently for invalid user names; see below).
      if (result == nullptr) {
        delete[] buffer;
        return None();
      }

      // Entry found.
      gid_t gid = pwd.pw_gid;
      delete[] buffer;
      return gid;
    } else {
      delete[] buffer;

      if (errno == ERANGE) {
        // Buffer too small; enlarge it and retry.
        size *= 2;
        continue;
      }

      // According to POSIX, a non-zero return value from `getpwnam_r`
      // indicates an error. However, some versions of glibc return
      // non-zero and set errno to ENOENT, ESRCH, EBADF, EPERM,
      // EINVAL, or other values if the user name was invalid and/or
      // not found. POSIX and Linux manpages also list certain errno
      // values (e.g., EIO, EMFILE) as definitely indicating an error.
      //
      // Hence, we check for those specific error values and return an
      // error to the caller; for any errno value not in that list, we
      // assume the user name wasn't found.
      //
      // TODO(neilc): Consider retrying on EINTR.
      if (errno != EIO &&
          errno != EINTR &&
          errno != EMFILE &&
          errno != ENFILE &&
          errno != ENOMEM) {
        return None();
      }

      return ErrnoError("Failed to get username information");
    }
  }

  abort();
}


inline Try<Nothing> setgid(gid_t gid)
{
  if (::setgid(gid) == -1) {
    return ErrnoError();
  }

  return Nothing();
}


inline Try<std::vector<gid_t>> getgrouplist(const std::string& user)
{
  // TODO(jieyu): Consider adding a 'gid' parameter and avoid calling
  // getgid here. In some cases, the primary gid might be known.
  Result<gid_t> gid = os::getgid(user);
  if (!gid.isSome()) {
    return Error("Failed to get the gid of the user: " +
                 (gid.isError() ? gid.error() : "group not found"));
  }

#ifdef __APPLE__
  // TODO(gilbert): Instead of setting 'ngroups' as a large value,
  // we should figure out a way to probe 'ngroups' on OS X. Currently
  // neither '_SC_NGROUPS_MAX' nor 'NGROUPS_MAX' is appropriate,
  // because both are fixed as 16 on Darwin kernel, which is the
  // cache size.
  int ngroups = 65536;
  int gids[ngroups];
#else
  int ngroups = NGROUPS_MAX;
  gid_t gids[ngroups];
#endif
  if (::getgrouplist(user.c_str(), gid.get(), gids, &ngroups) == -1) {
    return ErrnoError();
  }

  return std::vector<gid_t>(gids, gids + ngroups);
}


inline Try<Nothing> setgroups(
    const std::vector<gid_t>& gids,
    const Option<uid_t>& uid = None())
{
  int ngroups = static_cast<int>(gids.size());
  gid_t _gids[ngroups];

  for (int i = 0; i < ngroups; i++) {
    _gids[i] = gids[i];
  }

#ifdef __APPLE__
  // Cannot simply call 'setgroups' here because it only updates
  // the list of groups in kernel cache, but not the ones in
  // opendirectoryd. Darwin kernel caches part of the groups in
  // kernel, and the rest in opendirectoryd.
  // For more detail please see:
  // https://github.com/practicalswift/osx/blob/master/src/samba/patches/support-darwin-initgroups-syscall // NOLINT
  int maxgroups = sysconf(_SC_NGROUPS_MAX);
  if (maxgroups == -1) {
    return Error("Failed to get sysconf(_SC_NGROUPS_MAX)");
  }

  if (ngroups > maxgroups) {
    ngroups = maxgroups;
  }

  if (uid.isNone()) {
    return Error(
        "The uid of the user who is associated with the group "
        "list we are setting is missing");
  }

  // NOTE: By default, the maxgroups on Darwin kernel is fixed
  // as 16. If we have more than 16 gids to set for a specific
  // user, then SYS_initgroups would send up to 16 of them to
  // kernel cache, while the rest would still be performed
  // correctly by the kernel (asking Directory Service to resolve
  // the groups membership).
  if (::syscall(SYS_initgroups, ngroups, _gids, uid.get()) == -1) {
    return ErrnoError();
  }
#else
  if (::setgroups(ngroups, _gids) == -1) {
    return ErrnoError();
  }
#endif

  return Nothing();
}


inline Result<std::string> user(Option<uid_t> uid = None())
{
  if (uid.isNone()) {
    uid = ::getuid();
  }

  int size = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (size == -1) {
    // Initial value for buffer size.
    size = 1024;
  }

  while (true) {
    struct passwd pwd;
    struct passwd* result;
    char* buffer = new char[size];

    if (getpwuid_r(uid.get(), &pwd, buffer, size, &result) == 0) {
      // getpwuid_r will return 0 but set result == nullptr if the uid is
      // not found.
      if (result == nullptr) {
        delete[] buffer;
        return None();
      }

      std::string user(pwd.pw_name);
      delete[] buffer;
      return user;
    } else {
      delete[] buffer;

      if (errno != ERANGE) {
        return ErrnoError();
      }

      // getpwuid_r set ERANGE so try again with a larger buffer.
      size *= 2;
    }
  }
}


inline Try<Nothing> su(const std::string& user)
{
  Result<gid_t> gid = os::getgid(user);
  if (gid.isError() || gid.isNone()) {
    return Error("Failed to getgid: " +
        (gid.isError() ? gid.error() : "unknown user"));
  } else if (::setgid(gid.get())) {
    return ErrnoError("Failed to set gid");
  }

  // Set the supplementary group list. We ignore EPERM because
  // performing a no-op call (switching to same group) still
  // requires being privileged, unlike 'setgid' and 'setuid'.
  if (::initgroups(user.c_str(), gid.get()) == -1 && errno != EPERM) {
    return ErrnoError("Failed to set supplementary groups");
  }

  Result<uid_t> uid = os::getuid(user);
  if (uid.isError() || uid.isNone()) {
    return Error("Failed to getuid: " +
        (uid.isError() ? uid.error() : "unknown user"));
  } else if (::setuid(uid.get())) {
    return ErrnoError("Failed to setuid");
  }

  return Nothing();
}

namespace fs {

// Returns the total disk size in bytes.
inline Try<Bytes> size(const std::string& path = "/")
{
  struct statvfs buf;
  if (::statvfs(path.c_str(), &buf) < 0) {
    return ErrnoError();
  }
  return Bytes(buf.f_blocks * buf.f_frsize);
}


// Returns relative disk usage of the file system that the given path
// is mounted at.
inline Try<double> usage(const std::string& path = "/")
{
  struct statvfs buf;
  if (statvfs(path.c_str(), &buf) < 0) {
    return ErrnoError("Error invoking statvfs on '" + path + "'");
  }
  return (double) (buf.f_blocks - buf.f_bfree) / buf.f_blocks;
}


inline Try<Nothing> symlink(
    const std::string& original,
    const std::string& link)
{
  if (::symlink(original.c_str(), link.c_str()) < 0) {
    return ErrnoError();
  }
  return Nothing();
}

} // namespace fs

namespace net {

inline struct addrinfo createAddrInfo(int socktype, int family, int flags)
{
  struct addrinfo addr;
  memset(&addr, 0, sizeof(addr));
  addr.ai_socktype = socktype;
  addr.ai_family = family;
  addr.ai_flags |= flags;

  return addr;
}


// Returns a Try of the hostname for the provided IP. If the hostname
// cannot be resolved, then a string version of the IP address is
// returned.
//
// TODO(benh): Merge with `net::hostname`.
inline Try<std::string> getHostname(const IP& ip)
{
  struct sockaddr_storage storage;
  memset(&storage, 0, sizeof(storage));

  switch (ip.family()) {
    case AF_INET: {
      struct sockaddr_in addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_addr = ip.in().get();
      addr.sin_port = 0;

      memcpy(&storage, &addr, sizeof(addr));
      break;
    }
    case AF_INET6: {
      struct sockaddr_in6 addr;
      memset(&addr, 0, sizeof(addr));
      addr.sin6_family = AF_INET6;
      addr.sin6_addr = ip.in6().get();
      addr.sin6_port = 0;

      memcpy(&storage, &addr, sizeof(addr));
      break;
    }
    default: {
      ABORT("Unsupported family type: " + stringify(ip.family()));
    }
  }

  char hostname[MAXHOSTNAMELEN];
  socklen_t length;

  if (ip.family() == AF_INET) {
    length = sizeof(struct sockaddr_in);
  } else if (ip.family() == AF_INET6) {
    length = sizeof(struct sockaddr_in6);
  } else {
    return Error("Unknown address family: " + stringify(ip.family()));
  }

  int error = getnameinfo(
      (struct sockaddr*) &storage,
      length,
      hostname,
      MAXHOSTNAMELEN,
      nullptr,
      0,
      0);

  if (error != 0) {
    return Error(std::string(gai_strerror(error)));
  }

  return std::string(hostname);
}


// Returns a Try of the IP for the provided hostname or an error if no IP is
// obtained.
inline Try<IP> getIP(const std::string& hostname, int family = AF_UNSPEC)
{
  struct addrinfo hints = createAddrInfo(SOCK_STREAM, family, 0);
  struct addrinfo* result = nullptr;

  int error = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);

  if (error != 0) {
    return Error(gai_strerror(error));
  }

  if (result->ai_addr == nullptr) {
    freeaddrinfo(result);
    return Error("No addresses found");
  }

  Try<IP> ip = IP::create(*result->ai_addr);

  if (ip.isError()) {
    freeaddrinfo(result);
    return Error("Unsupported family type");
  }

  freeaddrinfo(result);
  return ip.get();
}


// Returns the names of all the link devices in the system.
inline Try<std::set<std::string>> links()
{
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    return ErrnoError();
  }

  std::set<std::string> names;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_name != nullptr) {
      names.insert(ifa->ifa_name);
    }
  }

  freeifaddrs(ifaddr);
  return names;
}


inline Try<std::string> hostname()
{
  char host[512];

  if (gethostname(host, sizeof(host)) < 0) {
    return ErrnoError();
  }

  struct addrinfo hints = createAddrInfo(SOCK_STREAM, AF_UNSPEC, AI_CANONNAME);
  struct addrinfo* result = nullptr;

  int error = getaddrinfo(host, nullptr, &hints, &result);

  if (error != 0) {
    return Error(gai_strerror(error));
  }

  std::string hostname = result->ai_canonname;
  freeaddrinfo(result);

  return hostname;
}


// Returns a `Try` of the result of attempting to set the `hostname`.
inline Try<Nothing> setHostname(const std::string& hostname)
{
  if (sethostname(hostname.c_str(), hostname.size()) != 0) {
    return ErrnoError();
  }

  return Nothing();
}

} // namespace net 

// Abstractions around forking process trees. You can declare a
// process tree "template" using 'Fork', 'Exec', and 'Wait'. For
// example, to describe a simple "fork/exec" you can do:
//
//   Fork f = Fork(Exec("sleep 10));
//
// The command passed to an 'Exec' is run via 'sh -c'. You can
// construct more complicated templates via nesting, for example:
//
//   Fork f =
//     Fork(None(),
//          Fork(Exec("echo 'grandchild 1'")),
//          Fork(None(),
//               Fork(Exec("echo 'great-grandchild'")),
//               Exec("echo 'grandchild 2'"))
//          Exec("echo 'child'"));
//
// Note that the first argument to 'Fork' here is an optional function
// that can be invoked before forking any more children or executing a
// command. THIS FUNCTION SHOULD BE ASYNC SIGNAL SAFE.
//
// To wait for children, you can use 'Wait' instead of 'Exec', for
// example:
//
//   Fork f =
//     Fork(None(),
//          Fork(Exec("echo 'grandchild 1'")),
//          Fork(Exec("echo 'grandchild 2'")),
//          Wait());
//
// You can also omit either an 'Exec' or a 'Wait' and the forked
// process will just 'exit(0)'. For example, the following will cause
// to processes to get reparented by 'init'.
//
//   Fork f =
//     Fork(None(),
//          Fork(Exec("echo 'grandchild 1'")),
//          Fork(Exec("echo 'grandchild 2'")));
//
// A template can be instantiated by invoking the 'Fork' as a
// functor. For example, using any of the templates above we can do:
//
//   Try<ProcessTree> tree = f();
//
// It's important to note that the process tree returned represents
// the instant in time after the forking has completed but before
// 'Exec', 'Wait' or 'exit(0)' has occurred (i.e., the process tree
// will be complete).

// Forward declaration.
inline Result<Process> process(pid_t);


struct Exec
{
  Exec(const std::string& _command)
    : command(_command) {}

  const std::string command;
};


struct Wait {};


struct Fork
{
  //  -+- parent.
  Fork(const Option<void(*)()>& _function,
       const Exec& _exec)
    : function(_function),
      exec(_exec) {}

  Fork(const Exec& _exec) : exec(_exec) {}

  //  -+- parent
  //   \--- child.
  Fork(const Option<void(*)()>& _function,
       const Fork& fork1)
    : function(_function)
  {
    children.push_back(fork1);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Exec& _exec)
    : function(_function),
      exec(_exec)
  {
    children.push_back(fork1);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Wait& _wait)
    : function(_function),
      wait(_wait)
  {
    children.push_back(fork1);
  }


  // -+- parent
  //   |--- child
  //   \--- child.
  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2)
    : function(_function)
  {
    children.push_back(fork1);
    children.push_back(fork2);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2,
       const Exec& _exec)
    : function(_function),
      exec(_exec)
  {
    children.push_back(fork1);
    children.push_back(fork2);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2,
       const Wait& _wait)
    : function(_function),
      wait(_wait)
  {
    children.push_back(fork1);
    children.push_back(fork2);
  }


  // -+- parent
  //   |--- child
  //   |--- child
  //   \--- child.
  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2,
       const Fork& fork3)
    : function(_function)
  {
    children.push_back(fork1);
    children.push_back(fork2);
    children.push_back(fork3);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2,
       const Fork& fork3,
       const Exec& _exec)
    : function(_function),
      exec(_exec)
  {
    children.push_back(fork1);
    children.push_back(fork2);
    children.push_back(fork3);
  }

  Fork(const Option<void(*)()>& _function,
       const Fork& fork1,
       const Fork& fork2,
       const Fork& fork3,
       const Wait& _wait)
    : function(_function),
      wait(_wait)
  {
    children.push_back(fork1);
    children.push_back(fork2);
    children.push_back(fork3);
  }

private:
  // Represents the "tree" of descendants where each node has a
  // pointer (into shared memory) from which we can read the
  // descendants process information as well as a vector of children.
  struct Tree
  {
    // NOTE: This struct is stored in shared memory and thus cannot
    // hold any pointers to heap allocated memory.
    struct Memory {
      pid_t pid;
      pid_t parent;
      pid_t group;
      pid_t session;

      std::atomic_bool set; // Has this been initialized?
    };

    std::shared_ptr<Memory> memory;
    std::vector<Tree> children;
  };

  // We use shared memory to "share" the pids of forked descendants.
  // The benefit of shared memory over pipes is that each forked
  // process can read its descendants' pids leading to a simpler
  // implementation (with pipes, only one reader can ever read the
  // value from the pipe, forcing much more complicated coordination).
  //
  // Shared memory works like a file (in memory) that gets deleted by
  // "unlinking" it, but it won't get completely deleted until all
  // open file descriptors referencing it have been closed. Each
  // forked process has the shared memory mapped into it as well as an
  // open file descriptor, both of which should get cleaned up
  // automagically when the process exits, but we use a special
  // "deleter" (in combination with shared_ptr) in order to clean this
  // stuff up when we are actually finished using the shared memory.
  struct SharedMemoryDeleter
  {
    SharedMemoryDeleter(int _fd) : fd(_fd) {}

    void operator()(Tree::Memory* process) const
    {
      if (munmap(process, sizeof(Tree::Memory)) == -1) {
        ABORT(std::string("Failed to unmap memory: ") + os::strerror(errno));
      }
      if (::close(fd) == -1) {
        ABORT(std::string("Failed to close shared memory file descriptor: ") +
              os::strerror(errno));
      }
    }

    const int fd;
  };

  // Constructs a Tree (see above) from this fork template.
  Try<Tree> prepare() const
  {
    static std::atomic_int forks(0);

    // Each "instance" of an instantiated Fork needs a unique name for
    // creating shared memory.
    int instance = forks.fetch_add(1);

    std::string name =
      "/stout-forks-" + stringify(getpid()) + stringify(instance);

    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);

    if (fd == -1) {
      return ErrnoError("Failed to open a shared memory object");
    }

    Try<Nothing> truncated = ftruncate(fd, sizeof(Tree::Memory));
    if (truncated.isError()) {
      return Error(
          "Failed to set size of shared memory object: " + truncated.error());
    }

    void* memory = mmap(
        nullptr,
        sizeof(Tree::Memory),
        PROT_READ | PROT_WRITE, MAP_SHARED,
        fd,
        0);

    if (memory == MAP_FAILED) {
      return ErrnoError("Failed to map shared memory object");
    }

    if (shm_unlink(name.c_str()) == -1) {
      return ErrnoError("Failed to unlink shared memory object");
    }

    SharedMemoryDeleter deleter(fd);

    Tree tree;
    tree.memory = std::shared_ptr<Tree::Memory>((Tree::Memory*)memory, deleter);
    tree.memory->set.store(false);

    for (size_t i = 0; i < children.size(); i++) {
      Try<Tree> tree_ = children[i].prepare();
      if (tree_.isError()) {
        return Error(tree_.error());
      }
      tree.children.push_back(tree_.get());
    }

    return tree;
  }

  // Performs the fork, executes the function, recursively
  // instantiates any children, and then executes/waits/exits.
  pid_t instantiate(const Tree& tree) const
  {
    pid_t pid = ::fork();
    if (pid > 0) {
      return pid;
    }

    // Set the basic process information.
    Tree::Memory process;
    process.pid = getpid();
    process.parent = getppid();
    process.group = getpgid(0);
    process.session = getsid(0);
    process.set.store(true);

    // Copy it into shared memory.
    memcpy(tree.memory.get(), &process, sizeof(Tree::Memory));

    // Execute the function, if any.
    if (function.isSome()) {
      function.get()();
    }

    // Fork the children, if any.
    CHECK(children.size() == tree.children.size());
    std::set<pid_t> pids;
    for (size_t i = 0; i < children.size(); i++) {
      pids.insert(children[i].instantiate(tree.children[i]));
    }

    // Execute or wait.
    if (exec.isSome()) {
      // Execute the command (via '/bin/sh -c command').
      const char* command = exec.get().command.c_str();
      execlp("sh", "sh", "-c", command, (char*) nullptr);
      EXIT(EXIT_FAILURE)
        << "Failed to execute '" << command << "': " << os::strerror(errno);
    } else if (wait.isSome()) {
      foreach (pid_t pid, pids) {
        // TODO(benh): Check for signal interruption or other errors.
        waitpid(pid, nullptr, 0);
      }
    }

    exit(0);
    return -1;
  }

  // Waits for all of the descendant processes in the tree to update
  // their pids and constructs a ProcessTree using the Tree::Memory
  // information from shared memory.
  static Try<ProcessTree> coordinate(const Tree& tree)
  {
    // Wait for the forked process.
    // TODO(benh): Don't wait forever?
    while (!tree.memory->set.load());

    // All processes in the returned ProcessTree will have the
    // command-line of the top level process, since we construct the
    // tree using post-fork pre-exec information. So, we'll grab the
    // command of the current process here.
    Result<Process> self = os::process(getpid());

    Process process = Process(
        tree.memory->pid,
        tree.memory->parent,
        tree.memory->group,
        tree.memory->session,
        None(),
        None(),
        None(),
        self.isSome() ? self.get().command : "",
        false);

    std::list<ProcessTree> children;
    for (size_t i = 0; i < tree.children.size(); i++) {
      Try<ProcessTree> child = coordinate(tree.children[i]);
      if (child.isError()) {
        return Error(child.error());
      }
      children.push_back(child.get());
    }

    return ProcessTree(process, children);
  }

public:
  // Prepares and instantiates the process tree.
  Try<ProcessTree> operator()() const
  {
    Try<Tree> tree = prepare();

    if (tree.isError()) {
      return Error(tree.error());
    }

    Try<pid_t> pid = instantiate(tree.get());

    if (pid.isError()) {
      return Error(pid.error());
    }

    return coordinate(tree.get());
  }

private:
  Option<void(*)()> function;
  Option<const Exec> exec;
  Option<const Wait> wait;
  std::vector<Fork> children;
};


// Forward declaration.
inline Try<std::list<Process>> processes();


// Returns a process tree rooted at the specified pid using the
// specified list of processes (or an error if one occurs).
inline Try<ProcessTree> pstree(
    pid_t pid,
    const std::list<Process>& processes)
{
  std::list<ProcessTree> children;
  foreach (const Process& process, processes) {
    if (process.parent == pid) {
      Try<ProcessTree> tree = pstree(process.pid, processes);
      if (tree.isError()) {
        return Error(tree.error());
      }
      children.push_back(tree.get());
    }
  }

  foreach (const Process& process, processes) {
    if (process.pid == pid) {
      return ProcessTree(process, children);
    }
  }

  return Error("No process found at " + stringify(pid));
}


// Returns a process tree for the specified pid (or all processes if
// pid is none or the current process if pid is 0).
inline Try<ProcessTree> pstree(Option<pid_t> pid = None())
{
  if (pid.isNone()) {
    pid = 1;
  } else if (pid.get() == 0) {
    pid = getpid();
  }

  const Try<std::list<Process>> processes = os::processes();

  if (processes.isError()) {
    return Error(processes.error());
  }

  return pstree(pid.get(), processes.get());
}


// Returns the minimum list of process trees that include all of the
// specified pids using the specified list of processes.
inline Try<std::list<ProcessTree>> pstrees(
    const std::set<pid_t>& pids,
    const std::list<Process>& processes)
{
  std::list<ProcessTree> trees;

  foreach (pid_t pid, pids) {
    // First, check if the pid is already connected to one of the
    // process trees we've constructed.
    bool disconnected = true;
    foreach (const ProcessTree& tree, trees) {
      if (tree.contains(pid)) {
        disconnected = false;
        break;
      }
    }

    if (disconnected) {
      Try<ProcessTree> tree = pstree(pid, processes);
      if (tree.isError()) {
        return Error(tree.error());
      }

      // Now see if any of the existing process trees are actually
      // contained within the process tree we just created and only
      // include the disjoint process trees.
      // C++11:
      // trees = trees.filter([](const ProcessTree& t) {
      //   return tree.get().contains(t);
      // });
      std::list<ProcessTree> trees_ = trees;
      trees.clear();
      foreach (const ProcessTree& t, trees_) {
        if (tree.get().contains(t.process.pid)) {
          continue;
        }
        trees.push_back(t);
      }
      trees.push_back(tree.get());
    }
  }

  return trees;
}

// Forward declarations from os.hpp.
inline std::set<pid_t> children(pid_t, const std::list<Process>&, bool);
inline Result<Process> process(pid_t);
inline Option<Process> process(pid_t, const std::list<Process>&);
inline Try<std::list<Process>> processes();
inline Try<std::list<ProcessTree>> pstrees(
    const std::set<pid_t>&,
    const std::list<Process>&);


// Sends a signal to a process tree rooted at the specified pid.
// If groups is true, this also sends the signal to all encountered
// process groups.
// If sessions is true, this also sends the signal to all encountered
// process sessions.
// Note that processes of the group and session of the parent of the
// root process is not included unless they are part of the root
// process tree.
// Note that if the process 'pid' has exited we'll signal the process
// tree(s) rooted at pids in the group or session led by the process
// if groups = true or sessions = true, respectively.
// Returns the process trees that were successfully or unsuccessfully
// signaled. Note that the process trees can be stringified.
// TODO(benh): Allow excluding the root pid from stopping, killing,
// and continuing so as to provide a means for expressing "kill all of
// my children". This is non-trivial because of the current
// implementation.
inline Try<std::list<ProcessTree>> killtree(
    pid_t pid,
    int signal,
    bool groups = false,
    bool sessions = false)
{
  Try<std::list<Process>> processes = os::processes();

  if (processes.isError()) {
    return Error(processes.error());
  }

  Result<Process> process = os::process(pid, processes.get());

  std::queue<pid_t> queue;

  // If the root process has already terminated we'll add in any pids
  // that are in the process group originally led by pid or in the
  // session originally led by pid, if instructed.
  if (process.isNone()) {
    foreach (const Process& _process, processes.get()) {
      if (groups && _process.group == pid) {
        queue.push(_process.pid);
      } else if (sessions &&
                 _process.session.isSome() &&
                 _process.session.get() == pid) {
        queue.push(_process.pid);
      }
    }

    // Root process is not running and no processes found in the
    // process group or session so nothing we can do.
    if (queue.empty()) {
      return std::list<ProcessTree>();
    }
  } else {
    // Start the traversal from pid as the root.
    queue.push(pid);
  }

  struct {
    std::set<pid_t> pids;
    std::set<pid_t> groups;
    std::set<pid_t> sessions;
    std::list<Process> processes;
  } visited;

  // If we are following groups and/or sessions then we try and make
  // the group and session of the parent process "already visited" so
  // that we don't kill "up the tree". This can only be done if the
  // process is present.
  if (process.isSome() && (groups || sessions)) {
    Option<Process> parent =
      os::process(process.get().parent, processes.get());

    if (parent.isSome()) {
      if (groups) {
        visited.groups.insert(parent.get().group);
      }
      if (sessions && parent.get().session.isSome()) {
        visited.sessions.insert(parent.get().session.get());
      }
    }
  }

  while (!queue.empty()) {
    pid_t pid = queue.front();
    queue.pop();

    if (visited.pids.count(pid) != 0) {
      continue;
    }

    // Make sure this process still exists.
    process = os::process(pid);

    if (process.isError()) {
      return Error(process.error());
    } else if (process.isNone()) {
      continue;
    }

    // Stop the process to keep it from forking while we are killing
    // it since a forked child might get re-parented by init and
    // become impossible to find.
    kill(pid, SIGSTOP);

    visited.pids.insert(pid);
    visited.processes.push_back(process.get());

    // Now refresh the process list knowing that the current process
    // can't fork any more children (since it's stopped).
    processes = os::processes();

    if (processes.isError()) {
      return Error(processes.error());
    }

    // Enqueue the children for visiting.
    foreach (pid_t child, os::children(pid, processes.get(), false)) {
      queue.push(child);
    }

    // Now "visit" the group and/or session of the current process.
    if (groups) {
      pid_t group = process.get().group;
      if (visited.groups.count(group) == 0) {
        foreach (const Process& process, processes.get()) {
          if (process.group == group) {
            queue.push(process.pid);
          }
        }
        visited.groups.insert(group);
      }
    }

    // If we do not have a session for the process, it's likely
    // because the process is a zombie on OS X. This implies it has
    // not been reaped and thus is located somewhere in the tree we
    // are trying to kill. Therefore, we should discover it from our
    // tree traversal, or through its group (which is always present).
    if (sessions && process.get().session.isSome()) {
      pid_t session = process.get().session.get();
      if (visited.sessions.count(session) == 0) {
        foreach (const Process& process, processes.get()) {
          if (process.session.isSome() && process.session.get() == session) {
            queue.push(process.pid);
          }
        }
        visited.sessions.insert(session);
      }
    }
  }

  // Now that all processes are stopped, we send the signal.
  foreach (pid_t pid, visited.pids) {
    kill(pid, signal);
  }

  // There is a concern that even though some process is stopped,
  // sending a signal to any of its children may cause a SIGCLD to
  // be delivered to it which wakes it up (or any other signal maybe
  // delivered). However, from the Open Group standards on "Signal
  // Concepts":
  //
  //   "While a process is stopped, any additional signals that are
  //    sent to the process shall not be delivered until the process
  //    is continued, except SIGKILL which always terminates the
  //    receiving process."
  //
  // In practice, this is not what has been witnessed. Rather, a
  // process that has been stopped will respond to SIGTERM, SIGINT,
  // etc. That being said, we still continue the process below in the
  // event that it doesn't terminate from the sending signal but it
  // also doesn't get continued (as per the specifications above).

  // Try and continue the processes in case the signal is
  // non-terminating but doesn't continue the process.
  foreach (pid_t pid, visited.pids) {
    kill(pid, SIGCONT);
  }

  // Return the process trees representing the visited pids.
  return pstrees(visited.pids, visited.processes);
}

} // namespace os
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_STOUT_OS_POSIX_H_