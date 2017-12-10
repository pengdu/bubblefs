
// simcc/simcc/misc/dynlib.h

#ifndef BUBBLEFS_UTILS_SIMCC_DYNLIB_H_
#define BUBBLEFS_UTILS_SIMCC_DYNLIB_H_

#include "platform/types.h"

namespace bubblefs {
namespace mysimcc {

// Dynamic library. It is used load and find symbol in the library.
// @remark You should keep the object alive before unload the loaded library.
class DynLib {
public:
#ifdef H_OS_WINDOWS
    typedef HMODULE Handler;
#else
    typedef void* Handler;
#endif

    // @param dll_path The full path for a library. e.g. "e:/project/hph/bin/test.dll", "/root/bin/test.so"
    DynLib(const string& dll_path);
    ~DynLib();

    // Actively load the library.
    // @return false if failed to load.
    bool Load();

    // Unload the library.
    // @return false if failed to unload
    bool Unload();

    // Query whether the library has been loaded.
    bool IsLoaded();

    // Get full path of the library
    const string& path(void) const {
        return dll_path_;
    }

    // Gets symbol in the library.
    // @return NULL if find nothing.
    void* GetSymbol(const string& func_name);

    // Gets the last loading error. It is used get the error message
    // when failed to load or unload library.
    const string& GetLastError() const {
        return error_;
    }

private:
    Handler handler_;
    string dll_path_;
    string error_;

private:
    // Generate library load error. It generated from system.
    string GetError();
};

} // namespace mysimcc
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SIMCC_DYNLIB_H_