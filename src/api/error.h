
#ifndef BUBBLEFS_API_ERROR_H_
#define BUBBLEFS_API_ERROR_H_

namespace bubblefs {
namespace bfs {
  
const int OK = 0;
const int BAD_PARAMETER = -1;
const int PERMISSION_DENIED = -2;
const int NOT_ENOUGH_QUOTA = -3;
const int NETWORK_UNAVAILABLE = -4;
const int TIMEOUT = -5;
const int NOT_ENOUGH_SPACE = -6;
const int OVERLOAD = -7;
const int META_NOT_AVAILABLE = -8;
const int UNKNOWN_ERROR = -9;

const char* StrError(int error_code);

} // namespace bfs
} // namespace bubblefs

#endif // BUBBLEFS_API_ERROR_H_