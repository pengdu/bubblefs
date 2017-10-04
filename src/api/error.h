
#ifndef BUBBLEFS_API_ERROR_H_
#define BUBBLEFS_API_ERROR_H_

namespace bubblefs {
namespace bfs {
  
enum StatusCode {
  OK = 0,
  BAD_PARAMETER = -1,
  PERMISSION_DENIED = -2,
  NOT_ENOUGH_QUOTA = -3,
  NETWORK_UNAVAILABLE = -4,
  TIMEOUT = -5,
  NOT_ENOUGH_SPACE = -6,
  OVERLOAD = -7,
  META_NOT_AVAILABLE = -8,
  UNKNOWN_ERROR = -9
};

const char* StrError(int error_code);

} // namespace bfs
} // namespace bubblefs

#endif // BUBBLEFS_API_ERROR_H_