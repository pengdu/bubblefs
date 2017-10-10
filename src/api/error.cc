
#include "api/error.h"

namespace bubblefs {
namespace bfs {  

int32_t GetErrorCode(ErrorCode stat) {
    if (stat < 100) {
        if (stat == 0) {
            return OK;
        } else {
            return UNKNOWN_ERROR;
        }
    }
    switch (stat / 100) {
        case 1:
            return BAD_PARAMETER;
        case 2:
            return PERMISSION_DENIED;
        case 3:
            return NOT_ENOUGH_QUOTA;
        case 4:
            return NETWORK_UNAVAILABLE;
        case 5:
            return TIMEOUT;
        case 6:
            return NOT_ENOUGH_SPACE;
        case 7:
            return OVERLOAD;
        case 8:
            return META_NOT_AVAILABLE;
        default:
            return UNKNOWN_ERROR;
    }
}  
 
#define MAKE_CASE(name) case name: return (#name)

const char* StrError(int error_code) {
    switch (error_code) {
        MAKE_CASE(OK);
        MAKE_CASE(BAD_PARAMETER);
        MAKE_CASE(PERMISSION_DENIED);
        MAKE_CASE(NOT_ENOUGH_QUOTA);
        MAKE_CASE(NETWORK_UNAVAILABLE);
        MAKE_CASE(TIMEOUT);
        MAKE_CASE(NOT_ENOUGH_SPACE);
        MAKE_CASE(OVERLOAD);
        MAKE_CASE(META_NOT_AVAILABLE);
    }
    return "UNKNOWN_ERROR";
} 
 
} // namespace bfs  
} // namespace bubblefs