
#include "utils/brpc_randuint64.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include "platform/brpc_eintr_wrapper.h"
#include "platform/logging.h"
#include "platform/macros.h"
#include "platform/types.h"
#include "utils/brpc_lazy_instance.h"

namespace bubblefs {
namespace mybrpc {

namespace {

// We keep the file descriptor for /dev/urandom around so we don't need to
// reopen it (which is expensive), and since we may not even be able to reopen
// it if we are later put in a sandbox. This class wraps the file descriptor so
// we can use LazyInstance to handle opening it on the first access.
class URandomFd {
 public:
  URandomFd() : fd_(open("/dev/urandom", O_RDONLY)) {
    DCHECK_GE(fd_, 0) << "Cannot open /dev/urandom: " << errno;
  }

  ~URandomFd() { close(fd_); }

  int fd() const { return fd_; }

 private:
  const int fd_;
};

LazyInstance<URandomFd>::Leaky g_urandom_fd = LAZY_INSTANCE_INITIALIZER;

}  // namespace

bool ReadFromFD(int fd, char* buffer, size_t bytes) {
  size_t total_read = 0;
  while (total_read < bytes) {
    ssize_t bytes_read =
        HANDLE_EINTR(read(fd, buffer + total_read, bytes - total_read));
    if (bytes_read <= 0)
      break;
    total_read += bytes_read;
  }
  return total_read == bytes;
}

void RandBytes(void* output, size_t output_length) {
  const int urandom_fd = g_urandom_fd.Pointer()->fd();
  const bool success =
      ReadFromFD(urandom_fd, static_cast<char*>(output), output_length);
  CHECK(success);
}

// NOTE: This function must be cryptographically secure. http://crbug.com/140076
uint64_t RandUint64() {
  uint64_t number;
  RandBytes(&number, sizeof(number));
  return number;
}

int GetUrandomFD(void) {
  return g_urandom_fd.Pointer()->fd();
}

} // namespace mybrpc
} // namespace bubblefs