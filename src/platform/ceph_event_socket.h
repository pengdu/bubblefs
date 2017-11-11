/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2015 XSky <haomai@xsky.com>
 *
 * Author: Haomai Wang <haomaiwang@gmail.com>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 *
 */

#ifndef BUBBLEFS_PLATFORM_CEPH_EVENT_SOCKET_H_
#define BUBBLEFS_PLATFORM_CEPH_EVENT_SOCKET_H_

#include <errno.h>
#include <unistd.h>

namespace bubblefs {
namespace myceph {
  
constexpr int EVENT_SOCKET_TYPE_NONE = 0;
constexpr int EVENT_SOCKET_TYPE_PIPE = 1;
constexpr int EVENT_SOCKET_TYPE_EVENTFD = 2;  
  
class EventSocket {
  int socket;
  int type;

 public:
  EventSocket(): socket(-1), type(EVENT_SOCKET_TYPE_NONE) {}
  bool is_valid() const { return socket != -1; }
  int init(int fd, int t) {
    switch (t) {
      case EVENT_SOCKET_TYPE_PIPE:
      case EVENT_SOCKET_TYPE_EVENTFD:
      {
        socket = fd;
        type = t;
        return 0;
      }
    }
    return -EINVAL;
  }
  int notify() {
    int ret;
    switch (type) {
      case EVENT_SOCKET_TYPE_PIPE:
      {
        char buf[1];
        buf[0] = 'i';
        ret = write(socket, buf, 1);
        if (ret < 0)
          ret = -errno;
        else
          ret = 0;
        break;
      }
      case EVENT_SOCKET_TYPE_EVENTFD:
      {
        uint64_t value = 1;
        ret = write(socket, &value, sizeof (value));
        if (ret < 0)
          ret = -errno;
        else
          ret = 0;
        break;
      }
      default:
      {
        ret = -1;
        break;
      }
    }
    return ret;
  }
};

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_CEPH_EVENT_SOCKET_H_