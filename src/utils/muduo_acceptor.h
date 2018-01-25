// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is an internal header file, you should not include this.

// muduo/muduo/net/Acceptor.h

#ifndef BUBBLEFS_UTILS_MUDUO_NET_ACCEPTOR_H_
#define BUBBLEFS_UTILS_MUDUO_NET_ACCEPTOR_H_

#include <functional>
#include "utils/muduo_channel.h"
#include "utils/muduo_socket.h"

namespace bubblefs {
namespace mymuduo {
namespace net {

class EventLoop;
class InetAddress;

///
/// Acceptor of incoming TCP connections.
///
class Acceptor
{
 public:
  typedef std::function<void (int sockfd,
                        const InetAddress&)> NewConnectionCallback;

  Acceptor(EventLoop* loop, const InetAddress& listenAddr, bool reuseport);
  ~Acceptor();

  void setNewConnectionCallback(const NewConnectionCallback& cb)
  { newConnectionCallback_ = cb; }

  bool listenning() const { return listenning_; }
  void listen();

 private:
  void handleRead();

  EventLoop* loop_;
  Socket acceptSocket_;
  Channel acceptChannel_;
  NewConnectionCallback newConnectionCallback_;
  bool listenning_;
  int idleFd_;
};

} // namespace net
} // namespace mymuduo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MUDUO_NET_ACCEPTOR_H_