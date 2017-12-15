/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/utils/zmq_helper.h

#ifndef BUBBLEFS_UTILS_CAFFE2_ZMQ_HELPER_H_
#define BUBBLEFS_UTILS_CAFFE2_ZMQ_HELPER_H_

#include "platform/base_error.h"

#include "zmq.h"

namespace bubblefs {
namespace mycaffe2 {

class ZmqContext {
 public:
  explicit ZmqContext(int io_threads) : ptr_(zmq_ctx_new()) {
    PANIC_ENFORCE(ptr_ != nullptr, "Failed to create zmq context.");
    int rc = zmq_ctx_set(ptr_, ZMQ_IO_THREADS, io_threads);
    PANIC_ENFORCE_EQ(rc, 0);
    rc = zmq_ctx_set(ptr_, ZMQ_MAX_SOCKETS, ZMQ_MAX_SOCKETS_DFLT);
    PANIC_ENFORCE_EQ(rc, 0);
  }
  ~ZmqContext() {
    int rc = zmq_ctx_destroy(ptr_);
    PANIC_ENFORCE_EQ(rc, 0);
  }

  void* ptr() { return ptr_; }

 private:
  void* ptr_;

  DISABLE_COPY_AND_ASSIGN(ZmqContext);
};

class ZmqMessage {
 public:
  ZmqMessage() {
    int rc = zmq_msg_init(&msg_);
    PANIC_ENFORCE_EQ(rc, 0);
  }

  ~ZmqMessage() {
    int rc = zmq_msg_close(&msg_);
    PANIC_ENFORCE_EQ(rc, 0);
  }

  zmq_msg_t* msg() { return &msg_; }

  void* data() { return zmq_msg_data(&msg_); }
  size_t size() { return zmq_msg_size(&msg_); }

 private:
  zmq_msg_t msg_;
  DISABLE_COPY_AND_ASSIGN(ZmqMessage);
};

class ZmqSocket {
 public:
  explicit ZmqSocket(int type)
      : context_(1), ptr_(zmq_socket(context_.ptr(), type)) {
    PANIC_ENFORCE(ptr_ != nullptr, "Faild to create zmq socket.");
  }

  ~ZmqSocket() {
    int rc = zmq_close(ptr_);
    PANIC_ENFORCE_EQ(rc, 0);
  }

  void Bind(const string& addr) {
    int rc = zmq_bind(ptr_, addr.c_str());
    PANIC_ENFORCE_EQ(rc, 0);
  }

  void Unbind(const string& addr) {
    int rc = zmq_unbind(ptr_, addr.c_str());
    PANIC_ENFORCE_EQ(rc, 0);
  }

  void Connect(const string& addr) {
    int rc = zmq_connect(ptr_, addr.c_str());
    PANIC_ENFORCE_EQ(rc, 0);
  }

  void Disconnect(const string& addr) {
    int rc = zmq_disconnect(ptr_, addr.c_str());
    PANIC_ENFORCE_EQ(rc, 0);
  }

  int Send(const string& msg, int flags) {
    int nbytes = zmq_send(ptr_, msg.c_str(), msg.size(), flags);
    if (nbytes) {
      return nbytes;
    } else if (zmq_errno() == EAGAIN) {
      return 0;
    } else {
      PANIC("Cannot send zmq message. Error number: %d", zmq_errno());
      return 0;
    }
  }

  int SendTillSuccess(const string& msg, int flags) {
    PANIC_ENFORCE(msg.size(), "You cannot send an empty message.");
    int nbytes = 0;
    do {
      nbytes = Send(msg, flags);
    } while (nbytes == 0);
    return nbytes;
  }

  int Recv(ZmqMessage* msg) {
    int nbytes = zmq_msg_recv(msg->msg(), ptr_, 0);
    if (nbytes >= 0) {
      return nbytes;
    } else if (zmq_errno() == EAGAIN || zmq_errno() == EINTR) {
      return 0;
    } else {
      PANIC("Cannot receive zmq message. Error number: %d", zmq_errno());
      return 0;
    }
  }

  int RecvTillSuccess(ZmqMessage* msg) {
    int nbytes = 0;
    do {
      nbytes = Recv(msg);
    } while (nbytes == 0);
    return nbytes;
  }

 private:
  ZmqContext context_;
  void* ptr_;
};

}  // namespace mycaffe2
}  // namespace bubblfs

#endif  // BUBBLEFS_UTILS_CAFFE2_ZMQ_HELPER_H_