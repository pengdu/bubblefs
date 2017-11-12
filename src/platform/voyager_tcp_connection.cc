// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/tcp_connection.cc

#include "platform/voyager_tcp_connection.h"
#include <errno.h>
#include <unistd.h>
#include "platform/voyager_logging.h"
#include "utils/voyager_dispatch.h"
#include "utils/voyager_eventloop.h"
#include "utils/stringpiece.h"

namespace bubblefs {
namespace myvoyager {

TcpConnection::TcpConnection(const std::string& name, EventLoop* ev, int fd,
                             const SockAddr& local, const SockAddr& peer)
    : name_(name),
      eventloop_(VOYAGER_CHECK_NOTNULL(ev)),
      socket_(fd),
      local_addr_(local),
      peer_addr_(peer),
      state_(kConnecting),
      dispatch_(new Dispatch(ev, fd)),
      context_(nullptr),
      high_water_mark_(64 * 1024 * 1024) {
  dispatch_->SetReadCallback(std::bind(&TcpConnection::HandleRead, this));
  dispatch_->SetWriteCallback(std::bind(&TcpConnection::HandleWrite, this));
  dispatch_->SetCloseCallback(std::bind(&TcpConnection::HandleClose, this));
  dispatch_->SetErrorCallback(std::bind(&TcpConnection::HandleError, this));
  socket_.SetNonBlockAndCloseOnExec(true);
  socket_.SetKeepAlive(true);
  socket_.SetTcpNoDelay(true);
  VOYAGER_LOG(DEBUG) << "TcpConnection::TcpConnection [" << name_ << "] at "
                     << this << " fd=" << fd;
}

TcpConnection::~TcpConnection() {
  VOYAGER_LOG(DEBUG) << "TcpConnection::~TcpConnection [" << name_ << "] at "
                     << this << " fd=" << dispatch_->Fd()
                     << " ConnectState=" << StateToString();
}

void TcpConnection::StartWorking() {
  eventloop_->AssertInMyLoop();
  assert(state_ == kConnecting);
  state_ = kConnected;
  TcpConnectionPtr ptr(shared_from_this());
  dispatch_->Tie(ptr);
  dispatch_->EnableRead();
  eventloop_->AddConnection(ptr);
  if (connection_cb_) {
    connection_cb_(ptr);
  }
}

void TcpConnection::StartRead() {
  TcpConnectionPtr ptr(shared_from_this());
  eventloop_->RunInLoop([ptr]() {
    if (!ptr->dispatch_->IsReading()) {
      ptr->dispatch_->EnableRead();
    }
  });
}

void TcpConnection::StopRead() {
  TcpConnectionPtr ptr(shared_from_this());
  eventloop_->RunInLoop([ptr]() {
    if (ptr->dispatch_->IsReading()) {
      ptr->dispatch_->DisableRead();
    }
  });
}

void TcpConnection::ShutDown() {
  ConnectState expected = kConnected;
  if (state_.compare_exchange_weak(expected, kDisconnecting)) {
    TcpConnectionPtr ptr(shared_from_this());
    eventloop_->RunInLoop([ptr]() {
      if (!ptr->dispatch_->IsWriting()) {
        ptr->socket_.ShutDownWrite();
      }
    });
  }
}

void TcpConnection::ForceClose() {
  ConnectState expected = kConnected;
  if (state_.compare_exchange_weak(expected, kDisconnecting) ||
      state_ == kDisconnecting) {
    TcpConnectionPtr ptr(shared_from_this());
    eventloop_->QueueInLoop([ptr]() {
      if (ptr->state_ == kConnected || ptr->state_ == kDisconnecting) {
        ptr->HandleClose();
      }
    });
  }
}

void TcpConnection::HandleRead() {
  eventloop_->AssertInMyLoop();
  ssize_t n = readbuf_.ReadV(dispatch_->Fd());
  if (n > 0) {
    if (message_cb_) {
      message_cb_(shared_from_this(), &readbuf_);
    }
  } else if (n == 0) {
    HandleClose();
  } else {
    if (errno == EPIPE || errno == ECONNRESET) {
      HandleClose();
    }
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
      VOYAGER_LOG(ERROR) << "TcpConnection::HandleRead [" << name_
                         << "] - readv: " << strerror(errno);
    }
  }
}

void TcpConnection::HandleWrite() {
  eventloop_->AssertInMyLoop();
  if (dispatch_->IsWriting()) {
    ssize_t n =
        ::write(dispatch_->Fd(), writebuf_.Peek(), writebuf_.ReadableSize());
    if (n >= 0) {
      writebuf_.Retrieve(static_cast<size_t>(n));
      if (writebuf_.ReadableSize() == 0) {
        dispatch_->DisableWrite();
        if (writecomplete_cb_) {
          writecomplete_cb_(shared_from_this());
        }
        if (state_ == kDisconnecting) {
          HandleClose();
        }
      }
    } else {
      if (errno == EPIPE || errno == ECONNRESET) {
        HandleClose();
      }
      if (errno != EWOULDBLOCK && errno != EAGAIN) {
        VOYAGER_LOG(ERROR) << "TcpConnection::HandleWrite [" << name_
                           << "] - write: " << strerror(errno);
      }
    }
  } else {
    VOYAGER_LOG(INFO) << "TcpConnection::HandleWrite [" << name_
                      << "] - fd=" << dispatch_->Fd()
                      << " is down, no more writing";
  }
}

void TcpConnection::HandleClose() {
  eventloop_->AssertInMyLoop();
  assert(state_ == kConnected || state_ == kDisconnecting);
  state_ = kDisconnected;
  dispatch_->DisableAll();
  dispatch_->RemoveEvents();
  eventloop_->RemoveConnection(shared_from_this());
  if (close_cb_) {
    close_cb_(shared_from_this());
  }
}

void TcpConnection::HandleError() {
  int result = socket_.CheckSocketError();
  if (result != 0) {
    VOYAGER_LOG(ERROR) << "TcpConnection::HandleError [" << name_ << "] - "
                       << strerror(result);
  }
}

void TcpConnection::SendMessage(std::string&& message) {
  if (state_ == kConnected) {
    if (eventloop_->IsInMyLoop()) {
      SendInLoop(message.data(), message.size());
    } else {
      eventloop_->RunInLoop(std::bind(&TcpConnection::Send, shared_from_this(),
                                      std::move(message)));
    }
  }
}

void TcpConnection::SendMessage(const Slice& message) {
  if (state_ == kConnected) {
    if (eventloop_->IsInMyLoop()) {
      SendInLoop(message.data(), message.size());
    } else {
      eventloop_->RunInLoop(std::bind(&TcpConnection::Send, shared_from_this(),
                                      message.ToString()));
    }
  }
}

void TcpConnection::SendMessage(Buffer* message) {
  VOYAGER_CHECK_NOTNULL(message);
  if (state_ == kConnected) {
    if (eventloop_->IsInMyLoop()) {
      SendInLoop(message->Peek(), message->ReadableSize());
      message->RetrieveAll();
    } else {
      eventloop_->RunInLoop(std::bind(&TcpConnection::Send, shared_from_this(),
                                      message->RetrieveAllAsString()));
    }
  }
}

void TcpConnection::Send(const std::string& s) {
  SendInLoop(s.data(), s.size());
}

void TcpConnection::SendInLoop(const void* data, size_t size) {
  eventloop_->AssertInMyLoop();
  if (state_ == kDisconnected) {
    VOYAGER_LOG(WARN) << "TcpConnection::SendInLoop[" << name_ << "]"
                      << "has disconnected, give up writing.";
    return;
  }

  ssize_t nwrote = 0;
  size_t remaining = size;
  bool fault = false;

  if (!dispatch_->IsWriting() && writebuf_.ReadableSize() == 0) {
    nwrote = ::write(dispatch_->Fd(), data, size);
    if (nwrote >= 0) {
      remaining = size - static_cast<size_t>(nwrote);
      if (remaining == 0 && writecomplete_cb_) {
        writecomplete_cb_(shared_from_this());
      }
    } else {
      nwrote = 0;
      if (errno == EPIPE || errno == ECONNRESET) {
        HandleClose();
        fault = true;
      }
      if (errno != EWOULDBLOCK && errno != EAGAIN) {
        VOYAGER_LOG(ERROR) << "TcpConnection::SendInLoop [" << name_
                           << "] - write: " << strerror(errno);
      }
    }
  }

  assert(remaining <= size);
  if (!fault && remaining > 0) {
    size_t old = writebuf_.ReadableSize();
    if (high_water_mark_cb_ && old < high_water_mark_ &&
        (old + remaining) >= high_water_mark_) {
      eventloop_->QueueInLoop(
          std::bind(high_water_mark_cb_, shared_from_this(), old + remaining));
    }
    writebuf_.Append(static_cast<const char*>(data) + nwrote, remaining);
    if (!dispatch_->IsWriting()) {
      dispatch_->EnableWrite();
    }
  }
}

std::string TcpConnection::StateToString() const {
  const char* type;
  switch (state_.load(std::memory_order_relaxed)) {
    case kDisconnected:
      type = "Disconnected";
      break;
    case kDisconnecting:
      type = "Disconnecting";
      break;
    case kConnected:
      type = "Connected";
      break;
    case kConnecting:
      type = "Connecting";
      break;
    default:
      type = "Unknown State";
      break;
  }
  std::string result(type);
  return result;
}

}  // namespace myvoyager
}  // namespace bubblefs