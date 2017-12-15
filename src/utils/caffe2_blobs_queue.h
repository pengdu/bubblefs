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

// caffe2/caffe2/queue/blobs_queue.h
// caffe2/caffe2/queue/blobs_queue.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_BLOBS_QUEUE_H_
#defineBUBBLEFS_UTILS_CAFFE2_BLOBS_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/caffe2_stats.h"
#include "utils/caffe2_string_utils.h"
#include "utils/caffe2_tensor.h"
#include "utils/caffe2_workspace.h"

namespace bubblefs {
namespace mycaffe2 {

// A thread-safe, bounded, blocking queue.
// Modelled as a circular buffer.

// Containing blobs are owned by the workspace.
// On read, we swap out the underlying data for the blob passed in for blobs

class BlobsQueue : public std::enable_shared_from_this<BlobsQueue> {
 public:
  BlobsQueue(
      Workspace* ws,
      const string& queueName,
      size_t capacity,
      size_t numBlobs,
      bool enforceUniqueName,
      std::vector<std::string>& fieldNames);

  ~BlobsQueue() {
    close();
  }

  bool blockingRead(
      const std::vector<Blob*>& inputs,
      float timeout_secs = 0.0f);
  bool tryWrite(const std::vector<Blob*>& inputs);
  bool blockingWrite(const std::vector<Blob*>& inputs);
  void close();
  size_t getNumBlobs() const {
    return numBlobs_;
  }

 private:
  bool canWrite();
  void doWrite(const std::vector<Blob*>& inputs);

  std::atomic<bool> closing_{false};

  size_t numBlobs_;
  std::mutex mutex_; // protects all variables in the class.
  std::condition_variable cv_;
  int64_t reader_{0};
  int64_t writer_{0};
  std::vector<std::vector<Blob*>> queue_;
  const std::string name_;

  struct QueueStats {
    CAFFE_STAT_CTOR(QueueStats);
    CAFFE_EXPORTED_STAT(queue_balance);
    CAFFE_EXPORTED_STAT(queue_dequeued_records);
    CAFFE_DETAILED_EXPORTED_STAT(queue_dequeued_bytes);
  } stats_;
};

// Constants for user tracepoints
static constexpr int SDT_NONBLOCKING_OP = 0;
static constexpr int SDT_BLOCKING_OP = 1;
static constexpr uint64_t SDT_TIMEOUT = (uint64_t)-1;
static constexpr uint64_t SDT_ABORT = (uint64_t)-2;
static constexpr uint64_t SDT_CANCEL = (uint64_t)-3;

BlobsQueue::BlobsQueue(
    Workspace* ws,
    const std::string& queueName,
    size_t capacity,
    size_t numBlobs,
    bool enforceUniqueName,
    const std::vector<std::string>& fieldNames)
    : numBlobs_(numBlobs), name_(queueName), stats_(queueName) {
  if (!fieldNames.empty()) {
    PANIC_ENFORCE_EQ(
        fieldNames.size(), numBlobs) // "Wrong number of fieldNames provided.");
    stats_.queue_dequeued_bytes.setDetails(fieldNames);
  }
  queue_.reserve(capacity);
  for (auto i = 0; i < capacity; ++i) {
    std::vector<Blob*> blobs;
    blobs.reserve(numBlobs);
    for (auto j = 0; j < numBlobs; ++j) {
      const auto blobName = queueName + "_" + to_string(i) + "_" + to_string(j);
      if (enforceUniqueName) {
        PANIC_ENFORCE(
            !ws->GetBlob(blobName),
            "Queue internal blob already exists: ",
            blobName);
      }
      blobs.push_back(ws->CreateBlob(blobName));
    }
    queue_.push_back(blobs);
  }
  PANIC_ENFORCE_EQ(queue_.size(), capacity);
}

bool BlobsQueue::blockingRead(
    const std::vector<Blob*>& inputs,
    float timeout_secs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_read_start, name, (void*)this, SDT_BLOCKING_OP);
  std::unique_lock<std::mutex> g(mutex_);
  auto canRead = [this]() {
    PANIC_ENFORCE_LE(reader_, writer_);
    return reader_ != writer_;
  };
  CAFFE_EVENT(stats_, queue_balance, -1);
  if (timeout_secs > 0) {
    std::chrono::milliseconds timeout_ms(int(timeout_secs * 1000));
    cv_.wait_for(
        g, timeout_ms, [this, canRead]() { return closing_ || canRead(); });
  } else {
    cv_.wait(g, [this, canRead]() { return closing_ || canRead(); });
  }
  if (!canRead()) {
    if (timeout_secs > 0 && !closing_) {
      //LOG(ERROR) << "DequeueBlobs timed out in " << timeout_secs << " secs";
      CAFFE_SDT(queue_read_end, name, (void*)this, SDT_TIMEOUT);
    } else {
      CAFFE_SDT(queue_read_end, name, (void*)this, SDT_CANCEL);
    }
    return false;
  }
  //DCHECK(canRead());
  auto& result = queue_[reader_ % queue_.size()];
  PANIC_ENFORCE(inputs.size() >= result.size(), "");
  for (auto i = 0; i < result.size(); ++i) {
    auto bytes = BlobStat::sizeBytes(*result[i]);
    CAFFE_EVENT(stats_, queue_dequeued_bytes, bytes, i);
    using std::swap;
    swap(*(inputs[i]), *(result[i]));
  }
  CAFFE_SDT(queue_read_end, name, (void*)this, writer_ - reader_);
  CAFFE_EVENT(stats_, queue_dequeued_records);
  ++reader_;
  cv_.notify_all();
  return true;
}

bool BlobsQueue::tryWrite(const std::vector<Blob*>& inputs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_write_start, name, (void*)this, SDT_NONBLOCKING_OP);
  std::unique_lock<std::mutex> g(mutex_);
  if (!canWrite()) {
    CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
    return false;
  }
  CAFFE_EVENT(stats_, queue_balance, 1);
  //DCHECK(canWrite());
  doWrite(inputs);
  return true;
}

bool BlobsQueue::blockingWrite(const std::vector<Blob*>& inputs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_write_start, name, (void*)this, SDT_BLOCKING_OP);
  std::unique_lock<std::mutex> g(mutex_);
  CAFFE_EVENT(stats_, queue_balance, 1);
  cv_.wait(g, [this]() { return closing_ || canWrite(); });
  if (!canWrite()) {
    CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
    return false;
  }
  //DCHECK(canWrite());
  doWrite(inputs);
  return true;
}

void BlobsQueue::close() {
  closing_ = true;

  std::lock_guard<std::mutex> g(mutex_);
  cv_.notify_all();
}

bool BlobsQueue::canWrite() {
  // writer is always within [reader, reader + size)
  // we can write if reader is within [reader, reader + size)
  PANIC_ENFORCE_LE(reader_, writer_);
  PANIC_ENFORCE_LE(writer_, reader_ + queue_.size());
  return writer_ != reader_ + queue_.size();
}

void BlobsQueue::doWrite(const std::vector<Blob*>& inputs) {
  auto& result = queue_[writer_ % queue_.size()];
  PANIC_ENFORCE(inputs.size() >= result.size(), "");
  const auto& name = name_.c_str();
  for (auto i = 0; i < result.size(); ++i) {
    using std::swap;
    swap(*(inputs[i]), *(result[i]));
  }
  CAFFE_SDT(
      queue_write_end, name, (void*)this, reader_ + queue_.size() - writer_);
  ++writer_;
  cv_.notify_all();
}

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_BLOBS_QUEUE_H_