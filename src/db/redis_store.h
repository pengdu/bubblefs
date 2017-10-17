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

// caffe2/caffe2/distributed/store_handler.h
// caffe2/caffe2/distributed/redis_store_handler.h

#ifndef BUBBLEFS_DB_REDIS_STORE_H_
#define BUBBLEFS_DB_REDIS_STORE_H_

extern "C" {
#include "hiredis/hiredis.h"
#include "hiredis/read.h"
}

#include <stdint.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include "utils/status.h"

namespace bubblefs {
namespace db {
  
class RedisStore {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(30);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();
      
  RedisStore() : redis_(nullptr) { }
   
  virtual ~RedisStore();

  virtual Status Open(std::string& host, int port, std::string& prefix);
  
  /*
   * Set data for the key if it doesn't exist.
   * If the key exists the data should be the same as the existing key.
   */
  virtual Status Set(const std::string& name, const std::string& data);
  
  /*
   * Get the data for the key.
   */
  virtual Status Get(const std::string& name, std::string* data);
  
  /*
   * Get the data for the key.
   * The call should wait until the key is stored with default timeout
   * and return data if set else fail.
   */
  virtual Status TimedGet(const std::string& name, 
                          std::string* data,
                          const std::chrono::milliseconds& timeout = kDefaultTimeout);

  /*
   * Does an atomic add operation on the key and returns the latest updated
   * value.
   * Note: To access the current value for this counter call with value = 0
   */
  virtual Status Add(const std::string& name, int64_t value, int64_t *latest_value);

  /*
   * Check if a keys exist in the store.
   */
  virtual bool Exist(const std::vector<std::string>& names);

  /*
   * Wait for Keys to be stored.
   */
  virtual Status WaitExist(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout);
  
 private:
  std::string CompoundKey(const std::string& name);

 private:
  std::string host_;
  int port_;
  std::string prefix_;

  redisContext* redis_;
};

struct RedisStoreTimeoutException : public std::runtime_error {
  RedisStoreTimeoutException() = default;
  explicit RedisStoreTimeoutException(const std::string& msg)
      : std::runtime_error(msg) {}
};

} // namespace db
} // namespace bubblefs

#endif // BUBBLEFS_DB_REDIS_STORE_H_