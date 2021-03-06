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

#ifndef BUBBLEFS_UTILS_CAFFE2_STORE_HANDLER_H_
#define BUBBLEFS_UTILS_CAFFE2_STORE_HANDLER_H_

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bubblefs {
namespace mycaffe2 {

class StoreHandler {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(30);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  virtual ~StoreHandler();

  /*
   * Set data for the key if it doesn't exist.
   * If the key exists the data should be the same as the existing key.
   */
  virtual void set(const std::string& name, const std::string& data) = 0;

  /*
   * Get the data for the key.
   * The call should wait until the key is stored with default timeout
   * and return data if set else fail.
   */
  virtual std::string get(const std::string& name) = 0;

  /*
   * Does an atomic add operation on the key and returns the latest updated
   * value.
   * Note: To access the current value for this counter call with value = 0
   */
  virtual int64_t add(const std::string& name, int64_t value) = 0;

  /*
   * Check if a keys exist in the store.
   */
  virtual bool check(const std::vector<std::string>& names) = 0;

  /*
   * Wait for Keys to be stored.
   */
  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) = 0;
};

struct StoreHandlerTimeoutException : public std::runtime_error {
  StoreHandlerTimeoutException() = default;
  explicit StoreHandlerTimeoutException(const std::string& msg)
      : std::runtime_error(msg) {}
};

#define STORE_HANDLER_TIMEOUT(...)              \
  throw ::bubblefs::mycaffe2::StoreHandlerTimeoutException( \
      ::bubblefs::mycaffe2::MakeString("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__));

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_STORE_HANDLER_H_