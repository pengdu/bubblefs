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

// caffe2/caffe2/distributed/redis_store_handler.h

#ifndef BUBBLEFS_UTILS_CAFFE2_REDIS_STORE_HANDLER_H_
#define BUBBLEFS_UTILS_CAFFE2_REDIS_STORE_HANDLER_H_

#include "utils/caffe2_store_handler.h"

extern "C" {
#include "hiredis/hiredis.h"
}

#include <string>
#include "platform/macros.h"

namespace bubblefs {
namespace mycaffe2 {

class RedisStoreHandler : public StoreHandler {
 public:
  explicit RedisStoreHandler(std::string& host, int port, std::string& prefix);
  virtual ~RedisStoreHandler();

  virtual void set(const std::string& name, const std::string& data) override;

  virtual std::string get(const std::string& name) override;

  virtual int64_t add(const std::string& name, int64_t value) override;

  virtual bool check(const std::vector<std::string>& names) override;

  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 private:
  std::string host_;
  int port_ ATTRIBUTE_UNUSED;
  std::string prefix_;

  redisContext* redis_;

  std::string compoundKey(const std::string& name);
};

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_REDIS_STORE_HANDLER_H_