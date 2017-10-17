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

// caffe2/caffe2/distributed/redis_store_handler.cc

#include "db/redis_store.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include "utils/stl_stream.h"

namespace bubblefs {
namespace db {
  
#define THROW_TIMEOUT_EXCEPTION(...)              \
  throw ::bubblefs::db::RedisStoreTimeoutException( \
      ::bubblefs::MakeString("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__));
  
Status RedisStore::Open(
  std::string& host,
  int port,
  std::string& prefix)  {
  host_ = host;
  port_ = port;
  prefix_ = prefix;
  struct timeval tv = {
      .tv_sec = 5, .tv_usec = 0,
  };

  redis_ = redisConnectWithTimeout(host.c_str(), port, tv);
  if (nullptr == redis_) {
    std::stringstream ss;
    ss << "redisConnectWithTimeout return NULL, host: " << host << ", port: " << port;
    return Status(error::USER_ERROR, ss.str());
  }
  if (0 != redis_->err) {
    std::stringstream ss;
    ss << "Failed to redisConnectWithTimeout, host: " << host << ", port: " << port;
    ss << ", redis_err: " << redis_->err << ", " << redis_->errstr;
    return Status(error::USER_ERROR, ss.str());
  }
  return Status::OK();
}

RedisStore::~RedisStore() {
  redisFree(redis_);
}

std::string RedisStore::CompoundKey(const std::string& name) {
  return prefix_ + name;
}

Status RedisStore::Set(const std::string& name, const std::string& data) {
  std::string key = CompoundKey(name);
  void* ptr = redisCommand(
      redis_,
      "SETNX %b %b",
      key.c_str(),
      (size_t)key.size(),
      data.c_str(),
      (size_t)data.size());
  if (nullptr == ptr) {
    std::stringstream ss;
    ss << "Redis failed to SETNX, key: " << key;
    ss << ", " << redis_->errstr;
    return Status(error::USER_ERROR, ss.str());
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (REDIS_REPLY_INTEGER != reply->type) {
    std::stringstream ss;
    ss << "Redis failed to SETNX, key: " << key;
    ss << ", redisReply's type is not REDIS_REPLY_INTEGER";
    return Status(error::USER_ERROR, ss.str());
  }
  if (1 <= reply->integer) {
    std::stringstream ss;
    ss << "Redis failed to SETNX, key: " << key;
    ss << " was already set, (perhaps you reused a run ID you have used before?)";
    return Status(error::USER_ERROR, ss.str());
  }
  return Status::OK();
}

Status RedisStore::Get(const std::string& name, std::string* data) {
  if (!Exist({name})) {
    std::stringstream ss;
    ss << "Redis failed to Get, name: " << name;
    ss << " is not existed";
    return Status(error::USER_ERROR, ss.str());
  }

  std::string key = CompoundKey(name);
  void* ptr = redisCommand(redis_, "GET %b", key.c_str(), (size_t)key.size());
  if (nullptr == ptr) {
    std::stringstream ss;
    ss << "Redis failed to GET, key: " << key;
    ss << ", " << redis_->errstr;
    return Status(error::USER_ERROR, ss.str());
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (REDIS_REPLY_STRING != reply->type) {
    std::stringstream ss;
    ss << "Redis failed to GET, key: " << key;
    ss << ", redisReply's type is not REDIS_REPLY_STRING";
    return Status(error::USER_ERROR, ss.str());
  }
  data->assign(reply->str, reply->len);
  return Status::OK();
}

Status RedisStore::TimedGet(const std::string& name, 
                            std::string* data,
                            const std::chrono::milliseconds& timeout) {
  // Block until key is set
  Status s = WaitExist({name}, timeout);
  if (!s.ok()) {
    std::stringstream ss;
    ss << "Redis WaitExist timeout in TimedGet, timeout: " << timeout.count();
    ss << ", name: " << name;
    return Status(error::USER_ERROR, ss.str());
  }

  std::string key = CompoundKey(name);
  void* ptr = redisCommand(redis_, "GET %b", key.c_str(), (size_t)key.size());
  if (nullptr == ptr) {
    std::stringstream ss;
    ss << "Redis failed to GET in TimedGet, key: " << key;
    ss << ", " << redis_->errstr;
    return Status(error::USER_ERROR, ss.str());
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (REDIS_REPLY_STRING != reply->type) {
    std::stringstream ss;
    ss << "Redis failed to GET in TimedGet, key: " << key;
    ss << ", redisReply's type is not REDIS_REPLY_STRING";
    return Status(error::USER_ERROR, ss.str());
  }
  data->assign(reply->str, reply->len);
  return Status::OK();
}

Status RedisStore::Add(const std::string& name,
                       int64_t value,
                       int64_t* latest_value) {
  std::string key = CompoundKey(name);
  void* ptr = redisCommand(
      redis_, "INCRBY %b %ld", key.c_str(), (size_t)key.size(), value);
  if (nullptr == ptr) {
    std::stringstream ss;
    ss << "Redis failed to INCRBY, key: " << key;
    ss << ", " << redis_->errstr;
    return Status(error::USER_ERROR, ss.str());
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (REDIS_REPLY_INTEGER != reply->type) {
    std::stringstream ss;
    ss << "Redis failed to INCRBY, key: " << key;
    ss << ", redisReply's type is not REDIS_REPLY_INTEGER";
    return Status(error::USER_ERROR, ss.str());
  }
  *latest_value = reply->integer;
  return Status::OK();
}

bool RedisStore::Exist(const std::vector<std::string>& names) {
  std::vector<std::string> args;
  args.push_back("EXISTS");
  for (const auto& name : names) {
    args.push_back(CompoundKey(name));
  }

  std::vector<const char*> argv;
  std::vector<size_t> argvlen;
  for (const auto& arg : args) {
    argv.push_back(arg.c_str());
    argvlen.push_back(arg.length());
  }

  auto argc = argv.size();
  void* ptr = redisCommandArgv(redis_, argc, argv.data(), argvlen.data());
  if (nullptr == ptr)
    return false;
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (REDIS_REPLY_INTEGER != reply->type)
    return false;
  return static_cast<size_t>(reply->integer) == names.size();
}

Status RedisStore::WaitExist(
    const std::vector<std::string>& names,
    const std::chrono::milliseconds& timeout) {
  // Simple approach: poll...
  // Complex approach: use pub/sub.
  // Polling is fine for the typical rendezvous use case, as it is
  // only done at initialization time and  not at run time.
  const auto start = std::chrono::steady_clock::now();
  while (!Exist(names)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      std::stringstream ss;
      ss << "Redis WaitExist timeout, timeout: " << timeout.count();
      ss << ", names: " << JoinToString(", ", names);
      return Status(error::USER_ERROR, ss.str());
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return Status::OK();
}

} // namespace db
} // namespace bubblefs