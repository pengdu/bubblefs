/**
 * Copyright (c) 2016 by Contributors
 */

// ps-lite/include/ps/internal/env.h
// ps-lite/include/ps/internal/utils.h

#ifndef BUBBLEFS_UTILS_PSLITE_ENV_H_
#define BUBBLEFS_UTILS_PSLITE_ENV_H_

#include <cstdlib>
#include <unordered_map>
#include <memory>
#include <string>
#include "platform/types.h"

namespace bubblefs {
namespace mypslite {

/**
 * \brief Environment configurations
 */
class Environment {
 public:
  /**
   * \brief return the singleton instance
   */
  static inline Environment* Get() {
    return _GetSharedRef(nullptr).get();
  }
  /**
   * \brief return a shared ptr of the singleton instance
   */
  static inline std::shared_ptr<Environment> _GetSharedRef() {
    return _GetSharedRef(nullptr);
  }
  /**
   * \brief initialize the environment
   * \param envs key-value environment variables
   * \return the initialized singleton instance
   */
  static inline Environment* Init(const std::unordered_map<std::string, std::string>& envs) {
    return _GetSharedRef(&envs).get();
  }
  /**
   * \brief find the env value.
   *  User-defined env vars first. If not found, check system's environment
   * \param k the environment key
   * \return the related environment value, nullptr when not found
   */
  const char* find(const char* k) {
    std::string key(k);
    return kvs.find(key) == kvs.end() ? getenv(k) : kvs[key].c_str();
  }

 private:
  explicit Environment(const std::unordered_map<std::string, std::string>* envs) {
    if (envs) kvs = *envs;
  }

  static std::shared_ptr<Environment> _GetSharedRef(
      const std::unordered_map<std::string, std::string>* envs) {
    static std::shared_ptr<Environment> inst_ptr(new Environment(envs));
    return inst_ptr;
  }

  std::unordered_map<std::string, std::string> kvs;
};

/*!
 * \brief Get environment variable as int with default.
 * \param key the name of environment variable.
 * \param default_val the default value of environment vriable.
 * \return The value received
 */
template<typename V>
inline V GetEnv(const char *key, V default_val) {
  const char *val = Environment::Get()->find(key);
  if (val == nullptr) {
    return default_val;
  } else {
    return atoi(val);
  }
}

}  // namespace myps
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PSLITE_ENV_H_