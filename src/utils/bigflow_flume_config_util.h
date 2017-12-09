/***************************************************************************
 *
 * Copyright (c) 2015 Baidu, Inc. All Rights Reserved.
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
 *
 **************************************************************************/
// Author: Wang Cong <wangcong09@baidu.com>
//
// Utility for some default config parameters.

// bigflow/flume/util/config_util.h

#ifndef BUBBLEFS_UTILS_BIGFLOW_FLUME_CONFIG_UTIL_H_
#define BUBBLEFS_UTILS_BIGFLOW_FLUME_CONFIG_UTIL_H_

#include <string>

namespace bubblefs {
namespace mybdflume {
namespace util {

std::string DefaultHadoopConfigPath();

std::string DefaultHadoopClientPath();

std::string DefaultSparkHomePath();

std::string DefaultTempHDFSPath(const std::string& unique_id);

std::string DefaultTempLocalPath(const std::string& unique_id);

//   hdfs://host:port/xx => /xx
//   hdfs:///xx => /xx
//   /xx        => /xx
std::string RemoveHDFSPrefix(const std::string& path);

}  // namespace util
}  // namespace mybdflume
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BIGFLOW_FLUME_CONFIG_UTIL_H_