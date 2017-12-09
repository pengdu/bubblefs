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

// bigflow/flume/util/config_util.cpp

#include "utils/bigflow_flume_config_util.h"

#include <algorithm>
#include <map>
#include "platform/macros.h"
#include "utils/toft_base_string_string_piece.h"
#include "utils/tinyxml2.h"

#include "gflags/gflags.h"

DEFINE_string(flume_hadoop_client, "", "path to hadoop command which are used to launch dce job.");
DEFINE_string(flume_hadoop_config, "", "path to hadoop config which are used to launch dce job.");

namespace bubblefs {
namespace mybdflume {
namespace util {

namespace {

ALLOW_UNUSED bool ParseProperty(tinyxml2::XMLElement* node, std::string* name, std::string* value) {
    const char* kName = "name";
    const char* kValue = "value";
    tinyxml2::XMLElement* name_element = node->FirstChildElement(kName);
    tinyxml2::XMLElement* value_element = node->FirstChildElement(kValue);
    if (NULL == name_element || NULL == value_element) {
        return false;
    }
    const char* name_text = name_element->GetText();
    const char* value_text = value_element->GetText();
    if (NULL == name_text) {
        return false;
    }
    *name = name_text;
    // value_text may be NULL when the value is an empty xml tag <value/>.
    if (value_text != NULL) {
        *value = value_text;
    } else {
        value->clear();
    }
    return true;
}

ALLOW_UNUSED bool ParseProperties(tinyxml2::XMLDocument* doc, std::map<std::string, std::string>* properties) {
    tinyxml2::XMLElement* root = doc->RootElement();
    const char* kProperty = "property";
    std::string name;
    std::string value;
    for (tinyxml2::XMLElement* child = root->FirstChildElement(kProperty);
        child != NULL;
        child = child->NextSiblingElement(kProperty)) {
        if(!ParseProperty(child, &name, &value)) {
            return false;
        }
        (*properties)[name] = value;
    }
    return true;
}

} // namespace
  
std::string DefaultHadoopConfigPath() {
    if (!FLAGS_flume_hadoop_config.empty()) {
        return FLAGS_flume_hadoop_config;
    }

    std::string path = google::StringFromEnv("HADOOP_CONF_PATH", "");
    if (!path.empty()) {
        return path;
    }

    return google::StringFromEnv("HADOOP_HOME", "") + std::string("/etc/hadoop/core-site.xml");
}

std::string DefaultHadoopClientPath() {
    if (!FLAGS_flume_hadoop_client.empty()) {
        return FLAGS_flume_hadoop_client;
    }

    return google::StringFromEnv("HADOOP_HOME", "") + std::string("/bin/hadoop");
}

std::string DefaultSparkHomePath() {
    // TODO(wangcong09) Parse from GFlags maybe
    return google::StringFromEnv("SPARK_HOME", "");
}

std::string DefaultTempHDFSPath(const std::string& unique_id) {
    return "hdfs:///flume/app/tmp/" + unique_id;
}

std::string DefaultTempLocalPath(const std::string& unique_id) {
    return "./.tmp/" + unique_id;
}

//   hdfs://host:port/xx => /xx
//   hdfs:///xx => /xx
//   /xx        => /xx
std::string RemoveHDFSPrefix(const std::string& path) {
    const std::string hdfs_prefix = "hdfs://";
    const bool is_hdfs = mytoft::StringPiece(path).starts_with(hdfs_prefix);
    if (is_hdfs) {
        std::string ret = path.substr(hdfs_prefix.size());
        std::string::size_type pos = ret.find('/');
        assert(std::string::npos != pos);
        return ret.substr(pos);
    } else {
        return path;
    }
}

}  // namespace util
}  // namespace mybdflume
}  // namespace bubblefs