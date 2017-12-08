/***************************************************************************
 *
 * Copyright (c) 2014 Baidu, Inc. All Rights Reserved.
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
// Author: Chen Chi <chenchi01@baidu.com>

// bigflow/flume/util/jni_environment.cpp

#include "utils/bigflow_flume_jni_environment.h"

#include <cstdlib>

#include "utils/toft_base_scoped_array.h"
#include "utils/toft_base_string_algorithm.h"

namespace {

static void get_jvm_options(
        std::vector<std::string>* arg_strings,
        std::vector<JavaVMOption>* options) {
    // Get the environment variables for initializing the JVM
    char* hadoop_classpath = getenv("CLASSPATH");
    if (hadoop_classpath == NULL) {
      PRINTF_ERROR("Environment variable CLASSPATH not set!\n");
      return;
    }
    arg_strings->push_back("-Djava.class.path=" + std::string(hadoop_classpath));

    //const char* libhdfs_opts = getenv("LIBHDFS_OPTS");
    //if (libhdfs_opts != NULL) {
    //    std::vector<std::string> libhdfs_args;
    //    const char* jvm_arg_delims = " ";
    //    mytoft::SplitString(libhdfs_opts, jvm_arg_delims, &libhdfs_args);
    //    arg_strings->insert(arg_strings->end(), libhdfs_args.begin(), libhdfs_args.end());
    //}
    for (size_t i = 0; i < arg_strings->size(); ++i) {
        JavaVMOption option;
        option.optionString = const_cast<char*>((*arg_strings)[i].c_str());
        options->push_back(option);
    }
    PRINTF_INFO("JVM options:\n");
    for (size_t i = 0; i < options->size(); ++i) {
        PRINTF_INFO("%s\n", options->at(i).optionString);
    }
}

static JNIEnv* get_hdfs_jni_env() {
    char* hadoop_classpath = getenv("CLASSPATH");

    PRINTF_INFO("HADOOP classpath: %s\n", hadoop_classpath);

    const jsize vm_buf_length = 1;
    JavaVM* vm_buf[vm_buf_length];
    JNIEnv* env = NULL;
    jint num_vms = 0;
    jint rv = JNI_GetCreatedJavaVMs(&vm_buf[0], vm_buf_length, &num_vms);
    if (rv != 0) {
        PRINTF_ERROR("JNI_GetCreatedJavaVMs failed with error: %d\n", (int)rv);
        return NULL;
    }

    if (num_vms != 0) {
        // Attach this thread to the VM
        rv = vm_buf[0]->AttachCurrentThread(reinterpret_cast<void**>(&env), 0);
        if (rv != 0) {
            PRINTF_ERROR("Call to AttachCurrentThread failed with error: %d\n", (int)rv);
            return NULL;
        }
        PRINTF_INFO("Attached to existing JVM\n");
        return env;
    }

    std::vector<std::string> arg_strings;
    std::vector<JavaVMOption> options;
    get_jvm_options(&arg_strings, &options);

    // Create the VM
    JavaVMInitArgs vm_args;
    JavaVM* vm = NULL;
    vm_args.version = JNI_VERSION_1_2;
    vm_args.options = &options[0];
    vm_args.nOptions = options.size();
    vm_args.ignoreUnrecognized = 0;

    rv = JNI_CreateJavaVM(&vm, reinterpret_cast<void**>(&env), &vm_args);
    if (rv != 0) {
        PRINTF_ERROR("Call to JNI_CreateJavaVM failed with error: %d\n", (int)rv);
        return NULL;
    }
    return env;
}
}  // namespace

namespace bubblefs {
namespace mybdflume {
namespace util {

JniEnvironment::JniEnvironment() : _env(::get_hdfs_jni_env()) {}

JNIEnv* JniEnvironment::get_env() const {
    return _env;
}

void JniEnvironment::call_method(
        const std::string& clazz_name, const std::string& method_name, const std::string& signature,
        const std::string& in, std::string* out) {
    JniEnvironment environment;
    JNIEnv* env = environment.get_env();

    // get the class and prepare method argument, convert in(std::string) to byte[]
    jclass clazz = env->FindClass(clazz_name.c_str());
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        PANIC("find class %s exception", clazz_name.c_str());
    }
    if (static_cast<jclass>(NULL) == clazz) {
      PANIC("cannot find class: %s", clazz_name.c_str());
    }
    jbyteArray in_byte_array = env->NewByteArray(in.size());
    env->SetByteArrayRegion(in_byte_array, 0, in.size(),
            reinterpret_cast<const jbyte*>(in.c_str()));

    // find and call method
    jmethodID method = env->GetStaticMethodID(clazz, method_name.c_str(), signature.c_str());
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        PANIC("cannot find method %s.%s, %s", clazz_name.c_str(), method_name.c_str(), signature.c_str());
    }
    if (static_cast<jmethodID>(NULL) == method) {
      PANIC("cannot find static method: %s", method_name.c_str());
    }
    jobject output_byte_array_obj =
        env->CallStaticObjectMethod(clazz, method, in_byte_array);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        PANIC("call method %s.%s exception", clazz_name.c_str(), method_name.c_str());
    }
    jbyteArray out_byte_array =
        reinterpret_cast<jbyteArray>(output_byte_array_obj);
    if (static_cast<jbyteArray>(NULL) == out_byte_array) {
      PANIC("call method returned null, which is not expected");
    }

    // converts java output(byte[]) to c++ output(std::string)
    jint out_len = env->GetArrayLength(out_byte_array);
    if (0 > out_len) {
      PANIC("java output byte[] length should be positive");
    }
    mytoft::scoped_array<char> buffer(new char[out_len]);
    env->GetByteArrayRegion(out_byte_array, 0, out_len, reinterpret_cast<jbyte*>(buffer.get()));
    out->assign(buffer.get(), out_len);

    env->DeleteLocalRef(in_byte_array);
    env->DeleteLocalRef(output_byte_array_obj);
}

}  // namespace util
}  // namespace mybdflume
}  // namespace bubblefs