/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// See port_example.h for documentation for the following types/functions.

// rocksdb/port/port_posix.h
// tensorflow/tensorflow/core/platform/init_main.h
// tensorflow/tensorflow/core/platform/host_info.h

#ifndef BUBBLEFS_PLATFORM_PORT_H_
#define BUBBLEFS_PLATFORM_PORT_H_

#include <endian.h> // for linux
#include "platform/base.h"
#include "platform/macros.h"
#include "platform/types.h"

namespace bubblefs {
namespace port {
  
// Platform-specific initialization routine that may be invoked by a
// main() program that uses TensorFlow.
//
// Default implementation does nothing.
void InitMain(const char* usage, int* argc, char*** argv);

// Return the hostname of the machine on which this process is running
std::string Hostname();

}  // namespace port
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_PORT_H_