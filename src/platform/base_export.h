// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/base_export.h

#ifndef BUBBLEFS_PLATFORM_BASE_EXPORT_H_
#define BUBBLEFS_PLATFORM_BASE_EXPORT_H_

#include "platform/platform.h"

// Control visiblity outside .so
#if defined(COMPILER_MSVC)
#ifdef TF_COMPILE_LIBRARY
#define BASE_EXPORT __declspec(dllexport)
#define BASE_EXPORT_PRIVATE __declspec(dllexport)
#else
#define BASE_EXPORT __declspec(dllimport)
#define BASE_EXPORT_PRIVATE __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define BASE_EXPORT __attribute__((visibility("default")))
#define BASE_EXPORT_PRIVATE __attribute__((visibility("default")))
#endif  // COMPILER_MSVC

#endif  // BUBBLEFS_PLATFORM_BASE_EXPORT_H_