// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef BUBBLEFS_PLATFORM_BUILD_CONFIG_H_
#define BUBBLEFS_PLATFORM_BUILD_CONFIG_H_

/////////////////////////////////////////////////////
// custom macros

#ifndef TF_NO_RTTI
#define TF_NO_RTTI 0
#endif

// x86 and x86-64 can perform unaligned loads/stores directly.
#ifndef TF_USE_UNALIGNED
#define TF_USE_UNALIGNED 1
#endif 

/*! \brief whether use glog for logging */
#ifndef TF_USE_GLOG
#define TF_USE_GLOG 1
#endif

#ifndef TF_USE_PYTHON
#define TF_USE_PYTHON 2.7
#endif

#ifndef TF_USE_JEMALLOC
#define TF_USE_JEMALLOC 1
#endif

#ifndef TF_USE_SNAPPY
#define TF_USE_SNAPPY 1
#endif

#ifndef TF_USE_PTHREAD_SPINLOCK
#define TF_USE_PTHREAD_SPINLOCK 1
#endif

#ifndef TF_SUPPORT_THREAD_LOCAL
#define TF_SUPPORT_THREAD_LOCAL 1
#endif

#endif // #ifndef BUBBLEFS_PLATFORM_BUILD_CONFIG_H_