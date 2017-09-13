
#ifndef BUBBLEFS_PLATFORM_BASE_H_
#define BUBBLEFS_PLATFORM_BASE_H_

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

#endif // #ifndef BUBBLEFS_PLATFORM_BASE_H_