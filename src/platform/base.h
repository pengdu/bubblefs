
#ifndef BUBBLEFS_PLATFORM_BASE_H_
#define BUBBLEFS_PLATFORM_BASE_H_

/*! \brief whether use glog for logging */
#ifndef TF_USE_GLOG
#define TF_USE_GLOG 0
#endif

#ifndef TF_USE_JEMALLOC
#define TF_USE_JEMALLOC 0
#endif

#ifndef TF_USE_SNAPPY
#define TF_USE_SNAPPY 0
#endif

#ifndef TF_USE_ADAPTIVE_MUTEX
#define TF_USE_ADAPTIVE_MUTEX 0
#endif

#ifndef TF_USE_PTHREAD_SPINLOCK
#define TF_USE_PTHREAD_SPINLOCK 0
#endif

#ifndef TF_USE_PYTHON
#define TF_USE_PYTHON 2.7
#endif

#endif // #ifndef BUBBLEFS_PLATFORM_BASE_H_