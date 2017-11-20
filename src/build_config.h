#ifndef BUBBLEFS_BUILD_CONFIG_H_
#define BUBBLEFS_BUILD_CONFIG_H_

/////////////////////////////////////////////////////
// custom macros

#ifndef TF_COMPILE_LIBRARY
#define TF_COMPILE_LIBRARY 0
#endif

#ifndef TF_USE_NO_RTTI
#define TF_USE_NO_RTTI 0
#endif

// x86 and x86-64 can perform unaligned loads/stores directly.
#ifndef TF_USE_UNALIGNED
#define TF_USE_UNALIGNED 1
#endif 

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

#ifndef TF_USE_PTHREAD_SPINLOCK
#define TF_USE_PTHREAD_SPINLOCK 1
#endif

#ifndef TF_SUPPORT_THREAD_LOCAL
#define TF_SUPPORT_THREAD_LOCAL 1
#endif

#ifndef TF_USE_SSL
#define TF_USE_SSL 1
#endif

#endif // BUBBLEFS_BUILD_CONFIG_H_