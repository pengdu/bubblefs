// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com
// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/platform_thread.h
// baidu/common/include/thread.h

// WARNING: You should *NOT* be using this class directly.  PlatformThread is
// the low-level platform-specific abstraction to the OS's threading interface.
// You should instead be using a message-loop driven Thread, see thread.h.

#ifndef  BUBBLEFS_PLATFORM_PLATFORM_THREAD_H_
#define  BUBBLEFS_PLATFORM_PLATFORM_THREAD_H_

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>
#include <functional>
#include "platform/macros.h"

namespace bubblefs {

namespace concurrent {
// Valid values for SetThreadPriority()
enum ThreadPriority{
  kThreadPriority_Normal,
  // Suitable for low-latency, glitch-resistant audio.
  kThreadPriority_RealtimeAudio,
  // Suitable for threads which generate data for the display (at ~60Hz).
  kThreadPriority_Display,
  // Suitable for threads that shouldn't disrupt high priority work.
  kThreadPriority_Background
};

// Used for logging. Always an integer value.
#if defined(OS_WIN)
typedef DWORD PlatformThreadId;
#elif defined(OS_POSIX)
typedef pid_t PlatformThreadId;
#endif

// Used for thread checking and debugging.
// Meant to be as fast as possible.
// These are produced by PlatformThread::CurrentRef(), and used to later
// check if we are on the same thread or not by using ==. These are safe
// to copy between threads, but can't be copied to another process as they
// have no meaning there. Also, the internal identifier can be re-used
// after a thread dies, so a PlatformThreadRef cannot be reliably used
// to distinguish a new thread from an old, dead thread.
class PlatformThreadRef {
 public:
#if defined(OS_WIN)
  typedef DWORD RefType;
#elif defined(OS_POSIX)
  typedef pthread_t RefType;
#endif
  PlatformThreadRef()
      : id_(0) {
  }

  explicit PlatformThreadRef(RefType id)
      : id_(id) {
  }

  bool operator==(PlatformThreadRef other) const {
    return id_ == other.id_;
  }

  bool is_null() const {
    return id_ == 0;
  }
 private:
  RefType id_;
};

// Used to operate on threads.
class PlatformThreadHandle {
 public:
#if defined(OS_WIN)
  typedef void* Handle;
#elif defined(OS_POSIX)
  typedef pthread_t Handle;
#endif

  PlatformThreadHandle()
      : handle_(0),
        id_(0) {
  }

  explicit PlatformThreadHandle(Handle handle)
      : handle_(handle),
        id_(0) {
  }

  PlatformThreadHandle(Handle handle,
                       PlatformThreadId id)
      : handle_(handle),
        id_(id) {
  }

  bool is_equal(const PlatformThreadHandle& other) const {
    return handle_ == other.handle_;
  }

  bool is_null() const {
    return !handle_;
  }

  Handle platform_handle() const {
    return handle_;
  }

 private:
  friend class PlatformThread;

  Handle handle_;
  PlatformThreadId id_;
};

const PlatformThreadId kInvalidThreadId(0);

// A namespace for low-level thread functions.
class BASE_EXPORT PlatformThread {
 public:
  // Implement this interface to run code on a background thread.  Your
  // ThreadMain method will be called on the newly created thread.
  class BASE_EXPORT Delegate {
   public:
    virtual void ThreadMain() = 0;

   protected:
    virtual ~Delegate() {}
  };

  // Gets the current thread id, which may be useful for logging purposes.
  static PlatformThreadId CurrentId();

  // Gets the current thread reference, which can be used to check if
  // we're on the right thread quickly.
  static PlatformThreadRef CurrentRef();

  // Get the current handle.
  static PlatformThreadHandle CurrentHandle();

  // Yield the current thread so another thread can be scheduled.
  static void YieldCurrentThread();

  // Sleeps for the specified duration us.
  static void Sleep(int64_t duration);

  // Sets the thread name visible to debuggers/tools. This has no effect
  // otherwise. This name pointer is not copied internally. Thus, it must stay
  // valid until the thread ends.
  static void SetName(const char* name);

  // Gets the thread name, if previously set by SetName.
  static const char* GetName();

  // Creates a new thread.  The |stack_size| parameter can be 0 to indicate
  // that the default stack size should be used.  Upon success,
  // |*thread_handle| will be assigned a handle to the newly created thread,
  // and |delegate|'s ThreadMain method will be executed on the newly created
  // thread.
  // NOTE: When you are done with the thread handle, you must call Join to
  // release system resources associated with the thread.  You must ensure that
  // the Delegate object outlives the thread.
  static bool Create(size_t stack_size, Delegate* delegate,
                     PlatformThreadHandle* thread_handle);

  // CreateWithPriority() does the same thing as Create() except the priority of
  // the thread is set based on |priority|.  Can be used in place of Create()
  // followed by SetThreadPriority().  SetThreadPriority() has not been
  // implemented on the Linux platform yet, this is the only way to get a high
  // priority thread on Linux.
  static bool CreateWithPriority(size_t stack_size, Delegate* delegate,
                                 PlatformThreadHandle* thread_handle,
                                 ThreadPriority priority);

  // CreateNonJoinable() does the same thing as Create() except the thread
  // cannot be Join()'d.  Therefore, it also does not output a
  // PlatformThreadHandle.
  static bool CreateNonJoinable(size_t stack_size, Delegate* delegate);

  // Joins with a thread created via the Create function.  This function blocks
  // the caller until the designated thread exits.  This will invalidate
  // |thread_handle|.
  static void Join(PlatformThreadHandle thread_handle);

  static void SetThreadPriority(PlatformThreadHandle handle,
                                ThreadPriority priority);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(PlatformThread);
};

} // namespace concurrent
  
namespace bdcommon {

class ThisThread {
public:
    /// Sleep in ms
    static void Sleep(int64_t time_ms) {
        if (time_ms > 0) {
            struct timespec ts = {time_ms / 1000, (time_ms % 1000) * 1000000 };
            nanosleep(&ts, &ts);
        }
    }
    /// Get thread id
    static int GetId() {
        static __thread int s_thread_id = 0;
        if (s_thread_id == 0) {
            s_thread_id = syscall(__NR_gettid);
        }
        return s_thread_id;
    }
    /// Yield cpu
    static void Yield() {
        sched_yield();
    }
};  
  
class ThreadAttributes {
public:
    ThreadAttributes() {
        cpu_num_ = sysconf(_SC_NPROCESSORS_CONF);
        mask_ = GetCpuAffinity();
    }
    ~ThreadAttributes() {}

    int32_t GetCpuNum() {
        return cpu_num_;
    }

    cpu_set_t GetCpuAffinity() {
        ResetCpuMask();
        if (sched_getaffinity(0, sizeof(mask_), &mask_) == -1) {
            ResetCpuMask();
        }
        return mask_;
    }
    bool SetCpuAffinity() {
        if (sched_setaffinity(0, sizeof(mask_), &mask_) == -1) {
            return false;
        }
        return true;
    }

    bool SetCpuMask(int32_t cpu_id) {
        if (cpu_id < 0 || cpu_id > cpu_num_) {
            return false;
        }

        if (CPU_ISSET(cpu_id, &mask_)) {
            return true;
        }
        CPU_SET(cpu_id, &mask_);
        return true;
    }
    void ResetCpuMask() {
        CPU_ZERO(&mask_);
    }
    void MarkCurMask() {
        CPU_ZERO(&last_mask_);
        last_mask_ = mask_;
    }
    bool RevertCpuAffinity() {
        ResetCpuMask();
        mask_ = last_mask_;
        return SetCpuAffinity();
    }

private:
    int32_t cpu_num_;
    cpu_set_t mask_;
    cpu_set_t last_mask_;
};
  
// simple thread implement
class Thread {
public:
    Thread() {
        memset(&tid_, 0, sizeof(tid_));
    }
    int64_t CurrentId() {
        return static_cast<int64_t>(pthread_self());
    }
    bool Start(std::function<void ()> thread_proc) {
        user_proc_ = thread_proc;
        int ret = pthread_create(&tid_, nullptr, ProcWrapper, this);
        return (ret == 0);
    }
    typedef void* (Proc)(void*);
    bool Start(Proc proc, void* arg) {
        int ret = pthread_create(&tid_, nullptr, proc, arg);
        return (ret == 0);
    }
    bool Start(Proc proc, void* arg, size_t stack_size, bool joinable) {
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        // Pthreads are joinable by default, so only specify the detached
        // attribute if the thread should be non-joinable.
        if (!joinable) {
          pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        }
        // Get a better default if available.
        if (stack_size > 0)
          pthread_attr_setstacksize(&attributes, stack_size);
        else
          stack_size = 2 * (1 << 23);  // 2 times 8192K (the default stack size on Linux).
        
        int ret = pthread_create(&tid_, nullptr, proc, arg);
        
        pthread_attr_destroy(&attributes);
        return (ret == 0);
    }
    bool Join() {
        int ret = pthread_join(tid_, nullptr);
        return (ret == 0);
    }
    void Exit() {
        pthread_exit(nullptr);
    }
    void YieldCurrentThread() {
        sched_yield();
    }
    void Sleep(struct timespec &duration) {
        struct timespec sleep_time, remaining;

        // Break the duration into seconds and nanoseconds.
        // NOTE: TimeDelta's microseconds are int64s while timespec's
        // nanoseconds are longs, so this unpacking must prevent overflow.
        sleep_time.tv_sec = duration.tv_sec;
        sleep_time.tv_nsec = duration.tv_nsec;  // nanoseconds

        while (nanosleep(&sleep_time, &remaining) == -1 && errno == EINTR)
          sleep_time = remaining;
    }
private:
    static void* ProcWrapper(void* arg) {
        reinterpret_cast<Thread*>(arg)->user_proc_();
        return nullptr;
    }
    DISALLOW_COPY_AND_ASSIGN(Thread);
private:
    std::function<void ()> user_proc_;
    pthread_t tid_;
};

} // namespace bdcommon
} // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_PLATFORM_THREAD_H_