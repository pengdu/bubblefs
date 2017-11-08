// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/platform_thread_posix.cc
// brpc/src/butil/threading/platform_thread_linux.cc

#include "platform/brpc_platform_thread.h"
#include <errno.h>
#include <sched.h>
#include "platform/macros.h"
#include "platform/time.h"
#include "platform/brpc_waitable_event.h"
#include "utils/brpc_thread_id_name_manager.h"

#if defined(OS_LINUX)
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace bubblefs {
namespace brpc {
  
void InitThreading();
void InitOnThread();
void TerminateOnThread();
size_t GetDefaultThreadStackSize(const pthread_attr_t& attributes);  
  
namespace {  
  
struct ThreadParams {
  ThreadParams()
      : delegate(NULL),
        joinable(false),
        priority(kThreadPriority_Normal),
        handle(NULL),
        handle_set(false, false) {
  }

  PlatformThread::Delegate* delegate;
  bool joinable;
  ThreadPriority priority;
  PlatformThreadHandle* handle;
  WaitableEvent handle_set;
};

void* ThreadFunc(void* params) {
  InitOnThread();
  ThreadParams* thread_params = static_cast<ThreadParams*>(params);

  PlatformThread::Delegate* delegate = thread_params->delegate;
  //if (!thread_params->joinable) butil::ThreadRestrictions::SetSingletonAllowed(false);

  if (thread_params->priority != kThreadPriority_Normal) {
    PlatformThread::SetThreadPriority(PlatformThread::CurrentHandle(),
                                      thread_params->priority);
  }

  // Stash the id in the handle so the calling thread has a complete
  // handle, and unblock the parent thread.
  *(thread_params->handle) = PlatformThreadHandle(pthread_self(),
                                                  PlatformThread::CurrentId());
  thread_params->handle_set.Signal();

  ThreadIdNameManager::GetInstance()->RegisterThread(
      PlatformThread::CurrentHandle().platform_handle(),
      PlatformThread::CurrentId());

  delegate->ThreadMain();

  ThreadIdNameManager::GetInstance()->RemoveName(
      PlatformThread::CurrentHandle().platform_handle(),
      PlatformThread::CurrentId());

  TerminateOnThread();
  return NULL;
}

bool CreateThread(size_t stack_size, bool joinable,
                  PlatformThread::Delegate* delegate,
                  PlatformThreadHandle* thread_handle,
                  ThreadPriority priority) {
  InitThreading();

  bool success = false;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);

  // Pthreads are joinable by default, so only specify the detached
  // attribute if the thread should be non-joinable.
  if (!joinable) {
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  }

  // Get a better default if available.
  if (stack_size == 0)
    stack_size = GetDefaultThreadStackSize(attributes);

  if (stack_size > 0)
    pthread_attr_setstacksize(&attributes, stack_size);

  ThreadParams params;
  params.delegate = delegate;
  params.joinable = joinable;
  params.priority = priority;
  params.handle = thread_handle;

  pthread_t handle;
  int err = pthread_create(&handle,
                           &attributes,
                           ThreadFunc,
                           &params);
  success = !err;
  if (!success) {
    // Value of |handle| is undefined if pthread_create fails.
    handle = 0;
    errno = err;
    PLOG(ERROR) << "pthread_create";
  }

  pthread_attr_destroy(&attributes);

  // Don't let this call complete until the thread id
  // is set in the handle.
  if (success)
    params.handle_set.Wait();
  CHECK_EQ(handle, thread_handle->platform_handle());

  return success;
}

int ThreadNiceValue(ThreadPriority priority) {
  switch (priority) {
    case kThreadPriority_RealtimeAudio:
      return -10;
    case kThreadPriority_Background:
      return 10;
    case kThreadPriority_Normal:
      return 0;
    case kThreadPriority_Display:
      return -6;
    default:
      CHECK(false) << "Unknown priority.";
      return 0;
  }
}

}  // namespace

// static
PlatformThreadId PlatformThread::CurrentId() {
  // Pthreads doesn't have the concept of a thread ID, so we have to reach down
  // into the kernel.
#if defined(OS_MACOSX)
  return pthread_mach_thread_np(pthread_self());
#elif defined(OS_LINUX)
  return syscall(__NR_gettid);
#elif defined(OS_ANDROID)
  return gettid();
#elif defined(OS_SOLARIS) || defined(OS_QNX)
  return pthread_self();
#elif defined(OS_NACL) && defined(__GLIBC__)
  return pthread_self();
#elif defined(OS_NACL) && !defined(__GLIBC__)
  // Pointers are 32-bits in NaCl.
  return reinterpret_cast<int32_t>(pthread_self());
#elif defined(OS_POSIX)
  return reinterpret_cast<int64_t>(pthread_self());
#endif
}

// static
PlatformThreadRef PlatformThread::CurrentRef() {
  return PlatformThreadRef(pthread_self());
}

// static
PlatformThreadHandle PlatformThread::CurrentHandle() {
  return PlatformThreadHandle(pthread_self(), CurrentId());
}

// static
void PlatformThread::YieldCurrentThread() {
  sched_yield();
}

// static
void PlatformThread::Sleep(int64_t duration) {
  struct timespec sleep_time, remaining;

  // Break the duration into seconds and nanoseconds.
  // NOTE: TimeDelta's microseconds are int64s while timespec's
  // nanoseconds are longs, so this unpacking must prevent overflow.
  sleep_time = timeutil::microseconds_to_timespec(duration);

  while (nanosleep(&sleep_time, &remaining) == -1 && errno == EINTR)
    sleep_time = remaining;
}

// static
const char* PlatformThread::GetName() {
  //return ThreadIdNameManager::GetInstance()->GetName(CurrentId());
  assert(false);
  return "UnknownThreadName";
}

// static
bool PlatformThread::Create(size_t stack_size, Delegate* delegate,
                            PlatformThreadHandle* thread_handle) {
  //butil::ThreadRestrictions::ScopedAllowWait allow_wait;
  return CreateThread(stack_size, true /* joinable thread */,
                      delegate, thread_handle, kThreadPriority_Normal);
}

// static
bool PlatformThread::CreateWithPriority(size_t stack_size, Delegate* delegate,
                                        PlatformThreadHandle* thread_handle,
                                        ThreadPriority priority) {
  //butil::ThreadRestrictions::ScopedAllowWait allow_wait;
  return CreateThread(stack_size, true,  // joinable thread
                      delegate, thread_handle, priority);
}

// static
bool PlatformThread::CreateNonJoinable(size_t stack_size, Delegate* delegate) {
  PlatformThreadHandle unused;

  //butil::ThreadRestrictions::ScopedAllowWait allow_wait;
  bool result = CreateThread(stack_size, false /* non-joinable thread */,
                             delegate, &unused, kThreadPriority_Normal);
  return result;
}

// static
void PlatformThread::Join(PlatformThreadHandle thread_handle) {
  // Joining another thread may block the current thread for a long time, since
  // the thread referred to by |thread_handle| may still be running long-lived /
  // blocking tasks.
  //butil::ThreadRestrictions::AssertIOAllowed();
  CHECK_EQ(0, pthread_join(thread_handle.handle_, NULL));
}

// NOTE(gejun): PR_SET_NAME was added in 2.6.9, should be working on most of
// our machines, but missing from our linux headers.
#if !defined(PR_SET_NAME)
#define PR_SET_NAME 15
#endif

// static
void PlatformThread::SetName(const char* name) {
  //ThreadIdNameManager::GetInstance()->SetName(CurrentId(), name);

#if !defined(OS_NACL)
  // On linux we can get the thread names to show up in the debugger by setting
  // the process name for the LWP.  We don't want to do this for the main
  // thread because that would rename the process, causing tools like killall
  // to stop working.
  if (PlatformThread::CurrentId() == getpid())
    return;

  // http://0pointer.de/blog/projects/name-your-threads.html
  // Set the name for the LWP (which gets truncated to 15 characters).
  // Note that glibc also has a 'pthread_setname_np' api, but it may not be
  // available everywhere and it's only benefit over using prctl directly is
  // that it can set the name of threads other than the current thread.
  int err = prctl(PR_SET_NAME, name);
  // We expect EPERM failures in sandboxed processes, just ignore those.
  if (err < 0 && errno != EPERM)
    DLOG(ERROR) << "prctl(PR_SET_NAME)";
#endif  //  !defined(OS_NACL)
}

// static
void PlatformThread::SetThreadPriority(PlatformThreadHandle handle,
                                       ThreadPriority priority) {
#if !defined(OS_NACL)
  if (priority == kThreadPriority_RealtimeAudio) {
    const struct sched_param kRealTimePrio = { 8 };
    if (pthread_setschedparam(pthread_self(), SCHED_RR, &kRealTimePrio) == 0) {
      // Got real time priority, no need to set nice level.
      return;
    }
  }

  // setpriority(2) will set a thread's priority if it is passed a tid as
  // the 'process identifier', not affecting the rest of the threads in the
  // process. Setting this priority will only succeed if the user has been
  // granted permission to adjust nice values on the system.
  DCHECK_NE(handle.id_, kInvalidThreadId);
  const int kNiceSetting = ThreadNiceValue(priority);
  if (setpriority(PRIO_PROCESS, handle.id_, kNiceSetting)) {
    DLOG(ERROR) << "Failed to set nice value of thread ("
              << handle.id_ << ") to " << kNiceSetting;
  }
#endif  //  !defined(OS_NACL)
}

void InitThreading() {}

void InitOnThread() {}

void TerminateOnThread() {}

size_t GetDefaultThreadStackSize(const pthread_attr_t& attributes) {
#if !defined(THREAD_SANITIZER) && !defined(MEMORY_SANITIZER)
  return 0;
#else
  // ThreadSanitizer bloats the stack heavily. Evidence has been that the
  // default stack size isn't enough for some browser tests.
  // MemorySanitizer needs this as a temporary fix for http://crbug.com/353687
  return 2 * (1 << 23);  // 2 times 8192K (the default stack size on Linux).
#endif
}

} // namespace brpc
} // namespace bubblefs