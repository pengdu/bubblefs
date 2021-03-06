/**
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

// ambry/ambry-api/src/main/java/com.github.ambry/router/Callback.java
// ambry/ambry-api/src/main/java/com.github.ambry/router/FutureResult.java

#include "utils/simple_countdownlatch.h"

namespace bubblefs {
namespace mysimple {  
  
/**
 * A callback interface that the user can implement to allow code to execute when the request is complete. This callback
 * will generally execute in the background I/O thread so it should be fast.
 */
template <typename T>
class Callback {
  /**
   * The callback method that the user can implement to do asynchronous handling of the response. T defines the type of
   * the result. When T=Void then there is no specific response expected but any errors are reported on the
   * exception object. For all other values of T, one of the two arguments would be null.
   * @param result The result of the request. This would be non null when the request executed successfully
   * @param exception The exception that was reported on execution of the request
   */
public:
  virtual void OnCompletion(T result, int exception);
}; 
  
/**
 * A class that implements the future completion of a request.
 */
template <typename T>
class FutureResult {
private:
  CountDownLatch latch_;
  volatile int error_; // 0 means no-error
  volatile T result_;
  
public:
  FutureResult() : latch_(1), error_(0) { }
  
  /**
   * Mark this request as complete and unblock any threads waiting on its completion.
   * @param result The result for this request
   * @param error The error that occurred if there was one, or null.
   */
  void Done(T result, int error) {
    error_ = error;
    result_ = result;
    latch_.CountDownLatch(); 
  }
  
  /**
   * Await the completion of this request
   */
  bool Await(uint64 timeout) {
    latch_.TimedWait(uint64);
  }
  
  T Result() { return result_; }
  
  int Error() { return error_; }
  
  /**
   * Has the request completed?
   */
  bool IsDone() {
    return latch_.GetCount() == 0L;
  }
    
  bool Cancel(bool mayInterruptIfRunning) {
    return false;
  }
  
  bool IsCancelled() {
    return false;
  }
  
  T Get() {
    Await();
    return ResultOrThrow();
  }

  T Get(uint64 timeout) {
    bool occurred = Await(timeout);
    if (!occurred) {
      throw std::runtime_error("FutureResult Get(timeout) exception");
    }
    return ResultOrThrow();
  }

  T ResultOrThrow() {
    if (Error() != 0) {
      throw std::runtime_error("FutureResult ResultOrThrow exception");
    } else {
      return Result();
    }
  }
};
  
} // namespace mysimple 
} // namespace bubblefs