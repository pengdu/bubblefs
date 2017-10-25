// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "utils/simple_threadlocal_storage.h"
#include <stdio.h>
#include <string>
#include "utils/simple_thread.h"

namespace bubblefs {
namespace simple {

class ThreadLocalStorageTest {
 public:
  std::string name() const { return name_; }
  void set_name(const std::string& n) { name_ = n; }

 private:
  std::string name_;
};

ThreadLocalStorage<ThreadLocalStorageTest> obj1;
ThreadLocalStorage<ThreadLocalStorageTest> obj2;

void Test() {
  printf("obj1 %p name=%s\n", obj1.Get(), obj1.Get()->name().c_str());
  printf("obj2 %p name=%s\n", obj2.Get(), obj2.Get()->name().c_str());
}

void ThreadFunc(void*) {
  Test();
  obj1.Get()->set_name("threadfunc one");
  obj2.Get()->set_name("threadfunc two");
  Test();
}

}  // namespace simple
}  // namespace bubblefs

int main() {
  bubblefs::simple::obj1.Get()->set_name("main one");
  bubblefs::simple::obj2.Get()->set_name("main two");
  bubblefs::simple::Test();
  bubblefs::simple::Thread t1(bubblefs::simple::ThreadFunc, NULL);
  t1.Start();
  t1.Join();
  bubblefs::simple::Test();
  return 0;
}