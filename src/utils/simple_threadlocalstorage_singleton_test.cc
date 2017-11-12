// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "utils/simple_thread.h"
#include "utils/simple_threadlocal_singleton.h"

#include <stdio.h>
#include <string>

namespace bubblefs {
namespace mysimple {

class Test {
 public:
  Test() {}

  const std::string& name() const { return name_; }
  void set_name(const std::string& n) { name_ = n; }

 private:
  std::string name_;
};

void ThreadFunc(void* arg) {
  std::string* c = reinterpret_cast<std::string*>(arg);
  printf("name=%s\n", ThreadLocalSingleton<Test>::get()->name().c_str());
  ThreadLocalSingleton<Test>::get()->set_name(*c);
  printf("name=%s\n", ThreadLocalSingleton<Test>::get()->name().c_str());
}

}  // namespace mysimple
}  // namespace bubblefs

int main() {
  std::string s1("thread 1");
  std::string s2("thread 2");
  bubblefs::mysimple::Thread t1(bubblefs::mysimple::ThreadFunc, &s1);
  bubblefs::mysimple::Thread t2(bubblefs::mysimple::ThreadFunc, &s2);
  t1.Start();
  t2.Start();
  t1.Join();
  t2.Join();
  return 0;
}