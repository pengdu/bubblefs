// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "utils/simple_singleton.h"
#include "utils/simple_thread.h"
#include <stdio.h>
#include <string>

namespace bubblefs {
namespace mysimple {

class SingletonTest {
 public:
  SingletonTest() {}
  const std::string& name() const { return name_; }
  void setName(const std::string& n) { name_ = n; }

 private:
  std::string name_;
};

void ThreadFunc(void*) {
  Singleton<SingletonTest>::Instance()->setName("change singleton name");
}

void Test() {
  Singleton<SingletonTest>::Instance()->setName("singleton test");
  Thread t(&ThreadFunc, NULL);
  t.Start();
  t.Join();
  printf("name=%s\n", Singleton<SingletonTest>::Instance()->name().c_str());
}

}  // namespace simple
}  // namespace bubblefs

int main() {
  bubblefs::mysimple::Test();
  return 0;
}