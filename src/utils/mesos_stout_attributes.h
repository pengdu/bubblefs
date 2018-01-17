// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mesos/3rdparty/stout/include/stout/attributes.hpp

#ifndef BUBBLEFS_UTILS_MESOS_STOUT_ATTRIBUTES_H_
#define BUBBLEFS_UTILS_MESOS_STOUT_ATTRIBUTES_H_


#ifdef __WINDOWS__
#define NORETURN __declspec(noreturn)
#else
#define NORETURN __attribute__((noreturn))
#endif // __WINDOWS__

// expand the macro first then combine
#define CAT(a, b) CAT_I(a, b) 
#define CAT_I(a, b) a##b


#endif // BUBBLEFS_UTILS_MESOS_STOUT_ATTRIBUTES_H_