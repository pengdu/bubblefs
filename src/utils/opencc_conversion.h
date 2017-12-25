/*
 * Open Chinese Convert
 *
 * Copyright 2010-2014 BYVoid <byvoid@byvoid.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenCC/src/Conversion.hpp

#ifndef BUBBLEFS_UTILS_OPENCC_CONVERSION_H_
#define BUBBLEFS_UTILS_OPENCC_CONVERSION_H_

#include "utils/opencc_dict.h"
#include "utils/opencc_segments.h"

namespace bubblefs {
namespace myopencc {
/**
* Conversion interface
* @ingroup opencc_cpp_api
*/
class Conversion {
public:
  Conversion(DictPtr _dict) : dict(_dict) {}

  // Convert single phrase
  string Convert(const string& phrase) const;

  // Convert single phrase
  string Convert(const char* phrase) const;

  // Convert segmented text
  SegmentsPtr Convert(const SegmentsPtr& input) const;

  const DictPtr GetDict() const { return dict; }

private:
  const DictPtr dict;
};

typedef std::shared_ptr<Conversion> ConversionPtr;

} // namespace myopencc
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_OPENCC_CONVERSION_H_