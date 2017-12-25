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

// OpenCC/src/DartsDict.hpp

#ifndef BUBBLEFS_UTILS_OPENCC_DARTS_DICT_H_
#define BUBBLEFS_UTILS_OPENCC_DARTS_DICT_H_

#include "utils/opencc_dict.h"
#include "utils/opencc_serializable_dict.h"

namespace bubblefs {
namespace myopencc {
/**
* Darts dictionary
* @ingroup opencc_cpp_api
*/

class DartsDict;
typedef std::shared_ptr<DartsDict> DartsDictPtr;

class DartsDict : public Dict {
public:
  virtual ~DartsDict();

  virtual size_t KeyMaxLength() const;

  virtual Optional<const DictEntry*> Match(const char* word) const;

  virtual Optional<const DictEntry*> MatchPrefix(const char* word) const;

  virtual LexiconPtr GetLexicon() const;

  virtual void SerializeToFile(FILE* fp) const;

  /**
  * Constructs a DartsDict from another dictionary.
  */
  static DartsDictPtr NewFromDict(const Dict& thatDict);

  static DartsDictPtr NewFromFile(FILE* fp);

private:
  DartsDict();

  size_t maxLength;
  LexiconPtr lexicon;

  class DartsInternal;
  DartsInternal* internal;
};

} // namespace myopencc
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_OPENCC_DARTS_DICT_H_