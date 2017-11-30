
// slash/slash/include/random.h

#ifndef BUBBLEFS_UTILS_SLASH_RANDOM_H_
#define BUBBLEFS_UTILS_SLASH_RANDOM_H_

#include <stdlib.h>
#include <time.h>

namespace bubblefs {
namespace myslash {

class Random {
 public:
  Random() {
    srand(time(NULL));
  }

  /*
   * return Random number in [1...n]
   */
  static uint32_t Uniform(int n) {
    return (random() % n) + 1;
  }

};

} // namespace myslash
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SLASH_RANDOM_H_