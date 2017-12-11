
// parameter_server/src/util/bloom_filter.h

#include "utils/pslite_sketch.h"

namespace bubblefs {
namespace mypslite {

template <typename K>
class BloomFilter : public Sketch {
 public:
  BloomFilter() { }
  BloomFilter(int m, int k) { resize(m, k); }
  ~BloomFilter() { delete [] data_; }
  void resize(int m, int k) {
    delete [] data_;
    k_ = std::min(64, std::max(1, k));
    m_ = m;
    data_size_ = (m / 8) + 1;
    data_ = new char[data_size_];
    memset(data_, 0, data_size_ * sizeof(char));
  }

  bool operator[] (K key) const { return query(key); }
  bool query(K key) const {
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int j = 0; j < k_; ++j) {
      uint32 bitpos = h % m_;
      if ((data_[bitpos/8] & (1 << (bitpos % 8))) == 0) return false;
      h += delta;
    }
    return true;
  }

  void insert(K key) {
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int j = 0; j < k_; ++j) {
      uint32 bitpos = h % m_;
      data_[bitpos/8] |= (1 << (bitpos % 8));
      h += delta;
    }
  }

 private:
  char* data_ = NULL;
  int data_size_ = 0;
  uint32 m_ = 0;
  int k_ = 0;
};

} // namespace mypslite
} // namespace bubblefs