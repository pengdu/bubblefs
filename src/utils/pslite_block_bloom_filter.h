
// parameter_server/src/util/block_bloom_filter.h

#ifndef BUBBLEFS_UTILS_PSLITE_BLOCK_BLOOM_FILTER_H_
#define BUBBLEFS_UTILS_PSLITE_BLOCK_BLOOM_FILTER_H_

#include "utils/pslite_sketch.h"

namespace bubblefs {
namespace mypslite {

// a blocked version, see
// Cache-, Hash- and Space-Efficient Bloom Filters,
// http://algo2.iti.kit.edu/documents/cacheefficientbloomfilters-jea.pdf

// 1.2x - 1.8x faster than BloomFilter, but may give slightly large FPR
template <typename K>
class BlockBloomFilter : public Sketch {
 public:
  BlockBloomFilter() { }
  BlockBloomFilter(int m, int k) { resize(m, k); }
  ~BlockBloomFilter() { delete [] data_; }
  void resize(int m, int k) {
    m = std::max(m, 1024);
    num_bin_ = (m / 8 / bin_size_) + 1;
    data_size_ = num_bin_ * bin_size_;
    if (m > m_) {
      delete [] data_;
      data_ = new char[data_size_];
      // CHECK_EQ(posix_memalign((void**)&data_, bin_size_*8, data_size_), 0);
    }
    k_ = std::min(64, std::max(1, k));
    m_ = m;
    reset();
  }

  void reset() {
    memset(data_, 0, data_size_ * sizeof(char));
  }

  // make the api be similar to std::set
  bool count(K key) const { return query(key); }
  bool operator[] (K key) const { return query(key); }
  bool query(K key) const {
    // auto h = crc32(key);
    auto h = hash(key);
    auto delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    char* data = data_ + (h % num_bin_) * bin_size_;
    for (int j = 0; j < k_; ++j) {
      uint32 bitpos = h % (bin_size_ * 8);
      if ((data[bitpos/8] & (1 << (bitpos % 8))) == 0) return false;
      h += delta;
    }
    return true;
  }

  void insert(K key) {
    // auto h = crc32(key);
    auto h = hash(key);
    auto delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    char* data = data_ + (h % num_bin_) * bin_size_;
    for (int j = 0; j < k_; ++j) {
      uint32 bitpos = h % (bin_size_ * 8);
      data[bitpos/8] |= (1 << (bitpos % 8));
      h += delta;
    }
  }

 private:
  char* data_ = NULL;
  int data_size_ = 0;
  uint32 m_ = 0;
  int k_ = 0;
  const uint32 bin_size_ = 64;  // cache line size
  uint32 num_bin_ = 0;
};

} // namespace mypslite
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PSLITE_BLOCK_BLOOM_FILTER_H_