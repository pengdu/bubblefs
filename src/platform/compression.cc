
#include "platform/compression.h"
#if TF_USE_SNAPPY
#include "snappy.h"
#endif

namespace bubblefs {
namespace port {

bool Snappy_Compress(const char* input, size_t length, string* output) {
#if TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
#if TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
#if TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}

}  // namespace port
}  // namespace bubblefs