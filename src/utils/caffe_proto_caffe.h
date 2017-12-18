
// caffe/src/caffe/proto/caffe.proto

#ifndef BUBBLEFS_UTILS_CAFFE_PROTO_CAFFE_H_
#define BUBBLEFS_UTILS_CAFFE_PROTO_CAFFE_H_

//syntax = "proto2";
//package caffe;

#include <map>
#include <string>
#include <vector>
#include "platform/types.h"

namespace bubblefs {
namespace mycaffe {

typedef std::string bytes;  
  
// Specifies the shape (dimensions) of a Blob.
struct BlobShape {
  std::vector<int64> dim;
};

struct BlobProto {
  BlobShape shape;
  std::vector<float> data;
  std::vector<float> diff;
  std::vector<double> double_data;
  std::vector<double> double_diff;

  // 4D dimensions -- deprecated.  Use "shape" instead.
  int32 num;
  int32 channels;
  int32 height;
  int32 width;
};

// The BlobProtoVector is simply a way to pass multiple blobproto instances
// around.
struct BlobProtoVector {
  std::vector<BlobProto> blobs;
};  

struct Datum {
  int32 channels;
  int32 height;
  int32 width;
  // the actual image data, in bytes
  bytes data;
  int32 label5;
  // Optionally, the datum could also hold float data.
  std::vector<float> float_data;
  // If true data contains an encoded image that need to be decoded
  bool encoded;
};
  
} // namespace mycaffe  
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE_PROTO_CAFFE_H_ 