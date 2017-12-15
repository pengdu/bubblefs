/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/blob_serializer_base.h

#ifndef BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_
#define BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_

#include <string>
#include <functional>

namespace bubblefs {
namespace mycaffe2 {

class Blob;

constexpr int kDefaultChunkSize = -1;
constexpr int kNoChunking = 0;

/**
 * @brief BlobSerializerBase is an abstract class that serializes a blob to a
 * string.
 *
 * This class exists purely for the purpose of registering type-specific
 * serialization code. If you need to serialize a specific type, you should
 * write your own Serializer class, and then register it using
 * REGISTER_BLOB_SERIALIZER. For a detailed example, see TensorSerializer for
 * details.
 */
class BlobSerializerBase {
 public:
  virtual ~BlobSerializerBase() {}
  using SerializationAcceptor =
     std::function<void(const std::string& blobName, const std::string& data)>;
  /**
   * @brief The virtual function that returns a serialized string for the input
   * blob.
   * @param blob
   *     the input blob to be serialized.
   * @param name
   *     the blob name to be used in the serialization implementation. It is up
   *     to the implementation whether this name field is going to be used or
   *     not.
   * @param acceptor
   *     a lambda which accepts key value pairs to save them to storage.
   *     serailizer can use it to save blob in several chunks
   *     acceptor should be thread-safe
   */
  virtual void Serialize(const Blob& blob, const std::string& name,
                        SerializationAcceptor acceptor) = 0;

  virtual void SerializeWithChunkSize(
      const Blob& blob,
      const std::string& name,
      SerializationAcceptor acceptor,
      int /*chunk_size*/) {
    // Base implementation.
    Serialize(blob, name, acceptor);
  }
};

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_