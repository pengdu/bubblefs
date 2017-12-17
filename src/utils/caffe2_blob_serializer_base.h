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
// caffe2/caffe2/core/blob_serialization.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_
#define BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_

#include <string>
#include <functional>
#include "platform/base_error.h"

namespace bubblefs {
namespace mycaffe2 {

class Blob;
class BlobProto;

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

/**
 * @brief BlobDeserializerBase is an abstract class that deserializes a blob
 * from a BlobProto or a TensorProto.
 */
class BlobDeserializerBase {
 public:
  virtual ~BlobDeserializerBase() {}

  // Deserializes from a BlobProto object.
  virtual void Deserialize(const BlobProto& proto, Blob* blob) = 0;
};

/**
 * @brief StringSerializer is the serializer for String.
 *
 * StringSerializer takes in a blob that contains a String, and serializes it
 * into a BlobProto protocol buffer.
 */
class StringSerializer : public BlobSerializerBase {
 public:
  StringSerializer() {}
  ~StringSerializer() {}
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor<Context>,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override {
    PANIC_ENFORCE(blob.IsType<std::string>(), "");

    BlobProto blob_proto;
    //blob_proto.set_name(name);
    //blob_proto.set_type("std::string");
    //blob_proto.set_content(blob.template Get<std::string>());
    acceptor(name, blob_proto.SerializeAsString());
  }
};

/**
 * @brief StringDeserializer is the deserializer for Strings.
 *
 */
class StringDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    *blob->GetMutable<std::string>() = proto.content();
  }
};

// The blob serialization member function implementation.
void Blob::Serialize(
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(meta_.id()));
  PANIC_ENFORCE(serializer, "No known serializer for ", meta_.name());
  serializer->SerializeWithChunkSize(*this, name, acceptor, chunk_size);
}

// The blob serialization member function implementation.
std::string Blob::Serialize(const std::string& name) const {
  std::string data;
  BlobSerializerBase::SerializationAcceptor acceptor = [&data](
      const std::string&, const std::string& blob) {
    //DCHECK(data.empty()); // should be called once with kNoChunking
    data = blob;
  };
  this->Serialize(name, acceptor, kNoChunking);
  return data;
}
void Blob::Deserialize(const std::string& content) {
  BlobProto blob_proto;
  this->Deserialize(blob_proto);
}

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_BLOB_SERIALIZER_BASE_H_