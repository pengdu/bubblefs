/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/lib/io/record_reader.cc

#include "utils/record_reader.h"
#include <limits.h>
#include "platform/env.h"
#include "utils/buffered_inputstream.h"
#include "utils/coding.h"
#include "utils/crc32c.h"
#include "utils/errors.h"
#include "utils/random_inputstream.h"

namespace bubblefs {
namespace io {

RecordReaderOptions RecordReaderOptions::CreateRecordReaderOptions(
    const string& compression_type) {
  RecordReaderOptions options;
  if (compression_type == "ZLIB") {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
  } else {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No comprression will be used.";
  }
  return options;
}

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : src_(file), options_(options) {
  if (options.buffer_size > 0) {
    input_stream_.reset(new BufferedInputStream(file, options.buffer_size));
  } else {
    input_stream_.reset(new RandomAccessInputStream(file));
  }
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
// We don't have zlib available on all embedded platforms, so fail.
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
// May use *storage as backing store.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n,
                                     StringPiece* result, string* storage) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  storage->resize(expected);

    if (options_.buffer_size > 0) {
      // If we have a buffer, we assume that the file is being read
      // sequentially, and we use the underlying implementation to read the
      // data.
      //
      // No checks are done to validate that the file is being read
      // sequentially.
      TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(expected, storage));

      if (storage->size() != expected) {
        if (storage->empty()) {
          return errors::OutOfRange("eof");
        } else {
          return errors::DataLoss("truncated record at ", offset);
        }
      }

      const uint32 masked_crc = core::DecodeFixed32(storage->data() + n);
      if (crc32c::Unmask(masked_crc) != crc32c::Value(storage->data(), n)) {
        return errors::DataLoss("corrupted record at ", offset);
      }
      *result = StringPiece(storage->data(), n);
    } else {
      // This version supports reading from arbitrary offsets
      // since we are accessing the random access file directly.
      StringPiece data;
      TF_RETURN_IF_ERROR(src_->Read(offset, expected, &data, &(*storage)[0]));
      if (data.size() != expected) {
        if (data.empty()) {
          return errors::OutOfRange("eof");
        } else {
          return errors::DataLoss("truncated record at ", offset);
        }
      }
      const uint32 masked_crc = core::DecodeFixed32(data.data() + n);
      if (crc32c::Unmask(masked_crc) != crc32c::Value(data.data(), n)) {
        return errors::DataLoss("corrupted record at ", offset);
      }
      *result = StringPiece(data.data(), n);
    }

  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, string* record) {
  static const size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static const size_t kFooterSize = sizeof(uint32);

  // Read header data.
  StringPiece lbuf;
  Status s = ReadChecksummed(*offset, sizeof(uint64), &lbuf, record);
  if (!s.ok()) {
    return s;
  }
  const uint64 length = core::DecodeFixed64(lbuf.data());

  // Read data
  StringPiece data;
  s = ReadChecksummed(*offset + kHeaderSize, length, &data, record);
  if (!s.ok()) {
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record at ", *offset);
    }
    return s;
  }

  if (record->data() != data.data()) {
    // RandomAccessFile placed the data in some other location.
    memmove(&(*record)[0], data.data(), data.size());
  }

  record->resize(data.size());

  *offset += kHeaderSize + length + kFooterSize;
  return Status::OK();
}

Status RecordReader::SkipNBytes(uint64 offset) {
    if (options_.buffer_size > 0) {
      TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(offset));
    }
  return Status::OK();
}

SequentialRecordReader::SequentialRecordReader(
    RandomAccessFile* file, const RecordReaderOptions& options)
    : underlying_(file, options), offset_(0) {}

}  // namespace io
}  // namespace bubblefs