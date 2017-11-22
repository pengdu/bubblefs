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

// tensorflow/tensorflow/core/lib/io/record_writer.cc

#include "utils/record_writer.h"
#include "utils/coding.h"
#include "utils/crc32c.h"
#include "platform/env.h"

namespace bubblefs {
namespace io {
namespace {
bool IsZlibCompressed(RecordWriterOptions options) {
  return options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION;
}
}  // namespace

RecordWriterOptions RecordWriterOptions::CreateRecordWriterOptions(
    const string& compression_type) {
  RecordWriterOptions options;
  if (compression_type == "ZLIB") {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
    //options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
  } else {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No comprression will be used.";
  }
  return options;
}

RecordWriter::RecordWriter(WritableFile* dest,
                           const RecordWriterOptions& options)
    : dest_(dest), options_(options) {
  if (IsZlibCompressed(options)) {
// We don't have zlib available on all embedded platforms, so fail.
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
  } else if (options.compression_type == RecordWriterOptions::NONE) {
    // Nothing to do
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordWriter::~RecordWriter() {
  if (dest_ != nullptr) {
    Status s = Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

static uint32 MaskedCrc(const char* data, size_t n) {
  return crc32c::Mask(crc32c::Value(data, n));
}

Status RecordWriter::WriteRecord(StringPiece data) {
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[sizeof(uint64) + sizeof(uint32)];
  core::EncodeFixed64(header + 0, data.size());
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
  char footer[sizeof(uint32)];
  core::EncodeFixed32(footer, MaskedCrc(data.data(), data.size()));

  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  TF_RETURN_IF_ERROR(dest_->Append(data));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}

Status RecordWriter::Close() {
  if (IsZlibCompressed(options_)) {
    Status s = dest_->Close();
    delete dest_;
    dest_ = nullptr;
    return s;
  }
  return Status::OK();
}

Status RecordWriter::Flush() {
  if (IsZlibCompressed(options_)) {
    return dest_->Flush();
  }
  return Status::OK();
}

}  // namespace io
}  // namespace bubblefs