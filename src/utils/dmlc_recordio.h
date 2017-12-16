/*!
 *  Copyright (c) 2015 by Contributors
 * \file recordio.h
 * \brief recordio that is able to pack binary data into a splittable
 *   format, useful to exchange data in binary serialization,
 *   such as binary raw data or protobuf
 */

// dmlc-core/include/dmlc/recordio.h

#ifndef BUBBLEFS_UTILS_DMLC_RECORDIO_H_
#define BUBBLEFS_UTILS_DMLC_RECORDIO_H_

#include <algorithm>
#include <cstring>
#include <string>
#include "platform/base_error.h"
#include "utils/dmlc_base.h"
#include "utils/dmlc_io.h"

namespace bubblefs {
namespace mydmlc {
/*!
 * \brief writer of binary recordio
 *  binary format for recordio
 *  recordio format: magic lrecord data pad
 *
 *  - magic is magic number
 *  - pad is simply a padding space to make record align to 4 bytes
 *  - lrecord encodes length and continue bit
 *     - data.length() = (lrecord & (1U<<29U - 1));
 *     - cflag == (lrecord >> 29U) & 7;
 *
 *  cflag was used to handle (rare) special case when magic number
 *  occured in the data sequence.
 *
 *  In such case, the data is splitted into multiple records by
 *  the cells of magic number
 *
 *  (1) cflag == 0: this is a complete record;
 *  (2) cflag == 1: start of a multiple-rec;
 *      cflag == 2: middle of multiple-rec;
 *      cflag == 3: end of multiple-rec
 */
class RecordIOWriter {
 public:
  /*!
   * \brief magic number of recordio
   * note: (kMagic >> 29U) & 7 > 3
   * this ensures lrec will not be kMagic
   */
  static const uint32_t kMagic = 0xced7230a;
  /*!
   * \brief encode the lrecord
   * \param cflag cflag part of the lrecord
   * \param length length part of lrecord
   * \return the encoded data
   */
  inline static uint32_t EncodeLRec(uint32_t cflag, uint32_t length) {
    return (cflag << 29U) | length;
  }
  /*!
   * \brief decode the flag part of lrecord
   * \param rec the lrecord
   * \return the flag
   */
  inline static uint32_t DecodeFlag(uint32_t rec) {
    return (rec >> 29U) & 7U;
  }
  /*!
   * \brief decode the length part of lrecord
   * \param rec the lrecord
   * \return the length
   */
  inline static uint32_t DecodeLength(uint32_t rec) {
    return rec & ((1U << 29U) - 1U);
  }
  /*!
   * \brief constructor
   * \param stream the stream to be constructed
   */
  explicit RecordIOWriter(Stream *stream)
      : stream_(stream), seek_stream_(dynamic_cast<SeekStream*>(stream)),
        except_counter_(0) {
    PANIC_ENFORCE(sizeof(uint32_t) == 4, "uint32_t needs to be 4 bytes");
  }
  /*!
   * \brief write record to the stream
   * \param buf the buffer of memory region
   * \param size the size of record to write out
   */
  void WriteRecord(const void *buf, size_t size);
  /*!
   * \brief write record to the stream
   * \param data the data to write out
   */
  inline void WriteRecord(const std::string &data) {
    this->WriteRecord(data.c_str(), data.length());
  }
  /*!
   * \return number of exceptions(occurance of magic number)
   *   during the writing process
   */
  inline size_t except_counter(void) const {
    return except_counter_;
  }

  /*! \brief tell the current position of the input stream */
  inline size_t Tell(void) {
    PANIC_ENFORCE(seek_stream_ != NULL, "The input stream is not seekable");
    return seek_stream_->Tell();
  }

 private:
  /*! \brief output stream */
  Stream *stream_;
  /*! \brief seekable stream */
  SeekStream *seek_stream_;
  /*! \brief counts the number of exceptions */
  size_t except_counter_;
};
/*!
 * \brief reader of binary recordio to reads in record from stream
 * \sa RecordIOWriter
 */
class RecordIOReader {
 public:
  /*!
   * \brief constructor
   * \param stream the stream to be constructed
   */
  explicit RecordIOReader(Stream *stream)
      : stream_(stream), seek_stream_(dynamic_cast<SeekStream*>(stream)),
        end_of_stream_(false) {
    PANIC_ENFORCE(sizeof(uint32_t) == 4, "uint32_t needs to be 4 bytes");
  }
  /*!
   * \brief read next complete record from stream
   * \param out_rec used to store output record in string
   * \return true of read was successful, false if end of stream was reached
   */
  bool NextRecord(std::string *out_rec);

  /*! \brief seek to certain position of the input stream */
  inline void Seek(size_t pos) {
    PANIC_ENFORCE(seek_stream_ != NULL, "The input stream is not seekable");
    seek_stream_->Seek(pos);
  }

  /*! \brief tell the current position of the input stream */
  inline size_t Tell(void) {
    PANIC_ENFORCE(seek_stream_ != NULL, "The input stream is not seekable");
    return seek_stream_->Tell();
  }

 private:
  /*! \brief output stream */
  Stream *stream_;
  SeekStream *seek_stream_;
  /*! \brief whether we are at end of stream */
  bool end_of_stream_;
};

/*!
 * \brief reader of binary recordio from Blob returned by InputSplit
 *  This class divides the blob into several independent parts specified by caller,
 *  and read from one segment.
 *  The part reading can be used together with InputSplit::NextChunk for
 *  multi-threaded parsing(each thread take a RecordIOChunkReader)
 *
 * \sa RecordIOWriter, InputSplit
 */
class RecordIOChunkReader {
 public:
  /*!
   * \brief constructor
   * \param chunk source data returned by InputSplit
   * \param part_index which part we want to reado
   * \param num_parts number of total segments
   */
  explicit RecordIOChunkReader(InputSplit::Blob chunk,
                               unsigned part_index = 0,
                               unsigned num_parts = 1);
  /*!
   * \brief read next complete record from stream
   *   the blob contains the memory content
   *   NOTE: this function is not threadsafe, use one
   *   RecordIOChunkReader per thread
   * \param out_rec used to store output blob, the header is already
   *        removed and out_rec only contains the memory content
   * \return true of read was successful, false if end was reached
   */
  bool NextRecord(InputSplit::Blob *out_rec);

 private:
  /*! \brief internal temporal data */
  std::string temp_;
  /*! \brief internal data pointer */
  char *pbegin_, *pend_;
};

// implementation
void RecordIOWriter::WriteRecord(const void *buf, size_t size) {
  PANIC_ENFORCE(size < (1 << 29U), "RecordIO only accept record less than 2^29 bytes");
  const uint32_t umagic = kMagic;
  // initialize the magic number, in stack
  const char *magic = reinterpret_cast<const char*>(&umagic);
  const char *bhead = reinterpret_cast<const char*>(buf);
  uint32_t len = static_cast<uint32_t>(size);
  uint32_t lower_align = (len >> 2U) << 2U;
  uint32_t upper_align = ((len + 3U) >> 2U) << 2U;
  uint32_t dptr = 0;
  for (uint32_t i = 0; i < lower_align ; i += 4) {
    // use char check for alignment safety reason
    if (bhead[i] == magic[0] &&
        bhead[i + 1] == magic[1] &&
        bhead[i + 2] == magic[2] &&
        bhead[i + 3] == magic[3]) {
      uint32_t lrec = EncodeLRec(dptr == 0 ? 1U : 2U,
                                 i - dptr);
      stream_->Write(magic, 4);
      stream_->Write(&lrec, sizeof(lrec));
      if (i != dptr) {
        stream_->Write(bhead + dptr, i - dptr);
      }
      dptr = i + 4;
      except_counter_ += 1;
    }
  }
  uint32_t lrec = EncodeLRec(dptr != 0 ? 3U : 0U,
                             len - dptr);
  stream_->Write(magic, 4);
  stream_->Write(&lrec, sizeof(lrec));
  if (len != dptr) {
    stream_->Write(bhead + dptr, len - dptr);
  }
  // write padded bytes
  uint32_t zero = 0;
  if (upper_align != len) {
    stream_->Write(&zero, upper_align - len);
  }
}

bool RecordIOReader::NextRecord(std::string *out_rec) {
  if (end_of_stream_) return false;
  const uint32_t kMagic = RecordIOWriter::kMagic;
  out_rec->clear();
  size_t size = 0;
  while (true) {
    uint32_t header[2];
    size_t nread = stream_->Read(header, sizeof(header));
    if (nread == 0) {
      end_of_stream_ = true; return false;
    }
    PANIC_ENFORCE(nread == sizeof(header), "Inavlid RecordIO File");
    PANIC_ENFORCE(header[0] == RecordIOWriter::kMagic, "Invalid RecordIO File");
    uint32_t cflag = RecordIOWriter::DecodeFlag(header[1]);
    uint32_t len = RecordIOWriter::DecodeLength(header[1]);
    uint32_t upper_align = ((len + 3U) >> 2U) << 2U;
    out_rec->resize(size + upper_align);
    if (upper_align != 0) {
      PANIC_ENFORCE(stream_->Read(BeginPtr(*out_rec) + size, upper_align) == upper_align,
                    "Invalid RecordIO File upper_align=%u", upper_align);
    }
    // squeeze back
    size += len; out_rec->resize(size);
    if (cflag == 0U || cflag == 3U) break;
    out_rec->resize(size + sizeof(kMagic));
    std::memcpy(BeginPtr(*out_rec) + size, &kMagic, sizeof(kMagic));
    size += sizeof(kMagic);
  }
  return true;
}

// helper function to find next recordio head
inline char *FindNextRecordIOHead(char *begin, char *end) {
  PANIC_ENFORCE_EQ((reinterpret_cast<size_t>(begin) & 3UL),  0U);
  PANIC_ENFORCE_EQ((reinterpret_cast<size_t>(end) & 3UL), 0U);
  uint32_t *p = reinterpret_cast<uint32_t *>(begin);
  uint32_t *pend = reinterpret_cast<uint32_t *>(end);
  for (; p + 1 < pend; ++p) {
    if (p[0] == RecordIOWriter::kMagic) {
      uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
      if (cflag == 0 || cflag == 1) {
        return reinterpret_cast<char*>(p);
      }
    }
  }
  return end;
}

RecordIOChunkReader::RecordIOChunkReader(InputSplit::Blob chunk,
                                         unsigned part_index,
                                         unsigned num_parts) {
  size_t nstep = (chunk.size + num_parts - 1) / num_parts;
  // align
  nstep = ((nstep + 3UL) >> 2UL) << 2UL;
  size_t begin = std::min(chunk.size, nstep * part_index);
  size_t end = std::min(chunk.size, nstep * (part_index + 1));
  char *head = reinterpret_cast<char*>(chunk.dptr);
  pbegin_ = FindNextRecordIOHead(head + begin, head + chunk.size);
  pend_ = FindNextRecordIOHead(head + end, head + chunk.size);
}

bool RecordIOChunkReader::NextRecord(InputSplit::Blob *out_rec) {
  if (pbegin_ >= pend_) return false;
  uint32_t *p = reinterpret_cast<uint32_t *>(pbegin_);
  PANIC_ENFORCE(p[0] == RecordIOWriter::kMagic, "");
  uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
  uint32_t clen = RecordIOWriter::DecodeLength(p[1]);
  if (cflag == 0) {
    // skip header
    out_rec->dptr = pbegin_ + 2 * sizeof(uint32_t);
    // move pbegin
    pbegin_ += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
    PANIC_ENFORCE(pbegin_ <= pend_, "Invalid RecordIO Format");
    out_rec->size = clen;
    return true;
  } else {
    const uint32_t kMagic = RecordIOWriter::kMagic;
    // abnormal path, read into string
    PANIC_ENFORCE(cflag == 1U, "Invalid RecordIO Format");
    temp_.resize(0);
    while (true) {
      PANIC_ENFORCE(pbegin_ + 2 * sizeof(uint32_t) <= pend_, "");
      p = reinterpret_cast<uint32_t *>(pbegin_, "");
      PANIC_ENFORCE(p[0] == RecordIOWriter::kMagic);
      cflag = RecordIOWriter::DecodeFlag(p[1]);
      clen = RecordIOWriter::DecodeLength(p[1]);
      size_t tsize = temp_.length();
      temp_.resize(tsize + clen);
      if (clen != 0) {
        std::memcpy(BeginPtr(temp_) + tsize,
                    pbegin_ + 2 * sizeof(uint32_t),
                    clen);
        tsize += clen;
      }
      pbegin_ += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
      if (cflag == 3U) break;
      temp_.resize(tsize + sizeof(kMagic));
      std::memcpy(BeginPtr(temp_) + tsize, &kMagic, sizeof(kMagic));
    }
    out_rec->dptr = BeginPtr(temp_);
    out_rec->size = temp_.length();
    return true;
  }
}

}  // namespace mydmlc
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_DMLC_RECORDIO_H_