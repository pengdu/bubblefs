/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/Reader.h
// LucenePlusPlus/src/core/util/Reader.cpp

#ifndef BUBBLEFS_UTILS_LUCENE_READER_H_
#define BUBBLEFS_UTILS_LUCENE_READER_H_

#include "utils/lucene_object.h"

namespace bubblefs {
namespace mylucene {

/// Abstract class for reading character streams.
class Reader : public Object {
protected:
    Reader() {};

public:
    virtual ~Reader() {};
    LUCENE_CLASS(Reader)

public:
    static constexpr int32_t READER_EOF = -1;

    /// Read a single character.
    virtual int32_t read() {
      wchar_t buffer;
      return read(&buffer, 0, 1) == READER_EOF ? READER_EOF : buffer;
    };

    /// Read characters into a portion of an array.
    virtual int32_t read(wchar_t* buffer, int32_t offset, int32_t length) = 0;

    /// Skip characters.
    virtual int64_t skip(int64_t n) { return 0; };

    /// Close the stream.
    virtual void close() = 0;

    /// Tell whether this stream supports the mark() operation
    virtual bool markSupported() { return false; };

    /// Mark the present position in the stream.  Subsequent calls to reset() will attempt to reposition the
    /// stream to this point.
    virtual void mark(int32_t readAheadLimit) {};

    /// Reset the stream. If the stream has been marked, then attempt to reposition it at the mark.  If the stream
    /// has not been marked, then attempt to reset it in some way appropriate to the particular stream, for example
    /// by repositioning it to its starting point.
    virtual void reset() {};

    /// The number of bytes in the stream.
    virtual int64_t length() { return 0; };
};

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_READER_H_