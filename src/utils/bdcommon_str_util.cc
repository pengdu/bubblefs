// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/base/string_number.cc
// tera/src/common/base/string_ext.cc

#include "utils/bdcommon_str_util.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include "utils/string_format.h"

// GLOBAL_NOLINT(runtime/int)

#define ARRAY_SIZE(a) \
    ((sizeof(a) / sizeof(*(a))) / (size_t)(!(sizeof(a) % sizeof(*(a)))))

using namespace std;

namespace bubblefs {
namespace bdcommon {

namespace
{

template <typename T>
struct StringToNumber
{
    // static T Convert(const char* str, char** endptr, int base);
};

template <>
struct StringToNumber<long>
{
    static long Convert(const char* str, char** endptr, int base)
    {
        return strtol(str, endptr, base);
    }
};

template <>
struct StringToNumber<unsigned long>
{
    static unsigned long Convert(const char* str, char** endptr, int base)
    {
        return strtoul(str, endptr, base);
    }
};

template <>
struct StringToNumber<long long>
{
    static long long Convert(const char* str, char** endptr, int base)
    {
        return strtoll(str, endptr, base);
    }
};

template <>
struct StringToNumber<unsigned long long>
{
    static unsigned long long Convert(const char* str, char** endptr, int base)
    {
        return strtoull(str, endptr, base);
    }
};

template <typename IntermediaType, typename T>
bool ParseNumberT(const char* str, T* value, char** endptr, int base)
{
//     STATIC_ASSERT(TypeTraits::IsSignedInteger<T>::Value ==
//                   TypeTraits::IsSignedInteger<IntermediaType>::Value);
//     STATIC_ASSERT(sizeof(T) <= sizeof(IntermediaType));

    int old_errno = errno;
    errno = 0;
    IntermediaType number = StringToNumber<IntermediaType>::Convert(str, endptr, base);
    if (errno != 0)
        return false;

    if (sizeof(T) < sizeof(IntermediaType) &&
        (number > std::numeric_limits<T>::max() || number < std::numeric_limits<T>::min()))
    {
        errno = ERANGE;
        return false;
    }

    if (*endptr == str)
    {
        errno = EINVAL;
        return false;
    }

    errno = old_errno;
    *value = static_cast<T>(number);
    return true;
}

}

bool ParseNumber(const char* str, signed char* value, char** endptr, int base)
{
    return ParseNumberT<long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, unsigned char* value, char** endptr, int base)
{
    return ParseNumberT<unsigned long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, short* value, char** endptr, int base)
{
    return ParseNumberT<long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, unsigned short* value, char** endptr, int base)
{
    return ParseNumberT<unsigned long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, int* value, char** endptr, int base)
{
    return ParseNumberT<long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, unsigned int* value, char** endptr, int base)
{
    return ParseNumberT<unsigned long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, long* value, char** endptr, int base)
{
    return ParseNumberT<long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, unsigned long* value, char** endptr, int base)
{
    return ParseNumberT<unsigned long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, long long* value, char** endptr, int base)
{
    return ParseNumberT<long long>(str, value, endptr, base);
}

bool ParseNumber(const char* str, unsigned long long* value, char** endptr, int base)
{
    return ParseNumberT<unsigned long long>(str, value, endptr, base);
}

namespace
{

template <typename T> struct StringToFloat { };

template <>
struct StringToFloat<float>
{
    static float Convert(const char* str, char** endptr)
    {
        return strtof(str, endptr);
    }
};

template <>
struct StringToFloat<double>
{
    static double Convert(const char* str, char** endptr)
    {
        return strtod(str, endptr);
    }
};

template <>
struct StringToFloat<long double>
{
    static long double Convert(const char* str, char** endptr)
    {
        return strtold(str, endptr);
    }
};

template <typename T>
bool ParseFloatNumber(const char* str, T* value, char** endptr)
{
    int old_errno = errno;
    errno = 0;
    *value = StringToFloat<T>::Convert(str, endptr);
    if (errno != 0)
        return false;
    if (*endptr == str)
        errno = EINVAL;
    errno = old_errno;
    return true;
}

}

bool ParseNumber(const char* str, float* value, char** endptr)
{
    return ParseFloatNumber(str, value, endptr);
}

bool ParseNumber(const char* str, double* value, char** endptr)
{
    return ParseFloatNumber(str, value, endptr);
}

bool ParseNumber(const char* str, long double* value, char** endptr)
{
    return ParseFloatNumber(str, value, endptr);
}

// ---------------------------------------------------------
// unsigned int to hex buffer or string.
// ---------------------------------------------------------
static char *UIntToHexBufferInternal(uint64_t value, char* buffer, int num_byte)
{
    static const char hexdigits[] = "0123456789abcdef";
    int digit_byte = 2 * num_byte;
    for (int i = digit_byte - 1; i >= 0; i--)
    {
        buffer[i] = hexdigits[uint32_t(value) & 0xf];
        value >>= 4;
    }
    return buffer + digit_byte;
}

char* WriteHexUInt16ToBuffer(uint16_t value, char* buffer)
{
    return UIntToHexBufferInternal(value, buffer, sizeof(value));
}

char* WriteHexUInt32ToBuffer(uint32_t value, char* buffer)
{
    return UIntToHexBufferInternal(value, buffer, sizeof(value));
}

char* WriteHexUInt64ToBuffer(uint64_t value, char* buffer)
{
    return UIntToHexBufferInternal(value, buffer, sizeof(value));
}

char* UInt16ToHexString(uint16_t value, char* buffer)
{
    *WriteHexUInt16ToBuffer(value, buffer) = '\0';
    return buffer;
}

char* UInt32ToHexString(uint32_t value, char* buffer)
{
    *WriteHexUInt32ToBuffer(value, buffer) = '\0';
    return buffer;
}

char* UInt64ToHexString(uint64_t value, char* buffer)
{
    *WriteHexUInt64ToBuffer(value, buffer) = '\0';
    return buffer;
}

string UInt16ToHexString(uint16_t value)
{
    char buffer[2*sizeof(value) + 1];
    return std::string(buffer, WriteHexUInt16ToBuffer(value, buffer));
}

string UInt32ToHexString(uint32_t value)
{
    char buffer[2*sizeof(value) + 1];
    return std::string(buffer, WriteHexUInt32ToBuffer(value, buffer));
}

string UInt64ToHexString(uint64_t value)
{
    char buffer[2*sizeof(value) + 1];
    return std::string(buffer, WriteHexUInt64ToBuffer(value, buffer));
}

// -----------------------------------------------------------------
// Double to string or buffer.
// Make sure buffer size >= kMaxDoubleStringSize
// -----------------------------------------------------------------
char* WriteDoubleToBuffer(double value, char* buffer)
{
    // DBL_DIG is 15 on almost all platforms.
    // If it's too big, the buffer will overflow
//     STATIC_ASSERT(DBL_DIG < 20, "DBL_DIG is too big");

    if (value >= numeric_limits<double>::infinity())
    {
        strcpy(buffer, "inf"); // NOLINT
        return buffer + 3;
    }
    else if (value <= -numeric_limits<double>::infinity())
    {
        strcpy(buffer, "-inf"); // NOLINT
        return buffer + 4;
    }
    else if (IsNaN(value))
    {
        strcpy(buffer, "nan"); // NOLINT
        return buffer + 3;
    }

    return buffer + snprintf(buffer, kMaxDoubleStringSize, "%.*g", DBL_DIG, value);
}

// -------------------------------------------------------------
// Float to string or buffer.
// Makesure buffer size >= kMaxFloatStringSize
// -------------------------------------------------------------
char* WriteFloatToBuffer(float value, char* buffer)
{
    // FLT_DIG is 6 on almost all platforms.
    // If it's too big, the buffer will overflow
//     STATIC_ASSERT(FLT_DIG < 10, "FLT_DIG is too big");
    if (value >= numeric_limits<double>::infinity())
    {
        strcpy(buffer, "inf"); // NOLINT
        return buffer + 3;
    }
    else if (value <= -numeric_limits<double>::infinity())
    {
        strcpy(buffer, "-inf"); // NOLINT
        return buffer + 4;
    }
    else if (IsNaN(value))
    {
        strcpy(buffer, "nan"); // NOLINT
        return buffer + 3;
    }

    return buffer + snprintf(buffer, kMaxFloatStringSize, "%.*g", FLT_DIG, value);
}

char* DoubleToString(double n, char* buffer)
{
    WriteDoubleToBuffer(n, buffer);
    return buffer;
}

char* FloatToString(float n, char* buffer)
{
    WriteFloatToBuffer(n, buffer);
    return buffer;
}

string DoubleToString(double value)
{
    char buffer[kMaxDoubleStringSize];
    return std::string(buffer, WriteDoubleToBuffer(value, buffer));
}

string FloatToString(float value)
{
    char buffer[kMaxFloatStringSize];
    return std::string(buffer, WriteFloatToBuffer(value, buffer));
}

// ------------------------------------------------------
// Int to string or buffer.
// The following data and functions are for internal use.
// ------------------------------------------------------
static const char two_ASCII_digits[100][2] = {
    {'0', '0'}, {'0', '1'}, {'0', '2'}, {'0', '3'}, {'0', '4'},
    {'0', '5'}, {'0', '6'}, {'0', '7'}, {'0', '8'}, {'0', '9'},
    {'1', '0'}, {'1', '1'}, {'1', '2'}, {'1', '3'}, {'1', '4'},
    {'1', '5'}, {'1', '6'}, {'1', '7'}, {'1', '8'}, {'1', '9'},
    {'2', '0'}, {'2', '1'}, {'2', '2'}, {'2', '3'}, {'2', '4'},
    {'2', '5'}, {'2', '6'}, {'2', '7'}, {'2', '8'}, {'2', '9'},
    {'3', '0'}, {'3', '1'}, {'3', '2'}, {'3', '3'}, {'3', '4'},
    {'3', '5'}, {'3', '6'}, {'3', '7'}, {'3', '8'}, {'3', '9'},
    {'4', '0'}, {'4', '1'}, {'4', '2'}, {'4', '3'}, {'4', '4'},
    {'4', '5'}, {'4', '6'}, {'4', '7'}, {'4', '8'}, {'4', '9'},
    {'5', '0'}, {'5', '1'}, {'5', '2'}, {'5', '3'}, {'5', '4'},
    {'5', '5'}, {'5', '6'}, {'5', '7'}, {'5', '8'}, {'5', '9'},
    {'6', '0'}, {'6', '1'}, {'6', '2'}, {'6', '3'}, {'6', '4'},
    {'6', '5'}, {'6', '6'}, {'6', '7'}, {'6', '8'}, {'6', '9'},
    {'7', '0'}, {'7', '1'}, {'7', '2'}, {'7', '3'}, {'7', '4'},
    {'7', '5'}, {'7', '6'}, {'7', '7'}, {'7', '8'}, {'7', '9'},
    {'8', '0'}, {'8', '1'}, {'8', '2'}, {'8', '3'}, {'8', '4'},
    {'8', '5'}, {'8', '6'}, {'8', '7'}, {'8', '8'}, {'8', '9'},
    {'9', '0'}, {'9', '1'}, {'9', '2'}, {'9', '3'}, {'9', '4'},
    {'9', '5'}, {'9', '6'}, {'9', '7'}, {'9', '8'}, {'9', '9'}
};

template <typename OutputIterator>
static OutputIterator OutputUInt32AsString(uint32_t u, OutputIterator output)
{
    int digits;
    const char *ASCII_digits = NULL;
    if (u >= 1000000000) // >= 1,000,000,000
    {
        digits = u / 100000000;  // 100,000,000
        ASCII_digits = two_ASCII_digits[digits];
        *output++ = ASCII_digits[0];
        *output++ = ASCII_digits[1];
sublt100_000_000:
        u -= digits * 100000000;  // 100,000,000
lt100_000_000:
        digits = u / 1000000;  // 1,000,000
        ASCII_digits = two_ASCII_digits[digits];
        *output++ = ASCII_digits[0];
        *output++ = ASCII_digits[1];
sublt1_000_000:
        u -= digits * 1000000;  // 1,000,000
lt1_000_000:
        digits = u / 10000;  // 10,000
        ASCII_digits = two_ASCII_digits[digits];
        *output++ = ASCII_digits[0];
        *output++ = ASCII_digits[1];
sublt10_000:
        u -= digits * 10000;  // 10,000
lt10_000:
        digits = u / 100;
        ASCII_digits = two_ASCII_digits[digits];
        *output++ = ASCII_digits[0];
        *output++ = ASCII_digits[1];
sublt100:
        u -= digits * 100;
lt100:
        digits = u;
        ASCII_digits = two_ASCII_digits[digits];
        *output++ = ASCII_digits[0];
        *output++ = ASCII_digits[1];
done:
        return output;
    }

    if (u < 100)
    {
        digits = u;
        if (u >= 10) goto lt100;
        *output++ = '0' + digits;
        goto done;
    }
    if (u  <  10000) // 10,000
    {
        if (u >= 1000) goto lt10_000;
        digits = u / 100;
        *output++ = '0' + digits;
        goto sublt100;
    }
    if (u  <  1000000) // 1,000,000
    {
        if (u >= 100000) goto lt1_000_000;
        digits = u / 10000;  //    10,000
        *output++ = '0' + digits;
        goto sublt10_000;
    }
    if (u  <  100000000) // 100,000,000
    {
        if (u >= 10000000) goto lt100_000_000;
        digits = u / 1000000;  //   1,000,000
        *output++ = '0' + digits;
        goto sublt1_000_000;
    }
    // u < 1,000,000,000
    digits = u / 100000000;   // 100,000,000
    *output++ = '0' + digits;
    goto sublt100_000_000;
}

template <typename OutputIterator>
OutputIterator OutputInt32AsString(int32_t i, OutputIterator output)
{
    uint32_t u = i;
    if (i < 0)
    {
        *output++ = '-';
        u = -i;
    }
    return OutputUInt32AsString(u, output);
}

template <typename OutputIterator>
OutputIterator OutputUInt64AsString(uint64_t u64, OutputIterator output)
{
    int digits;
    const char *ASCII_digits = NULL;

    uint32_t u = static_cast<uint32_t>(u64);
    if (u == u64) return OutputUInt32AsString(u, output);

    uint64_t top_11_digits = u64 / 1000000000;
    output = OutputUInt64AsString(top_11_digits, output);
    u = static_cast<uint32_t>(u64 - (top_11_digits * 1000000000));

    digits = u / 10000000;  // 10,000,000
    ASCII_digits = two_ASCII_digits[digits];
    *output++ = ASCII_digits[0];
    *output++ = ASCII_digits[1];
    u -= digits * 10000000;  // 10,000,000
    digits = u / 100000;  // 100,000
    ASCII_digits = two_ASCII_digits[digits];
    *output++ = ASCII_digits[0];
    *output++ = ASCII_digits[1];
    u -= digits * 100000;  // 100,000
    digits = u / 1000;  // 1,000
    ASCII_digits = two_ASCII_digits[digits];
    *output++ = ASCII_digits[0];
    *output++ = ASCII_digits[1];
    u -= digits * 1000;  // 1,000
    digits = u / 10;
    ASCII_digits = two_ASCII_digits[digits];
    *output++ = ASCII_digits[0];
    *output++ = ASCII_digits[1];
    u -= digits * 10;
    digits = u;
    *output++ = '0' + digits;
    return output;
}

template <typename OutputIterator>
OutputIterator OutputInt64AsString(int64_t i, OutputIterator output)
{
    uint64_t u = i;
    if (i < 0)
    {
        *output++ = '-';
        u = -i;
    }
    return OutputUInt64AsString(u, output);
}

///////////////////////////////////////////////////////////////////////////
// generic interface

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(int n, OutputIterator output)
{
    return OutputInt32AsString(n, output);
}

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(unsigned int n, OutputIterator output)
{
    return OutputUInt32AsString(n, output);
}

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(long n, OutputIterator output)
{
    return sizeof(n) == 4 ?
        OutputInt32AsString(static_cast<int32_t>(n), output):
        OutputInt64AsString(static_cast<int64_t>(n), output);
}

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(unsigned long n, OutputIterator output)
{
    return sizeof(n) == 4 ?
        OutputUInt32AsString(static_cast<uint32_t>(n), output):
        OutputUInt64AsString(static_cast<uint64_t>(n), output);
}

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(long long n, OutputIterator output)
{
    return sizeof(n) == 4 ?
        OutputInt32AsString(static_cast<int32_t>(n), output):
        OutputInt64AsString(static_cast<int64_t>(n), output);
}

template <typename OutputIterator>
OutputIterator OutputIntegerAsString(unsigned long long n, OutputIterator output)
{
    return sizeof(n) == 4 ?
        OutputUInt32AsString(static_cast<uint32_t>(n), output):
        OutputUInt64AsString(static_cast<uint64_t>(n), output);
}

template <typename T>
class CountOutputIterator
{
public:
    CountOutputIterator() : m_count(0) {}
    CountOutputIterator& operator++()
    {
        ++m_count;
        return *this;
    }
    CountOutputIterator operator++(int)
    {
        CountOutputIterator org(*this);
        ++*this;
        return org;
    }
    CountOutputIterator& operator*()
    {
        return *this;
    }
    CountOutputIterator& operator=(T value)
    {
        return *this;
    }
    size_t Count() const
    {
        return m_count;
    }
private:
    size_t m_count;
};

size_t IntegerStringLength(int n)
{
    return OutputIntegerAsString(n, CountOutputIterator<char>()).Count();
}

/// output n to buffer as string
/// @return end position
/// @note buffer must be large enougn, and no ending '\0' append
char* WriteUInt32ToBuffer(uint32_t n, char* buffer)
{
    return OutputUInt32AsString(n, buffer);
}

/// output n to buffer as string
/// @return end position
/// @note buffer must be large enougn, and no ending '\0' append
char* WriteInt32ToBuffer(int32_t n, char* buffer)
{
    return OutputInt32AsString(n, buffer);
}

char* WriteUInt64ToBuffer(uint64_t n, char* buffer)
{
    return OutputUInt64AsString(n, buffer);
}

char* WriteInt64ToBuffer(int64_t n, char* buffer)
{
    return OutputInt64AsString(n, buffer);
}

char* WriteIntegerToBuffer(int n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

char* WriteIntegerToBuffer(unsigned int n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

char* WriteIntegerToBuffer(long n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

char* WriteIntegerToBuffer(unsigned long n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

char* WriteIntegerToBuffer(long long n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

char* WriteIntegerToBuffer(unsigned long long n, char* buffer)
{
    return OutputIntegerAsString(n, buffer);
}

void AppendIntegerToString(int n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}
void AppendIntegerToString(unsigned int n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}
void AppendIntegerToString(long n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}
void AppendIntegerToString(unsigned long n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}
void AppendIntegerToString(long long n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}
void AppendIntegerToString(unsigned long long n, std::string* str)
{
    OutputIntegerAsString(n, std::back_inserter(*str));
}

///////////////////////////////////////////////////////////////////////////
// output number to buffer as string, with ending '\0'

char* UInt32ToString(uint32_t u, char* buffer)
{
    *OutputUInt32AsString(u, buffer) = '\0';
    return buffer;
}

char* Int32ToString(int32_t i, char* buffer)
{
    *OutputInt32AsString(i, buffer) = '\0';
    return buffer;
}

char* UInt64ToString(uint64_t u64, char* buffer)
{
    *OutputUInt64AsString(u64, buffer) = '\0';
    return buffer;
}

char* Int64ToString(int64_t i, char* buffer)
{
    *OutputInt64AsString(i, buffer) = '\0';
    return buffer;
}

// -----------------------------------------------------
// interface for int to string or buffer
// Make sure the buffer is big enough
// -----------------------------------------------------
char* IntegerToString(int i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

char* IntegerToString(unsigned int i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

char* IntegerToString(long i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

char* IntegerToString(unsigned long i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

char* IntegerToString(long long i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

char* IntegerToString(unsigned long long i, char* buffer)
{
    *OutputIntegerAsString(i, buffer) = '\0';
    return buffer;
}

string IntegerToString(int i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

string IntegerToString(long i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

string IntegerToString(long long i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

string IntegerToString(unsigned int i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

string IntegerToString(unsigned long i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

string IntegerToString(unsigned long long i)
{
    char buffer[kMaxIntegerStringSize];
    return std::string(buffer, OutputIntegerAsString(i, buffer) - buffer);
}

//////////////////////////////////////////////////////////////////////////////
// human readable conversion

namespace {

template <int Base>
void GetMantissaAndShift(
    double number,
    int min_shift,
    int max_shift,
    double* mantissa,
    int* shift
    )
{
    double n = number;
    *shift = 0;

    if (isnan(n) || isinf(n))
    {
        *mantissa = n;
        return;
    }

    if (n >= 1)
    {
        while (n >= Base)
        {
            n /= Base;
            ++*shift;
        }
    }
    else
    {
        if (n > 0 || n < 0) // bypass float-equal warning
        {
            while (n < 1)
            {
                n *= Base;
                --*shift;
            }
        }
    }

    if (*shift < min_shift)
    {
        n = number;
        *shift = 0;
    }
    else if (*shift > max_shift)
    {
        n = number;
        *shift = 0;
    }

    *mantissa = n;
}

template <typename T, int Base>
std::string NumberToHumanReadableString(
    T number,
    const char* const*prefixes,
    const char* unit,
    int min_shift,
    int max_shift
    )
{
    bool neg = number < 0;
    double n = fabs(number);

    int shift;
    GetMantissaAndShift<Base>(n, min_shift, max_shift, &n, &shift);

    const char* sep = "";
    if (unit[0] == ' ')
    {
        ++unit;
        // ignore unit if it is " " and prefix is unnecessary
        if (shift != 0 || unit[0] != '\0')
            sep = " ";
    }

    char buffer[16];
    int length = sprintf(buffer, "%s%.*g%s%s", // NOLINT(runtime/printf)
                         neg ? "-": "", n < 1000 ? 3 : 4, n, sep,
                         prefixes[shift]);
    std::string result(buffer, length);
    result += unit;
    return result;
}

} // anonymous namespace

std::string FormatMeasure(double n, const char* unit)
{
    // see http://zh.wikipedia.org/wiki/%E5%9B%BD%E9%99%85%E5%8D%95%E4%BD%8D%E5%88%B6%E8%AF%8D%E5%A4%B4
    static const char* const base_prefixes[] = {
        "y", "z", "a", "f", "p", "n", "u", "m", // negative exponential
        "", "k", "M", "G", "T", "P", "E", "Z", "Y"
    };
    static const int negative_prefixes_size = 8;
    static const int prefixes_size = ARRAY_SIZE(base_prefixes);

    const char* const* prefixes = base_prefixes + negative_prefixes_size;

    return NumberToHumanReadableString<double, 1000>(
        n, prefixes, unit, -negative_prefixes_size,
        prefixes_size - negative_prefixes_size - 1);
}

std::string FormatBinaryMeasure(int64_t n, const char* unit)
{
    // see http://zh.wikipedia.org/wiki/%E4%BA%8C%E8%BF%9B%E5%88%B6%E4%B9%98%E6%95%B0%E8%AF%8D%E5%A4%B4
    static const char* const prefixes[] = {
        "", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"
    };

    return NumberToHumanReadableString<int64_t, 1024>(
        n, prefixes, unit, 0, ARRAY_SIZE(prefixes) - 1);
}

std::string ConvertByteToString(const uint64_t size) {
    std::string hight_unit;
    double min_size;
    const uint64_t kKB = 1 << 10;
    const uint64_t kMB = kKB << 10;
    const uint64_t kGB = kMB << 10;
    const uint64_t kTB = kGB << 10;
    const uint64_t kPB = kTB << 10;

    if (size == 0) {
        return "0";
    }

    if (size > kPB) {
        min_size = (1.0 * size) / kPB;
        hight_unit = "P";
    } else if (size > kTB) {
        min_size = (1.0 * size) / kTB;
        hight_unit = "T";
    } else if (size > kGB) {
        min_size = (1.0 * size) / kGB;
        hight_unit = "G";
    } else if (size > kMB) {
        min_size = (1.0 * size) / kMB;
        hight_unit = "M";
    } else if (size > kKB) {
        min_size = (1.0 * size) / kKB;
        hight_unit = "K";
    } else {
        min_size = size;
        hight_unit = "";
    }

    if ((int)min_size - min_size == 0) {
        return StringFormat("%d%s", (int)min_size, hight_unit.c_str());
    } else {
        return StringFormat("%.2f%s", min_size, hight_unit.c_str());
    }
}

void SplitString(const std::string& full,
                 const std::string& delim,
                 std::vector<std::string>* result) {
    result->clear();
    if (full.empty()) {
        return;
    }

    std::string tmp;
    std::string::size_type pos_begin = full.find_first_not_of(delim);
    std::string::size_type comma_pos = 0;

    while (pos_begin != std::string::npos) {
        comma_pos = full.find(delim, pos_begin);
        if (comma_pos != std::string::npos) {
            tmp = full.substr(pos_begin, comma_pos - pos_begin);
            pos_begin = comma_pos + delim.length();
        } else {
            tmp = full.substr(pos_begin);
            pos_begin = comma_pos;
        }

        if (!tmp.empty()) {
            result->push_back(tmp);
            tmp.clear();
        }
    }
}

void SplitStringEnd(const std::string& full, std::string* begin_part,
                    std::string* end_part, std::string delim) {
    std::string::size_type pos = full.find_last_of(delim);
    if (pos != std::string::npos && pos != 0) {
        if (end_part) {
            *end_part = full.substr(pos + 1);
        }
        if (begin_part) {
            *begin_part = full.substr(0, pos);
        }
    } else {
        if (end_part) {
            *end_part = full;
        }
    }
}

std::string ReplaceString(const std::string& str, const std::string& src,
                          const std::string& dest) {
    std::string ret;

    std::string::size_type pos_begin = 0;
    std::string::size_type pos = str.find(src);
    while (pos != std::string::npos) {
        // cout <<"replacexxx:" << pos_begin <<" " << pos <<"\n";
        ret.append(str.data() + pos_begin, pos - pos_begin);
        ret += dest;
        pos_begin = pos + src.length();
        pos = str.find(src, pos_begin);
    }
    if (pos_begin < str.length()) {
        ret.append(str.begin() + pos_begin, str.end());
    }
    return ret;
}

std::string TrimString(const std::string& str, const std::string& trim) {
    std::string::size_type pos = str.find_first_not_of(trim);
    if (pos == std::string::npos) {
        return str;
    }
    std::string::size_type pos2 = str.find_last_not_of(trim);
    if (pos2 != std::string::npos) {
        return str.substr(pos, pos2 - pos + 1);
    }
    return str.substr(pos);
}

bool StringEndsWith(const std::string& str, const std::string& sub_str) {
    if (str.length() < sub_str.length()) {
        return false;
    }
    if (str.substr(str.length() - sub_str.length()) != sub_str) {
        return false;
    }
    return true;
}

bool StringStartWith(const std::string& str, const std::string& sub_str) {
    if (str.length() < sub_str.length()) {
        return false;
    }
    if (str.substr(0, sub_str.length()) != sub_str) {
        return false;
    }
    return true;
}

char* StringAsArray(std::string* str) {
    return str->empty() ? NULL : &*str->begin();
}

std::string DebugString(const std::string& src) {
    size_t src_len = src.size();
    std::string dst;
    dst.resize(src_len << 2);

    size_t j = 0;
    for (size_t i = 0; i < src_len; i++) {
        uint8_t c = src[i];
        if (IsVisible(c)) {
            dst[j++] = c;
        } else {
            dst[j++] = '\\';
            dst[j++] = 'x';
            dst[j++] = ToHex(c >> 4);
            dst[j++] = ToHex(c & 0xF);
        }
    }

    return dst.substr(0, j);
}

std::string RoundNumberToNDecimalPlaces(double n, int d) {
    if (d < 0 || 9 < d) {
        return "(null)";
    }
    std::stringstream ss;
    ss << std::fixed;
    ss.precision(d);
    ss << n;
    return ss.str();
}

struct EditDistanceMatrix {
    EditDistanceMatrix(int row, int col)
        : matrix_((int*)malloc(sizeof(int) * row * col)),
          n_(col) {}
    int& At(int row, int col) {return matrix_[row * n_ + col];}
    ~EditDistanceMatrix() {
        free(matrix_);
        matrix_ = NULL;
    }
    int* matrix_;
private:
    int n_; // columns(row size)
    EditDistanceMatrix(const EditDistanceMatrix& m);
    EditDistanceMatrix& operator=(const EditDistanceMatrix& m);
};

static int MinOfThreeNum(int a, int b, int c) {
    int min = (a < b) ? a : b;
    min = (min < c) ? min : c;
    return min;
}

/*
        a[0] a[1] a[2] a[3] . . . a[n-1]
  b[0]
  b[1]
  b[2]        +    +
  b[3]        +    *
  .
  .
  .
  b[m-1]
*/

// https://en.wikipedia.org/wiki/Edit_distance
// https://en.wikipedia.org/wiki/Levenshtein_distance
int EditDistance(const std::string& a, const std::string& b) {
    int n = a.size();
    int m = b.size();
    if ((n == 0) || (m == 0)) {
        return (n == 0) ? m : n;
    }
    EditDistanceMatrix matrix(m, n);
    matrix.At(0, 0) = (a[0] == b[0]) ? 0 : 1;
    for (size_t i = 1; i < a.size(); i++) {
        matrix.At(0, i) = matrix.At(0, i-1) + 1;
    }
    for (size_t j = 1; j < b.size(); j++) {
        matrix.At(j, 0) = matrix.At(j-1, 0) + 1;
    }
    for (size_t j = 1; j < b.size(); j++) {
        for (size_t i = 1; i < a.size(); i++) {
            int min = MinOfThreeNum(matrix.At(j-1, i-1),
                                    matrix.At(j,   i-1),
                                    matrix.At(j-1, i));
            if (a[i] == b[j]) {
                matrix.At(j, i) = min;
            } else {
                matrix.At(j, i) = min + 1;
            }
        }
    }
    return matrix.At(m-1, n-1);
}

std::string GetLocalHostName() {
    char str[255 + 1];
    if (0 != gethostname(str, 255 + 1)) {
        return "";
    }
    std::string hostname(str);
    return hostname;
}

bool SplitPath(const std::string& path,
               std::vector<std::string>* element,
               bool* isdir) {
    if (path.empty() || path[0] != '/' || path.size() > 4096) { // MAX_PATH 4096
        return false;
    }
    element->clear();
    size_t last_pos = 0;
    for (size_t i = 1; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (last_pos + 1 < i) {
                element->push_back(path.substr(last_pos + 1, i - last_pos - 1));
            }
            last_pos = i;
        }
    }
    if (isdir) {
        *isdir = (path[path.size() - 1] == '/');
    }
    return true;
}

std::string StripBasename(const std::string &full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

} // namespace bdcommon
} // namespace bubblefs