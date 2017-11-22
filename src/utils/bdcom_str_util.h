// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/base/string_number.h
// tera/src/common/base/string_ext.h
// baidu/common/include/string_util.h

#ifndef BUBBLEFS_UTILS_BDCOM_STR_UTIL_H_
#define BUBBLEFS_UTILS_BDCOM_STR_UTIL_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include "utils/bdcom_ascii.h"

namespace bubblefs {
namespace mybdcom {

extern const std::string kNullptrString;  

// stringfy
template <typename T>
inline std::string ToString(T value) {
#if !(defined OS_ANDROID) && !(defined CYGWIN) && !(defined OS_FREEBSD)
  return std::to_string(value);
#else
  // Andorid or cygwin doesn't support all of C++11, std::to_string() being
  // one of the not supported features.
  std::ostringstream os;
  os << value;
  return os.str();
#endif
}

// ----------------------------------------------------------------------
// ascii_isalnum()
//    Check if an ASCII character is alphanumeric.  We can't use ctype's
//    isalnum() because it is affected by locale.  This function is applied
//    to identifiers in the protocol buffer language, not to natural-language
//    strings, so locale should not be taken into account.
// ascii_isdigit()
//    Like above, but only accepts digits.
// ascii_isspace()
//    Check if the character is a space character.
// ----------------------------------------------------------------------

inline bool ascii_isalnum(char c) {
  return ('a' <= c && c <= 'z') ||
         ('A' <= c && c <= 'Z') ||
         ('0' <= c && c <= '9');
}

inline bool ascii_isdigit(char c) {
  return ('0' <= c && c <= '9');
}

inline bool ascii_isspace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
      c == '\r';
}

inline bool ascii_isupper(char c) {
  return c >= 'A' && c <= 'Z';
}

inline bool ascii_islower(char c) {
  return c >= 'a' && c <= 'z';
}

inline char ascii_toupper(char c) {
  return ascii_islower(c) ? c - ('a' - 'A') : c;
}

inline char ascii_tolower(char c) {
  return ascii_isupper(c) ? c + ('a' - 'A') : c;
}

inline int hex_digit_to_int(char c) {
  /* Assume ASCII. */
  int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}  

inline bool IsVisible(char c) {
  return (c >= 0x20 && c <= 0x7E);
}

// 2 small internal utility functions, for efficient hex conversions
// and no need for snprintf, toupper etc...
// Originally from wdt/util/EncryptionUtils.cpp - for ToString(true)/DecodeHex:
static inline char ToHex(uint8_t v) {
  if (v <= 9) {
    return '0' + v;
  }
  return 'A' + v - 10;
}

// most of the code is for validation/error check
inline int FromHex(char c) {
  // toupper:
  if (c >= 'a' && c <= 'f') {
    c -= ('a' - 'A');  // aka 0x20
  }
  // validation
  if (c < '0' || (c > '9' && (c < 'A' || c > 'F'))) {
    return -1;  // invalid not 0-9A-F hex char
  }
  if (c <= '9') {
    return c - '0';
  }
  return c - 'A' + 10;
}

// ASCII-specific tolower.  The standard library's tolower is locale sensitive,
// so we don't want to use it here.
inline char ToLowerASCII(char c) {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}
inline uint16_t ToLowerASCII(uint16_t c) {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

// ASCII-specific toupper.  The standard library's toupper is locale sensitive,
// so we don't want to use it here.
inline char ToUpperASCII(char c) {
  return (c >= 'a' && c <= 'z') ? (c + ('A' - 'a')) : c;
}
inline uint16_t ToUpperASCII(uint16_t c) {
  return (c >= 'a' && c <= 'z') ? (c + ('A' - 'a')) : c;
}

// Function objects to aid in comparing/searching strings.
// butil::CaseInsensitiveCompare<typename STR::value_type>()
template<typename Char> struct CaseInsensitiveCompare {
 public:
  bool operator()(Char x, Char y) const {
    // TODO(darin): Do we really want to do locale sensitive comparisons here?
    // See http://crbug.com/24917
    return tolower(x) == tolower(y);
  }
};

template<typename Char> struct CaseInsensitiveCompareASCII {
 public:
  bool operator()(Char x, Char y) const {
    return ToLowerASCII(x) == ToLowerASCII(y);
  }
};

// Determines the type of ASCII character, independent of locale (the C
// library versions will change based on locale).
template <typename Char>
inline bool IsAsciiWhitespace(Char c) {
  return c == ' ' || c == '\r' || c == '\n' || c == '\t';
}
template <typename Char>
inline bool IsAsciiAlpha(Char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
template <typename Char>
inline bool IsAsciiUpper(Char c) {
  return c >= 'A' && c <= 'Z';
}
template <typename Char>
inline bool IsAsciiLower(Char c) {
  return c >= 'a' && c <= 'z';
}
template <typename Char>
inline bool IsAsciiDigit(Char c) {
  return c >= '0' && c <= '9';
}

template <typename Char>
inline bool IsHexDigit(Char c) {
  return (c >= '0' && c <= '9') ||
         (c >= 'A' && c <= 'F') ||
         (c >= 'a' && c <= 'f');
}

template <typename Char>
inline Char HexDigitToInt(Char c) {
  if (!IsHexDigit(c))
    return 0;
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  return 0;
}

// Append a human-readable time in micros.
int AppendHumanMicros(uint64_t micros, char* output, int len,
                      bool fixed_format);
// Append a human-readable size in bytes
int AppendHumanBytes(uint64_t bytes, char* output, int len);

std::string HumanReadableString(int64_t num);
// Return a human-readable version of num.
// for num >= 10.000, prints "xxK"
// for num >= 10.000.000, prints "xxM"
// for num >= 10.000.000.000, prints "xxG"
std::string NumberToHumanString(int64_t num);
// Return a human-readable version of bytes
// ex: 1048576 -> 1.00 GB
std::string BytesToHumanString(uint64_t bytes);

std::string NumberInt64ToString(int64_t num);
std::string NumberIntToString(int num);
std::string NumberUint32ToString(uint32_t num);
std::string NumberDoubleToString(double num);
std::string RoundNumberToNDecimalPlaces(double n, int d);

bool SerializeIntVector(const std::vector<int>& vec, std::string* value);

bool ParseBoolean(const std::string& type, const std::string& value);
uint32_t ParseUint32(const std::string& value);
uint64_t ParseUint64(const std::string& value);
int ParseInt(const std::string& value);
double ParseDouble(const std::string& value);
size_t ParseSizeT(const std::string& value);
std::vector<int> ParseVectorInt(const std::string& value);

void ToUpper(std::string* str);
void ToLower(std::string* str);
std::string Hex2Str(const char* _str, unsigned int _len);
std::string Str2Hex(const char* _str, unsigned int _len);
bool Hex2Bin(const char* hex_str, std::string* bin_str);
bool Bin2Hex(const char* bin_str, std::string* hex_str);

bool StartsWith(const std::string& str, const std::string& prefix);
bool EndsWith(const std::string& str, const std::string& suffix);

// ----------------------------------------------------------------------
// HasSuffixString()
//    Return true if str ends in suffix.
// ----------------------------------------------------------------------
inline bool HasSuffixString(const std::string& str,
                            const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
// ----------------------------------------------------------------------
// HasPrefixString()
//    Check if a string begins with a given prefix.
// ----------------------------------------------------------------------
inline bool HasPrefixString(const std::string& str,
                            const std::string& prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

bool StripSuffix(std::string* str, const std::string& suffix);
bool StripPrefix(std::string* str, const std::string& prefix);
// Obtains the base name from a full path.
std::string StripBasename(const std::string& full_path);

std::string& Ltrim(std::string& str); // NOLINT
std::string& Rtrim(std::string& str); // NOLINT
std::string& Trim(std::string& str); // NOLINT
void Trim(std::vector<std::string>* str_list);

void SplitStringChar(const std::string& str,
                     char delim,
                     std::vector<std::string>* result);
// Splits |line| into key value pairs according to the given delimiters and
// removes whitespace leading each key and trailing each value. Returns true
// only if each pair has a non-empty key and value. |key_value_pairs| will
// include ("","") pairs for entries without |key_value_delimiter|.
typedef std::vector<std::pair<std::string, std::string> > StringPairs;
bool SplitStringIntoKeyValuePairs(const std::string& line,
                                  char key_value_delimiter,
                                  char key_value_pair_delimiter,
                                  StringPairs* key_value_pairs);


int EditDistance(const std::string& a, const std::string& b);

std::string GetLocalHostName();

std::string DebugString(const std::string& src);
  
/// -----------------------------------------------------------------------
/// @brief Parse the string from the first position. stop when error occurs.
/// @return true if part of the string can be converted to a valid number
/// @return false if the number exceeds limit or nothing is parsed.
/// @param str the string to parse
/// @param value store the parsed number if success
/// @param endptr not null, *endptr stores the address of the first invalid char
/// !!! If no invalid char is allowed, use function below: StringToNumber()
/// @param base specify the base to be used for the conversion
/// -----------------------------------------------------------------------

bool ParseNumber(const char* str, signed char* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, unsigned char* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, short* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, unsigned short* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, int* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, unsigned int* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, long* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, unsigned long* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, long long* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, unsigned long long* value, char** endptr, int base = 0);
bool ParseNumber(const char* str, float* value, char** endptr);
bool ParseNumber(const char* str, double* value, char** endptr);
bool ParseNumber(const char* str, long double* value, char** endptr);

/// ---------------------------------------------------------------
/// @brief interface for parsing string to number
/// ---------------------------------------------------------------
template <typename Type>
bool ParseNumber(const std::string& str, Type* value, char** endptr, int base)
{
    return ParseNumber(str.c_str(), value, endptr, base);
}

template <typename Type>
bool ParseNumber(const std::string& str, Type* value, char** endptr)
{
    return ParseNumber(str.c_str(), value, endptr);
}

/// ---------------------------------------------------------------
/// @brief interface for string to number
/// @return true if total string is successfully parsed.
/// ---------------------------------------------------------------
template <typename Type>
bool StringToNumber(const std::string& str, Type* value, int base)
{
//     STATIC_ASSERT(TypeTraits::IsInteger<Type>::Value, "Type must be integral type");
    char* endptr;
    bool ret = ParseNumber(str.c_str(), value, &endptr, base);
    return (ret && *endptr == '\0');
}

template <typename Type>
bool StringToNumber(const char* str, Type* value, int base)
{
//     STATIC_ASSERT(TypeTraits::IsInteger<Type>::Value, "Type must be integral type");
    char* endptr;
    bool ret = ParseNumber(str, value, &endptr, base);
    return (ret && *endptr == '\0');
}

template <typename Type>
bool StringToNumber(const std::string& str, Type* value)
{
    char* endptr;
    bool ret = ParseNumber(str.c_str(), value, &endptr);
    return (ret && *endptr == '\0');
}

template <typename Type>
bool StringToNumber(const char* str, Type* value)
{
    char* endptr;
    bool ret = ParseNumber(str, value, &endptr);
    return (ret && *endptr == '\0');
}

/// ---------------------------------------------------------------
/// @brief converting numbers  to buffer, buffer size should be big enough
/// ---------------------------------------------------------------
const int kMaxIntegerStringSize = 32;
const int kMaxDoubleStringSize = 32;
const int kMaxFloatStringSize = 24;
const int kMaxIntStringSize = kMaxIntegerStringSize;

/// @brief judge a number if it's nan
inline bool IsNaN(double value)
{
    return !(value > value) && !(value <= value);
}

/// @brief write number to buffer as string
/// @return end of result
/// @note without '\0' appended
/// private functions for common library, don't use them in your code
char* WriteDoubleToBuffer(double n, char* buffer);
char* WriteFloatToBuffer(float n, char* buffer);
char* WriteInt32ToBuffer(int32_t i, char* buffer);
char* WriteUInt32ToBuffer(uint32_t u, char* buffer);
char* WriteInt64ToBuffer(int64_t i, char* buffer);
char* WriteUInt64ToBuffer(uint64_t u64, char* buffer);

char* WriteIntegerToBuffer(int n, char* buffer);std::string DebugString(const std::string& src);

std::string HumanReadableString(int64_t num);

std::string RoundNumberToNDecimalPlaces(double n, int d);
int EditDistance(const std::string& a, const std::string& b);

std::string GetLocalHostName();

// Obtains the base name from a full path.
std::string StripBasename(const std::string& full_path);
char* WriteIntegerToBuffer(unsigned int n, char* buffer);
char* WriteIntegerToBuffer(long n, char* buffer);
char* WriteIntegerToBuffer(unsigned long n, char* buffer);
char* WriteIntegerToBuffer(long long n, char* buffer);
char* WriteIntegerToBuffer(unsigned long long n, char* buffer);

void AppendIntegerToString(int n, std::string* str);
void AppendIntegerToString(unsigned int n, std::string* str);
void AppendIntegerToString(long n, std::string* str);
void AppendIntegerToString(unsigned long n, std::string* str);
void AppendIntegerToString(long long n, std::string* str);
void AppendIntegerToString(unsigned long long n, std::string* str);

/// @brief convert number to hex string
/// buffer size should be at least [2 * sizeof(value) + 1]
char* WriteHexUInt16ToBuffer(uint16_t value, char* buffer);
char* WriteHexUInt32ToBuffer(uint32_t value, char* buffer);
char* WriteHexUInt64ToBuffer(uint64_t value, char* buffer);

/// @brief write number to buffer as string
/// @return start of buffer
/// @note with '\0' appended
char* DoubleToString(double n, char* buffer);
char* FloatToString(float n, char* buffer);
char* Int32ToString(int32_t i, char* buffer);
char* UInt32ToString(uint32_t u, char* buffer);
char* Int64ToString(int64_t i, char* buffer);
char* UInt64ToString(uint64_t u64, char* buffer);
char* IntegerToString(int n, char* buffer);
char* IntegerToString(unsigned int n, char* buffer);
char* IntegerToString(long n, char* buffer);
char* IntegerToString(unsigned long n, char* buffer);
char* IntegerToString(long long n, char* buffer);
char* IntegerToString(unsigned long long n, char* buffer);
char* UInt16ToHexString(uint16_t value, char* buffer);
char* UInt32ToHexString(uint32_t value, char* buffer);
char* UInt64ToHexString(uint64_t value, char* buffer);

/// @brief convert float number to string
std::string FloatToString(float n);

/// @brief convert float number to string
std::string DoubleToString(double n);

/// ---------------------------------------------------------------
/// @brief convert number to string, not so efficient but more convenient
/// ---------------------------------------------------------------
std::string IntegerToString(int n);
std::string IntegerToString(unsigned int n);
std::string IntegerToString(long n);
std::string IntegerToString(unsigned long n);
std::string IntegerToString(long long n);
std::string IntegerToString(unsigned long long n);

/// @brief convert number to hex string, not so efficient but more convenient
std::string UInt16ToHexString(uint16_t n);
std::string UInt32ToHexString(uint32_t n);
std::string UInt64ToHexString(uint64_t n);

/// @brief convert numeric type object to string
/// for generic programming code
inline std::string NumberToString(double n) { return DoubleToString(n); }
inline std::string NumberToString(float n) { return FloatToString(n); }
inline std::string NumberToString(int n) { return IntegerToString(n); }
inline std::string NumberToString(unsigned int n) { return IntegerToString(n); }
inline std::string NumberToString(long n) { return IntegerToString(n); }
inline std::string NumberToString(unsigned long n) { return IntegerToString(n); }
inline std::string NumberToString(long long n) { return IntegerToString(n); }
inline std::string NumberToString(unsigned long long n) { return IntegerToString(n); }

///////////////////////////////////////////////////////////////////////////////
// convert number to human readable string

/// Convert decimal number to human readable string, based on 1000.
/// @param n the number to be converted
/// @param unit the unit of the number, such as "m/s", "Hz", etc.
///             the unit can have an optional space(" ") prefix,
///             such as " bps", and then 10000 will be convert to "10 kbps"
/// @note this function support float number and negative number, keep 3
/// significant digits.
std::string FormatMeasure(double n, const char* unit = "");

/// Convert number to human readable string, based on 1024.
/// @param n the number to be converted
/// @param unit the unit of the number, such as "m/s"
///             the unit can have an optional space(" ") prefix,
///             such as " B", and then 10240 will be convert to "10 KiB"
/// this function support only integral value, keep 3 significant digits.
std::string FormatBinaryMeasure(int64_t n, const char* unit = "");

std::string ConvertByteToString(const uint64_t size);

void SplitString(const std::string& full,
                 const std::string& delim,
                 std::vector<std::string>* result);

void SplitStringEnd(const std::string& full,
                    std::string* begin_part,
                    std::string* end_part,
                    std::string delim = ".");

std::string ReplaceString(const std::string& str,
                          const std::string& src,
                          const std::string& dest);


std::string TrimString(const std::string& str,
                       const std::string& trim = " ");

bool StringEndsWith(const std::string& str,
                    const std::string& sub_str);

bool StringStartWith(const std::string& str,
                    const std::string& sub_str);

char* StringAsArray(std::string* str);

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_STR_UTIL_H_