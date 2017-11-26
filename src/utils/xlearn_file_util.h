//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*
Author: Chao Ma (mctt90@gmail.com)
This file contains facilitlies to control the file.
*/

#ifndef BUBBLEFS_UTILS_XLEARN_FILE_UTIL_H_
#define BUBBLEFS_UTILS_XLEARN_FILE_UTIL_H_

#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "platform/xlearn_logging.h"
#include "utils/xlearn_string_util.h"

namespace bubblefs {
namespace myxlearn {

//------------------------------------------------------------------------------
// Useage:
//
//    std::string filename = "test_file";
//
//    /* (1) Check whether a file exists */
//    bool bo = FileExist(filename.c_str());
//
//    /* (2) Open file : 'r' for read, 'w' for write */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
//
//    /* (3) Close file */
//    Close(file_r);
//    Close(file_w);
//
//    /* (4) Get file size */
//    uint64 size_w = GetFileSize(file_w);
//    uint64 size_r = GetFileSize(file_r);
//
//    /* (5) Get one line from file */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    std::string str_line;
//    GetLine(file_r, str_line);
//
//    /* (6) Write Binary data to disk file */
//    FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
//    int number = 999;
//    WriteDataToDisk(file_w, (char*)&number, sizeof(number));
//    Close(file_w);
//
//    /* (7) Read binary data from disk file */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    int number = 0;
//    ReadDataFromDisk(file_r, (char*)&number, sizeof(number));
//    CHECK_EQ(number, 999);
//    Close(file_r);
//
//    /* (8) Delete file from disk */
//    RemoveFile(filename.c_str());
//
//    /* (9) Print file size */
//    uint64 size = GetFileSize(filename.c_str());
//    cout << PrintSize(size) << endl;
//
//    /* (10) Write std::vector to disk file */
//    FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
//    std::vector<int> vec(10, 100);
//    WriteVectorToFile(file_w, vec);
//
//    /* (11) Read std::vector from disk file */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    std::vector<int> vec;
//    ReadVectorFromFile(file_r, vec);
//
//    /* (12) Write std::string to disk file */
//    FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
//    std::string str("apple");
//    WriteStringToFile(file_w, str);
//
//    /* (13) Read std::string from disk file */
//    FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
//    std::string str;
//    ReadStringFromFile(file_r, str);
//
//    /* (14) Generate hash value for file  */
//    uint64 hash_1 = HashFile(filename, true);   /* for one block */
//    uint64 hash_2 = HashFile(filename, false);  /* for the whole file */
//
//    /* (15) Read the whole file into in-memory buffer */
//    char *buffer = nullptr;
//    uint64 file_size = ReadFileToMemory(filename, &buffer);
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Basic operations for a file
//------------------------------------------------------------------------------

const size_t KB = 1024.0;
const size_t MB = 1024.0 * 1024.0;
const size_t GB = 1024.0 * 1024.0 * 1024.0;

/* 100 KB for one line of txt data */
static const uint32 kMaxLineSize = 100 * 1024;
/* Chunk size for hash */
static const uint32 kChunkSize = 10000000;

// Check whether the file exists.
inline bool FileExist(const char* filename) {
  if (access(filename, F_OK) != -1) {
    return true;
  }
  return false;
}

// Open file using fopen() and return the file pointer.
// model : "w" for write and "r" for read
inline FILE* OpenFileOrDie(const char* filename, const char* mode) {
  FILE* input_stream = fopen(filename, mode);
  if (input_stream == nullptr) {
    LOG(FATAL) << "Cannot open file: " << filename
               << " with mode: " << mode;
  }
  return input_stream;
}

// Close file using fclose().
inline void Close(FILE *file) {
  if (fclose(file) == -1) {
    LOG(FATAL) << "Error invoke fclose().";
  }
}

// Return the size (byte) of a target file.
inline uint64 GetFileSize(FILE* file) {
  if (fseek(file, 0L, SEEK_END) != 0) {
    LOG(FATAL) << "Error: invoke fseek().";
  }
  uint64 total_size = ftell(file);
  if (total_size == -1) {
    LOG(FATAL) << "Error: invoke ftell().";
  }
  rewind(file);  /* Return to the head */
  return total_size;
}

// Get one line of data from the file.
inline void GetLine(FILE* file, std::string& str_line) {
  CHECK_NOTNULL(file);
  static char* line = new char[kMaxLineSize];
 char* ret = fgets(line, kMaxLineSize, file);
 CHECK_NOTNULL(ret);
  int read_len = strlen(line);
  if (line[read_len - 1] != '\n') {
    LOG(FATAL) << "Encountered a too-long line.     \
                   Please check the data.";
  } else {
    line[read_len - 1] = '\0';
    // Handle the format in DOS and windows
    if (read_len > 1 && line[read_len - 2] == '\r') {
      line[read_len - 2] = '\0';
    }
  }
  str_line.assign(line);
}

// Write data from a buffer to disk file.
// Return the size (byte) of data we have written.
inline size_t WriteDataToDisk(FILE* file, const char* buf, size_t len) {
  CHECK_NOTNULL(file);
  CHECK_NOTNULL(buf);
  CHECK_GE(len, 0);
  size_t write_len = fwrite(buf, 1, len, file);
  if (write_len != len) {
    LOG(FATAL) << "Error: invoke fwrite().";
  }
  return write_len;
}

// Read data from disk file to a buffer.
// Return the data size (byte) we have read.
// If we reach the end of the file, return 0.
inline size_t ReadDataFromDisk(FILE* file, char* buf, size_t len) {
  CHECK_NOTNULL(file);
  CHECK_NOTNULL(buf);
  CHECK_GE(len, 0);
  /* Reach the end of the file */
  if (feof(file)) {
    return 0;
  }
  size_t ret = fread(buf, 1, len, file);
  if (ret > len) {
    LOG(FATAL) << "Error: invoke fread().";
  }
  return ret;
}

// Delete target file from disk.
inline void RemoveFile(const char* filename) {
  CHECK_NOTNULL(filename);
  if (remove(filename) == -1) {
    LOG(FATAL) << "Error: invoke remove().";
  }
}

// Print file size.
inline std::string PrintSize(uint64 file_size) {
  std::string res;
  if (file_size > GB) {
    SStringPrintf(&res, "%.2f GB",
            (double) file_size / GB);
  } else if (file_size > MB) {
    SStringPrintf(&res, "%.2f MB",
            (double) file_size / MB);
  } else {
    SStringPrintf(&res, "%.2f KB",
            (double) file_size / KB);
  }
  return res;
}

//------------------------------------------------------------------------------
// Serialize or Deserialize for std::vector and std::string
//------------------------------------------------------------------------------

// Write a std::vector to disk file.
template <typename T>
void WriteVectorToFile(FILE* file_ptr, const std::vector<T>& vec) {
  CHECK_NOTNULL(file_ptr);
  // We do not allow Serialize an empty vector
  CHECK(!vec.empty());
  size_t len = vec.size();
  WriteDataToDisk(file_ptr, (char*)&len, sizeof(len));
  WriteDataToDisk(file_ptr, (char*)vec.data(), sizeof(T)*len);
}

// Read a std::vector from disk file.
template <typename T>
void ReadVectorFromFile(FILE* file_ptr, std::vector<T>& vec) {
  CHECK_NOTNULL(file_ptr);
  // First, read the size of vector
  size_t len = 0;
  ReadDataFromDisk(file_ptr, (char*)(&len), sizeof(len));
  CHECK_GT(len, 0);
  vec.resize(len);
  ReadDataFromDisk(file_ptr, (char*)vec.data(), sizeof(T)*len);
}

// Write a std::string to disk file.
inline void WriteStringToFile(FILE* file_ptr, const std::string& str) {
  CHECK_NOTNULL(file_ptr);
  // We do not allow Serialize an empty string
  CHECK(!str.empty());
  size_t len = str.size();
  WriteDataToDisk(file_ptr, (char*)&len, sizeof(len));
  WriteDataToDisk(file_ptr, (char*)str.data(), len);
}

// Read a std::string from disk file.
inline void ReadStringFromFile(FILE* file_ptr, std::string& str) {
  CHECK_NOTNULL(file_ptr);
  // First, read the size of vector
  size_t len = 0;
  ReadDataFromDisk(file_ptr, (char*)(&len), sizeof(len));
  CHECK_GT(len, 0);
  str.resize(len);
  ReadDataFromDisk(file_ptr, (char*)str.data(), len);
}

//------------------------------------------------------------------------------
// Some tool functions used by Reader
//------------------------------------------------------------------------------

// Calculate the hash value of current txt file.
// If one_block == true, we just read a small chunk of data.
// If one_block == false, we read all the data from the file.
inline uint64_t HashFile(const std::string& filename, bool one_block=false) {
  std::ifstream f(filename, std::ios::ate | std::ios::binary);
  if(f.bad()) { return 0; }

  long end = (long) f.tellg();
  f.seekg(0, std::ios::beg);
  CHECK_EQ(static_cast<int>(f.tellg()), 0);

  uint64_t magic = 90359;
  for(long pos = 0; pos < end; ) {
    long next_pos = std::min(pos + kChunkSize, end);
    long size = next_pos - pos;
    std::vector<char> buffer(kChunkSize);
    f.read(buffer.data(), size);

    int i = 0;
    while(i < size - 8) {
      uint64_t x = *reinterpret_cast<uint64_t*>(buffer.data() + i);
      magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
      i += 8;
    }
    for(; i < size; i++) {
      char x = buffer[i];
      magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
    }

    pos = next_pos;
    if(one_block) { break; }
  }

  return magic;
}

// Read the whole file to a memory buffer and 
// Return size (byte) of current file.
inline uint64 ReadFileToMemory(const std::string& filename, char **buf) {
  CHECK(!filename.empty());
  FILE* file = OpenFileOrDie(filename.c_str(), "r");
  uint64 len = GetFileSize(file);
  try {
    *buf = new char[len];
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for Reader.";
  }
  uint64 read_size = fread(*buf, 1, len, file);
  CHECK_EQ(read_size, len);
  Close(file);
  return len;
}

} // namespace myxlearn
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_XLEARN_FILE_UTIL_H_