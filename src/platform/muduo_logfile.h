
// muduo/muduo/base/LogFile.h

#ifndef BUBBLEFS_UTILS_MUDUO_LOGFILE_H_
#define BUBBLEFS_UTILS_MUDUO_LOGFILE_H_

#include "platform/macros.h"
#include "platform/muduo_types.h"
#include "platform/muduo_mutex.h"

#include "boost/scoped_ptr.hpp"

namespace bubblefs {
namespace mymuduo {

namespace FileUtil
{
class AppendFile;
}

class LogFile
{
 public:
  LogFile(const string& basename,
          off_t rollSize,
          bool threadSafe = true,
          int flushInterval = 3,
          int checkEveryN = 1024);
  ~LogFile();

  void append(const char* logline, int len);
  void flush();
  bool rollFile();

 private:
  void append_unlocked(const char* logline, int len);

  static string getLogFileName(const string& basename, time_t* now);

  const string basename_;
  const off_t rollSize_;
  const int flushInterval_;
  const int checkEveryN_;

  int count_;

  boost::scoped_ptr<MutexLock> mutex_;
  time_t startOfPeriod_;
  time_t lastRoll_;
  time_t lastFlush_;
  boost::scoped_ptr<FileUtil::AppendFile> file_;

  const static int kRollPerSeconds_ = 60*60*24;
  
  DISALLOW_COPY_AND_ASSIGN(LogFile);
};

} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_LOGFILE_H_