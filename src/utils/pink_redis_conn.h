// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/redis_conn.h

#ifndef BUBBLEFS_UTILS_PINK_REDIS_CONN_H_
#define BUBBLEFS_UTILS_PINK_REDIS_CONN_H_

#include <map>
#include <string>
#include <vector>
#include "utils/pink_conn.h"
#include "utils/pink_define.h"

namespace bubblefs {
namespace mypink {

// Usage like:
// class MyHTTPConn : public pink::SimpleHTTPConn {
//  public:
//   MyHTTPConn(const int fd, const std::string& ip_port, ServerThread* worker) :
//    SimpleHTTPConn(fd, ip_port, worker) {
//   }
//   virtual void DealMessage(const pink::Request* req, pink::Response* res) {
//     for (auto& h : req->headers) {
//       std::cout << "   + " << h.first << ":" << h.second << std::endl;
//     }
//     std::cout << " + query_params: " << std::endl;
//     for (auto& q : req->query_params) {
//       std::cout << "   + " << q.first << ":" << q.second << std::endl;
//     }
//     std::cout << " + post_params: " << std::endl;
//     for (auto& q : req->post_params) {
//       std::cout << "   + " << q.first << ":" << q.second << std::endl;
//     }
//     res->SetStatusCode(200);
//     res->SetBody("china");
//    }
// };
//
// class MyConnFactory : public ConnFactory {
//  public:
//   virtual PinkConn *NewPinkConn(int connfd, const std::string &ip_port,
//                                 ServerThread *thread,
//                                 void* worker_specific_data) const {
//     return new MyHTTPConn(connfd, ip_port, thread);
//   }
// };
//
// SignalSetup();
// MyConnFactory conn_factory;
// ServerThread *st = NewDispatchThread(port, 4, &conn_factory, 1000);
// st->StartThread();
// while (running.load()) { sleep(1); }
// st->StopThread();
// DeleteServerThread(&st);  
  
typedef std::vector<std::string> RedisCmdArgsType;

class RedisConn: public PinkConn {
 public:
  RedisConn(const int fd, const std::string &ip_port, ServerThread *thread);
  virtual ~RedisConn();
  void ResetClient();

  bool ExpandWbufTo(uint32_t new_size);
  uint32_t wbuf_size_;

  virtual ReadStatus GetRequest();
  virtual WriteStatus SendReply();


  ConnStatus connStatus_;

 protected:
  char* wbuf_;
  uint32_t wbuf_len_;
  RedisCmdArgsType argv_;
  virtual int DealMessage() = 0;

 private:
  ReadStatus ProcessInputBuffer();
  ReadStatus ProcessMultibulkBuffer();
  ReadStatus ProcessInlineBuffer();
  int32_t FindNextSeparators();
  int32_t GetNextNum(int32_t pos, int32_t *value);
  int32_t last_read_pos_;
  int32_t next_parse_pos_;
  int32_t req_type_;
  int32_t multibulk_len_;
  int32_t bulk_len_;
  bool is_find_sep_;
  bool is_overtake_;

  /*
   * The Variable need by read the buf,
   * We allocate the memory when we start the server
   */
  char* rbuf_;
  uint32_t wbuf_pos_;
};

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_REDIS_CONN_H_