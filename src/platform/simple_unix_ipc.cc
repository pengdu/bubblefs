// Copyright (c) 2014 Baidu, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "platform/simple_unix_ipc.h"
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>  
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "platform/base_error.h"

namespace bubblefs {
namespace mysimple {

int unix_tcp_open(char *ip, unsigned int port) {
  int sockfd = -1;
  struct sockaddr_in c_addr;

  bzero(&c_addr, sizeof(c_addr));
  c_addr.sin_family = AF_INET;
  c_addr.sin_port = htons(port);
  c_addr.sin_addr.s_addr = inet_addr(ip);

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    PRINTF_ERROR("tcp socket creation error\n");
    return -1;
  }

  if (-1 == bind(sockfd, (struct sockaddr *)&c_addr, sizeof(c_addr))) {
    PRINTF_ERROR("tcp socket bind error\n");
    return -1;
  }
  return sockfd;
}

int unix_tcp_close(int sockfd)
{
  if (sockfd)
    close(sockfd);
  return 0;
}

int unix_tcp_listen(int sockfd) {
  if (-1 == listen(sockfd, SOMAXCONN)) {
    PRINTF_ERROR("tcp socket (%d) listen error\n", sockfd);
    return -1;
  }
  return 0;
}

int unix_tcp_connect(char *ip, unsigned int port) {
  int ret = -1;
  int sockfd = -1;
  struct sockaddr_in r_addr;
  
  bzero(&r_addr, sizeof(r_addr));
  r_addr.sin_family = AF_INET;
  r_addr.sin_port = htons(port);
  r_addr.sin_addr.s_addr = inet_addr(ip);

  sockfd = socket(AF_INET, SOCK_STREAM, 0);

  ret = connect(sockfd, (struct sockaddr *)&r_addr, sizeof(r_addr));
  if (ret == 0)
    return sockfd;
  else {
    PRINTF_ERROR("tcp socket conncet %s:%d error\n", ip, port);
    close(sockfd);
    return -1;
  }
}

int unix_tcp_poll_accept(int sockfd) {
  int newsockfd = -1;
  struct sockaddr_in r_addr;
  struct pollfd ufds[1];
  int pollres = -1;
  int nfds = 0;

  ufds[nfds].fd = sockfd;
  ufds[nfds++].events = POLLIN;

  pollres = poll(ufds, nfds, 1); //1ms
  if (pollres <= 0)
    return -1;

  if (ufds[0].revents & POLLIN) {
    socklen_t addr_len = sizeof(r_addr);
    newsockfd = accept(sockfd, (struct sockaddr *)&r_addr, &addr_len);
    if (newsockfd != -1) {
      PRINTF_INFO("tcp socket (%d) poll accepted connection from %s:%d\n",
                   sockfd, inet_ntoa(r_addr.sin_addr), ntohs(r_addr.sin_port));
      return newsockfd;
    }
  }
  return -1;
}

int unix_tcp_poll_send(int sockfd, char *buf, unsigned int buf_len) {
  int ret = -1;
  int sent = 0;
  
  struct pollfd ufds[1];
  int pollres = -1;
  int nfds = 0;

  ufds[nfds].fd = sockfd;
  ufds[nfds++].events = POLLOUT;

retry:  
  pollres = poll(ufds, nfds, 10); //10ms
  if (pollres <= 0)
    return pollres;

  if (ufds[0].revents & POLLOUT) {
    ret = send(sockfd, buf + sent, buf_len - sent, 0);
    if (ret < 0) {
      PRINTF_ERROR("tcp socket (%d) poll send error, close socket\n", sockfd);
      close(sockfd);
      return -1;
    }
    sent += ret;
    if (sent < buf_len)
      goto retry;
  }

  return buf_len;
}

int unix_tcp_poll_recv(int sockfd, char *buf, unsigned int buf_len) {
  int ret = -1;
  int sent = 0;
  int count = 16; // max recv count
  struct pollfd ufds[1];
  int nfds = 0;
  int pollres = -1;

  ufds[nfds].fd = sockfd;
  ufds[nfds++].events = POLLIN;

retry:  
  pollres = poll(ufds, nfds, 1); //1ms
  if (pollres <= 0)
    return pollres; // 0 means no msgs

  if (ufds[0].revents & POLLIN) {
    if (--count <= 0) {
      PRINTF_ERROR("tcp socket (%d) poll recv loop too many times, close socket\n", sockfd);
      close(sockfd);
      return -1;
    }
    ret = recv(sockfd, buf + sent, buf_len - sent, MSG_DONTWAIT);
    if (ret < 0) {
      PRINTF_ERROR("tcp socket (%d) poll recv error, close socket\n", sockfd);
      close(sockfd);
      return -1;
    }
    if (ret == 0) {
      PRINTF_ERROR("tcp socket (%d) poll recv error, peer is shutted down, close socket\n", sockfd);
      close(sockfd);
      return -1;
    }
    sent += ret;
    if (sent < buf_len)
      goto retry;

  }

  return buf_len;
}

/*Open a message queue. If the message queue specified by "name" is not existed, 
 *it will create one. return -1 if failed*/
int unix_msgq_open_by_key(int msqq_key) {
  int msgid = -1;
  msgid = msgget((key_t)msqq_key, 00666 | IPC_CREAT);
  return msgid;
}

/*Destroy a message queue specified by msgid*/
int unix_msgq_remove_by_id(int msgq_id) {
  int ret = -1;
  ret = msgctl(msgq_id, IPC_RMID, 0);
  return ret;
}

/*Get the status of a message queue specified by msgid*/
int unix_msgq_stat_by_id(int msgq_id, struct msqid_ds *pinfo) {
  int ret = -1;
  ret = msgctl(msgq_id, IPC_STAT, pinfo);
  return ret;     
}

int unix_msgq_stat_qnum_by_id(int msgq_id) {
  int ret = -1;
  struct msqid_ds info;
  ret = msgctl(msgq_id, IPC_STAT, &info);
  if (ret != 0)
    return -1;
  else
    return info.msg_qnum;
}

int unix_msgq_send_by_id(int msgq_id, unix_msgq_buf_t *msgbuf) {
  int ret = -1;
  if (msgbuf->mtype <= 0)
    return -1;
  // Returns the length of the string, in terms of bytes.
  // Note that string objects handle bytes without knowledge of the encoding 
  // that may eventually be used to encode the characters it contains. 
  const unsigned int buf_len = msgbuf->mtext.size();
  // IPC_NOWAIT means return if queue is full.
  ret = msgsnd(msgq_id, msgbuf, buf_len, IPC_NOWAIT);
  return ret;
}

int unix_msgq_send_char_by_id(int msgq_id, void *buf, unsigned int buf_len) {
  int ret = -1;
  char *p = NULL;
  long *type = NULL;
  p = (char *)malloc(sizeof(long) + buf_len);
  if (p == NULL)
    return -1;
  memcpy(p + sizeof(long), buf, buf_len);
  type = (long int*)p;
  *type = 1; // msgtype > 0
  // IPC_NOWAIT means return if queue is full.
  ret = msgsnd(msgq_id, p, buf_len, IPC_NOWAIT);
  free(p);
  return ret;
}

int unix_msgq_recv_by_id(int msgq_id, unix_msgq_buf_t *msgbuf) {
  int ret = -1;
  if (msgbuf->mtype <= 0)
    return -1;
  const unsigned int buf_len = msgbuf->mtext.size();
  ret = msgrcv(msgq_id, msgbuf, buf_len, msgbuf->mtype, IPC_NOWAIT); 
  return ret;
}

int unix_msgq_recv_char_by_id(int msgq_id, void *buf, unsigned int buf_len) {
  int ret = -1;
  const int msgq_stat_qnum = unix_msgq_stat_qnum_by_id(msgq_id);
  if (0 >= msgq_stat_qnum) {
    // none msg
    return -1;
  }
  char *p = NULL;
  p = (char *)malloc(sizeof(long) + buf_len);
  if (p == NULL)
    return -1;
  memcpy(p + sizeof(long), buf, buf_len);
  ret = msgrcv(msgq_id, p, buf_len, 1, IPC_NOWAIT); 
  if (ret <= 0) {
    free(p);
    return -1;
  }
  memcpy(buf, p + sizeof(long), ret);
  free(p);
  return ret;
}

}  // namespace mysimple
}  // namespace bubblefs