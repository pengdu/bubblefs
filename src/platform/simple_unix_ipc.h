
#ifndef BUBBLEFS_PLATFORM_SIMPLE_UNIX_IPC_H_
#define BUBBLEFS_PLATFORM_SIMPLE_UNIX_IPC_H_

#include <sys/epoll.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

namespace bubblefs {
namespace mysimple {

/*Open a socket bond with the specific IP address and port number. Return fd of the opened socket if success, otherwise -1.*/
int unix_tcp_open(char *ip, unsigned int port);
/*Close an opened socket*/
int unix_tcp_close(int sockfd);
/*Listen on the specific socket. Return 0 if success. Otherwise -1*/
int unix_tcp_listen(int sockfd);
/*Connect to the specific remote ip/port. Return sockfd if success. Otherwise return -1.*/
int unix_tcp_connect(char *ip, unsigned int port);
/*Non-blocking accept method. Wait 1ms to accepte incoming connections. Return new sockfd if success. Otherwise return 0 indicating no connection accepted.*/
int unix_tcp_poll_accept(int sockfd);
/*Non-blocking send method. Return whole len if no error occurs. Otherwise return <=0 value*/
int unix_tcp_poll_send(int sockfd, char *buf, unsigned int buf_len);
/*Non-blocking receive method. Return whole len if no error occurs. Otherwise return <=0 */
int unix_tcp_poll_recv(int sockfd, char *buf, unsigned int buf_len);

struct unix_msgq_buf_t
{  
    long mtype;  
    std::string mtext;
}; 

/*Open a message queue. If the message queue specified by "name" is not existed, 
 *it will create one. */
int unix_msgq_open_by_key(int msqq_key);
/*Destroy a message queue specified by msgid*/
int unix_msgq_remove_by_id(int msgq_id);
/*Get the status of a message queue specified by msgid*/
int unix_msgq_stat_by_id(int msgq_id, struct msqid_ds *pinfo);
/*Return the message count of specified queue*/
int unix_msgq_stat_qnum_by_id(int msgq_id);
/*Both send() and receive() are non-blocking*/
int unix_msgq_send_by_id(int msgq_id, unix_msgq_buf_t *msgbuf);
int unix_msgq_send_char_by_id(int msgq_id, void *buf, unsigned int buf_len);
int unix_msgq_recv_by_id(int msgq_id, unix_msgq_buf_t *msgbuf);
int unix_msgq_recv_char_by_id(int msgq_id, void *buf, unsigned int buf_len);

}  // namespace mysimple
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_SIMPLE_UNIX_IPC_H_