// Copyright (c) 2011 Baidu, Inc.
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

// brpc/src/butil/endpoint.h

#ifndef BUBBLEFS_PLATFORM_BRPC_ENDPOINT_H_
#define BUBBLEFS_PLATFORM_BRPC_ENDPOINT_H_

// Wrappers of IP and port.

#include <netinet/in.h>                          // in_addr
#include <iostream>                              // std::ostream
#include "utils/hash.h"
#include "utils/hash_tables.h"

namespace bubblefs {
 
namespace mybrpc {

// Type of an IP address
typedef struct in_addr ip_t;

constexpr ip_t IP_ANY = { INADDR_ANY };
constexpr ip_t IP_NONE = { INADDR_NONE };

// Convert |ip| to an integral
inline in_addr_t ip2int(ip_t ip) { return ip.s_addr; }

// Convert integral |ip_value| to an IP
inline ip_t int2ip(in_addr_t ip_value) {
    const ip_t ip = { ip_value };
    return ip;
}

// Convert string `ip_str' to ip_t *ip.
// `ip_str' is in IPv4 dotted-quad format: `127.0.0.1', `10.23.249.73' ...
// Returns 0 on success, -1 otherwise.
int str2ip(const char* ip_str, ip_t* ip);

struct IPStr {
    const char* c_str() const { return _buf; }
    char _buf[INET_ADDRSTRLEN];
};

// Represent ports inside [min_port, max_port]
struct PortRange {
    int min_port;
    int max_port;

    PortRange(int min_port2, int max_port2)
        : min_port(min_port2), max_port(max_port2) {
    }
};

// Convert IP to c-style string. Notice that you can serialize ip_t to
// std::ostream directly. Use this function when you don't have streaming log.
// Example: printf("ip=%s\n", ip2str(some_ip).c_str());
IPStr ip2str(ip_t ip);

// Convert `hostname' to ip_t *ip. If `hostname' is NULL, use hostname
// of this machine.
// `hostname' is typically in this form: `tc-cm-et21.tc' `db-cos-dev.db01' ...
// Returns 0 on success, -1 otherwise.
int hostname2ip(const char* hostname, ip_t* ip);

// Convert `ip' to `hostname'.
// Returns 0 on success, -1 otherwise and errno is set.
int ip2hostname(ip_t ip, char* hostname, size_t hostname_len);
int ip2hostname(ip_t ip, std::string* hostname);

// ipv4 + port
struct EndPoint {
    EndPoint() : ip(IP_ANY), port(0) {}
    EndPoint(ip_t ip2, int port2) : ip(ip2), port(port2) {}
    explicit EndPoint(const sockaddr_in& in)
        : ip(in.sin_addr), port(ntohs(in.sin_port)) {}
    
    ip_t ip;
    int port;
};

struct EndPointStr {
    const char* c_str() const { return _buf; }
    char _buf[INET_ADDRSTRLEN + 16];
};

// Convert EndPoint to c-style string. Notice that you can serialize 
// EndPoint to std::ostream directly. Use this function when you don't 
// have streaming log.
// Example: printf("point=%s\n", endpoint2str(point).c_str());
EndPointStr endpoint2str(const EndPoint&);

// Convert string `ip_and_port_str' to a EndPoint *point.
// Returns 0 on success, -1 otherwise.
int str2endpoint(const char* ip_and_port_str, EndPoint* point);
int str2endpoint(const char* ip_str, int port, EndPoint* point);

// Convert `hostname_and_port_str' to a EndPoint *point.
// Returns 0 on success, -1 otherwise.
int hostname2endpoint(const char* ip_and_port_str, EndPoint* point);
int hostname2endpoint(const char* name_str, int port, EndPoint* point);

// Convert `endpoint' to `hostname'.
// Returns 0 on success, -1 otherwise and errno is set.
int endpoint2hostname(const EndPoint& point, char* hostname, size_t hostname_len);
int endpoint2hostname(const EndPoint& point, std::string* host);

// Get the local end of a socket connection
int get_local_side(int fd, EndPoint *out);

// Create a TCP socket and connect it to `server'. Write port of this side
// into `self_port' if it's not NULL.
// Returns the socket descriptor, -1 otherwise and errno is set.
int tcp_connect(EndPoint server, int* self_port);

// Create and listen to a TCP socket bound with `ip_and_port'. If `reuse_addr'
// is true, ports in TIME_WAIT will be bound as well.
// Returns the socket descriptor, -1 otherwise and errno is set.
int tcp_listen(EndPoint ip_and_port, bool reuse_addr);

// Since ip_t is defined from in_addr which is globally defined, due to ADL
// we have to put overloaded operators globally as well.
inline bool operator<(ip_t lhs, ip_t rhs) {
    return ip2int(lhs) < ip2int(rhs);
}
inline bool operator>(ip_t lhs, ip_t rhs) {
    return rhs < lhs;
}
inline bool operator>=(ip_t lhs, ip_t rhs) {
    return !(lhs < rhs);
}
inline bool operator<=(ip_t lhs, ip_t rhs) {
    return !(rhs < lhs); 
}
inline bool operator==(ip_t lhs, ip_t rhs) {
    return ip2int(lhs) == ip2int(rhs);
}
inline bool operator!=(ip_t lhs, ip_t rhs) {
    return !(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const IPStr& ip_str) {
    return os << ip_str.c_str();
}
inline std::ostream& operator<<(std::ostream& os, ip_t ip) {
    return os << ip2str(ip);
}
  
// Overload operators for EndPoint in the same namespace due to ADL(argument-depentment lookup / Koenig Lookup).
inline bool operator<(EndPoint p1, EndPoint p2) {
    return (p1.ip != p2.ip) ? (p1.ip < p2.ip) : (p1.port < p2.port);
}
inline bool operator>(EndPoint p1, EndPoint p2) {
    return p2 < p1;
}
inline bool operator<=(EndPoint p1, EndPoint p2) { 
    return !(p2 < p1); 
}
inline bool operator>=(EndPoint p1, EndPoint p2) { 
    return !(p1 < p2); 
}
inline bool operator==(EndPoint p1, EndPoint p2) {
    return p1.ip == p2.ip && p1.port == p2.port;
}
inline bool operator!=(EndPoint p1, EndPoint p2) {
    return !(p1 == p2);
}

inline std::ostream& operator<<(std::ostream& os, const EndPoint& ep) {
    return os << ep.ip << ':' << ep.port;
}
inline std::ostream& operator<<(std::ostream& os, const EndPointStr& ep_str) {
    return os << ep_str.c_str();
}

}  // namespace mybrpc

template <>
struct hash<mybrpc::EndPoint> {
    std::size_t operator()(const mybrpc::EndPoint& ep) const {
        return HashPair(mybrpc::ip2int(ep.ip), ep.port);
    }
};

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_BRPC_ENDPOINT_H_