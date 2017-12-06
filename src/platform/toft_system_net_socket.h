// Copyright (c) 2011, The Toft Authors. All rights reserved.
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/net/socket.h
// toft/system/net/socket.inl

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_NET_SOCKET_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_NET_SOCKET_H_

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "platform/toft_system_net_os_socket.h"
#include "platform/toft_system_net_socket_address.h"

namespace bubblefs {
namespace mytoft {

/// socket error exception class
class SocketError : public std::runtime_error
{
public:
    SocketError(const char* info = "socket error", int error = SocketGetLastError());
    int Code() const { return m_error; }
private:
    int m_error;
};

/// Abstract socket base class
class Socket
{
public:
    static const SOCKET kInvalidHandle = INVALID_SOCKET_HANDLE;

protected:
    Socket();
    explicit Socket(SOCKET handle);
    bool Create(int af, int type, int protocol = 0);

public:
    virtual ~Socket();
    SOCKET Handle() const;
    bool IsValid() const;

    /// Attach a socket handle to this object
    void Attach(SOCKET handle);

    /// Detach socket handle from this object
    SOCKET Detach();

    bool Close();

    /// Set the FD_CLOEXEC flag of desc if value is nonzero,
    /// or clear the flag if value is 0.
    /// Return 0 on success, or -1 on error with errno set.
    static bool SetCloexec(int desc, bool value);
    bool SetCloexec(bool value = true);

    bool GetOption(int level, int name, void* value, socklen_t* length) const;
    bool SetOption(int level, int name, const void* value, socklen_t length);

    template <typename T>
    bool GetOption(int level, int name, T* value) const
    {
        socklen_t length = sizeof(value);
        return GetOption(level, name, value, &length);
    }

    template <typename T>
    bool SetOption(int level, int name, const T& value)
    {
        socklen_t length = sizeof(value);
        return SetOption(level, name, &value, length);
    }

    /// Get socket option with difference type
    template <typename Type, typename InternalType>
    bool GetOption(int level, int name, Type* value) const
    {
        InternalType internal_value;
        bool result = GetOption(level, name, &internal_value);
        *value = static_cast<Type>(internal_value);
        return result;
    }

    /// Set socket option with difference type
    template <typename Type, typename InternalType>
    bool SetOption(int level, int name, const Type& value)
    {
        return SetOption(level, name, static_cast<InternalType>(value));
    }

    bool GetOption(int level, int name, bool* value) const;
    bool SetOption(int level, int name, const bool& value);

    bool GetError(int* error);
    bool GetType(int* type) const;

    bool GetSendBufferSize(size_t* size) const;
    bool SetSendBufferSize(size_t size);
    bool GetReceiveBufferSize(size_t* size) const;
    bool SetReceiveBufferSize(size_t size);

#ifdef _WIN32
    bool SetSendTimeout(int seconds, int msec = 0);
    bool SetSendTimeout(const timeval& tv);
    bool SetReceiveTimeout(int seconds, int msec = 0);
    bool SetReceiveTimeout(const timeval& tv);
#else
    bool SetSendTimeout(const timeval& tv);
    bool SetSendTimeout(int seconds, int msec = 0);
    bool SetReceiveTimeout(const timeval& tv);
    bool SetReceiveTimeout(int seconds, int msec = 0);
#endif
    bool Ioctl(int cmd, int* value);

    bool SetBlocking(bool value = true);
    bool GetBlocking(bool* value);

    bool Bind(const SocketAddress& address);

    bool GetLocalAddress(SocketAddress* address) const;

    bool GetPeerAddress(SocketAddress* address) const;

    bool GetReuseAddress(bool* value);
    bool SetReuseAddress(bool value = true);

    bool SetLinger(bool onoff = true, int timeout = 0);
    bool SetKeepAlive(bool onoff = true);

    bool GetKeepAlive(bool* onoff);

#if __unix__
    bool SetTcpKeepAliveOption(int idle, int interval, int count);
#endif
    bool SetTcpNoDelay(bool onoff = true);
    bool GetTcpNoDelay(bool* onoff);

    bool WaitReadable(struct timeval* tv = NULL, bool restart = true);
    bool WaitWriteable(struct timeval* tv = NULL, bool restart = true);

    bool IsReadable();
    bool IsWriteable();

public:
    static int GetLastError();
    static std::string GetErrorString(int error);
    static std::string GetLastErrorString();

protected:
    bool CheckError(int result, const char* info = "socket") const;
    void ReportError(const char* info) const;
    static void SetLastError(int error);
    static void VerifyHandle(int fd);
    static bool IsInterruptedAndRestart(bool restart);

private:
    Socket(const Socket&);
    Socket& operator=(const Socket&);

private:
    SOCKET m_handle;
};

/// Listen streaming connections from client
class ListenerSocket : public Socket
{
public:
    ListenerSocket();
    ListenerSocket(int af, int type, int protocol);
    ListenerSocket(const SocketAddress& address, int type = SOCK_STREAM);
    using Socket::Create;
    bool Listen(int backlog = SOMAXCONN);

    bool Accept(Socket* socket, bool auto_restart = true);
    bool Accept(Socket* socket, SocketAddress* address, bool auto_restart = true);
};

/// Abstract data transfer socket
class DataSocket : public Socket
{
protected:
    DataSocket() {}

public:
    bool Connect(const SocketAddress& address);

    /// Connect with timeout
    bool Connect(const SocketAddress& address, int64_t timeout_ms);

    /// Send data
    bool Send(
        const void* buffer,
        size_t buffer_size,
        size_t* sent_length,
        int flags = 0,
        bool auto_restart = true
    );

    /// @return Whether received any data or connect close by peer.
    /// @note If connection is closed by peer, return true and received_size
    ///       is set to 0.
    bool Receive(
        void* buffer,
        size_t buffer_size,
        size_t* received_size,
        int flags = 0,
        bool auto_restart = true
    );

    /// Receive with timeout
    /// @return false if error or timeout, check Socket::GetLastError() for details
    bool Receive(
        void* buffer,
        size_t buffer_size,
        size_t* received_size,
        timeval* timeout,
        int flags = 0,
        bool auto_restart = true
    );
};

/// Stream socket, for example TCP socket
class StreamSocket : public DataSocket
{
public:
    StreamSocket() {}
    explicit StreamSocket(int af, int protocol);

    /// Create a stream socket
    bool Create(sa_family_t af = AF_INET, int protocol = 0);

    /// Shutdown connection
    bool Shutdown();

    /// Shutdown connection sending
    bool ShutdownSend();

    /// Shutdown connection receiving
    bool ShutdownReceive();

    /// @brief Receive data of all expected length
    /// @return Whether received all expacted data
    /// @note If return false, data may also received and received will be
    ///       greater than 0
    bool ReceiveAll(
        void *buffer,
        size_t buffer_size,
        size_t* received_size,
        int flags = 0,
        bool auto_restart = true
    );

    /// @brief Same as upper, expect without the out param 'received_size'
    bool ReceiveAll(
        void *buffer,
        size_t buffer_size,
        int flags = 0,
        bool auto_restart = true
    );

    /// @brief Receive all length, with timeout and out param received_size
    /// @return Whether received all data
    bool ReceiveAll(
        void *buffer,
        size_t buffer_size,
        size_t* received_size,
        timeval* timeout,
        int flags = 0,
        bool auto_restart = true
    );

    /// @brief Receive all length, with timeout
    bool ReceiveAll(
        void *buffer,
        size_t buffer_size,
        timeval* timeout,
        int flags = 0,
        bool auto_restart = true
    );

    /// Receive a line to buffer, include terminal '\n'
    /// @return Whether received a complete line
    bool ReceiveLine(
        void* buffer,
        size_t buffer_size,
        size_t* received_size,
        size_t max_peek_size = 80
    );

    /// Receive a line to string, include terminal '\n'
    /// @return Whether received a complete line
    bool ReceiveLine(std::string* str, size_t peek_size = 80);

    /// Send all data of buffer to socket
    /// @return Whether all data sent
    bool SendAll(
        const void* buffer,
        size_t buffer_size,
        size_t* sent_size,
        int flags = 0,
        bool auto_restart = true
    );

    /// @brief Send all buffer to socket
    /// @return true if all data sent, flase for any other case
    /// @note If false returned, partial data may alse be sent
    bool SendAll(
        const void* buffer,
        size_t buffer_size,
        int flags = 0,
        bool auto_restart = true
    );

    /// @brief Send all buffer to socket with timeout
    /// @return true if all data sent, flase for any other case
    /// @note If false returned, partial data may alse be sent
    bool SendAll(
        const void* buffer,
        size_t buffer_size,
        size_t* sent_size,
        timeval* tv,
        int flags = 0,
        bool auto_restart = true
    );
};

/// Represent a Datagram socket, such as UDP socket
class DatagramSocket : public DataSocket
{
public:
    /// Construct object and create a socket
    DatagramSocket(int af, int protocol = 0);

    /// Construct an empty object
    DatagramSocket() {}

    /// Create the system socket
    bool Create(int af = AF_INET, int protocol = 0);

    /// Send data with specified address
    bool SendTo(
        const void* buffer,
        size_t buffer_size,
        const SocketAddress& address,
        size_t* sent_size
    );

    /// Receive data and obtain remote address
    bool ReceiveFrom(
        void* buffer,
        size_t buffer_size,
        size_t* received_size,
        SocketAddress* address,
        int flags = 0
    );
};

//////////////////////////////////////////////////////////////////////////////
// Socket members

inline Socket::Socket() : m_handle(kInvalidHandle)
{
}

inline Socket::Socket(SOCKET handle) : m_handle(handle)
{
}

inline Socket::~Socket()
{
    Close();
}

inline SOCKET Socket::Handle() const
{
    return m_handle;
}

inline bool Socket::IsValid() const
{
    return m_handle != kInvalidHandle;
}

inline bool Socket::SetCloexec(bool value)
{
    return SetCloexec(static_cast<int>(m_handle), value);
}

inline bool Socket::GetOption(int level, int name, void* value, socklen_t* length) const
{
    return CheckError(getsockopt(m_handle, level, name, static_cast<char*>(value), length),
                      "GetOption");
}

inline bool Socket::SetOption(int level, int name, const void* value, socklen_t length)
{
    return CheckError(setsockopt(m_handle, level, name, static_cast<const char*>(value), length),
                      "SetOption");
}

inline bool Socket::GetOption(int level, int name, bool* value) const
{
    int int_value;
    bool result = GetOption(level, name, &int_value);
    *value = int_value != 0;
    return result;
}

inline bool Socket::SetOption(int level, int name, const bool& value)
{
    return SetOption(level, name, static_cast<int>(value));
}

inline bool Socket::GetError(int* error)
{
    return GetOption(SOL_SOCKET, SO_ERROR, error);
}

inline bool Socket::GetType(int* type) const
{
    return GetOption(SOL_SOCKET, SO_TYPE, type);
}

inline bool Socket::GetSendBufferSize(size_t* size) const
{
    return GetOption<size_t, int>(SOL_SOCKET, SO_SNDBUF, size);
}

inline bool Socket::SetSendBufferSize(size_t size)
{
    return SetOption<size_t, int>(SOL_SOCKET, SO_SNDBUF, size);
}

inline bool Socket::GetReceiveBufferSize(size_t* size) const
{
    return GetOption<size_t, int>(SOL_SOCKET, SO_RCVBUF, size);
}

inline bool Socket::SetReceiveBufferSize(size_t size)
{
    return SetOption<size_t, int>(SOL_SOCKET, SO_RCVBUF, size);
}

#ifdef _WIN32

inline bool Socket::SetSendTimeout(int seconds, int msec)
{
    int option = seconds * 1000 + msec;
    return SetOption(SOL_SOCKET, SO_SNDTIMEO, option);
}

inline bool Socket::SetReceiveTimeout(int seconds, int msec)
{
    int option = seconds * 1000 + msec;
    return SetOption(SOL_SOCKET, SO_RCVTIMEO, option);
}

inline bool Socket::SetSendTimeout(const timeval& tv)
{
    return SetSendTimeout(tv.tv_sec, tv.tv_usec / 1000);
}

inline bool Socket::SetReceiveTimeout(const timeval& tv)
{
    return SetReceiveTimeout(tv.tv_sec, tv.tv_usec / 1000);
}

#else // _WIN32

inline bool Socket::SetSendTimeout(const timeval& tv)
{
    return SetOption(SOL_SOCKET, SO_SNDTIMEO, tv);
}

inline bool Socket::SetReceiveTimeout(const timeval& tv)
{
    return SetOption(SOL_SOCKET, SO_RCVTIMEO, tv);
}

inline bool Socket::SetSendTimeout(int seconds, int msec)
{
    timeval tv = { seconds, msec * 1000 };
    return SetSendTimeout(tv);
}

inline bool Socket::SetReceiveTimeout(int seconds, int msec)
{
    timeval tv = { seconds, msec * 1000 };
    return SetReceiveTimeout(tv);
}

#endif // _WIN32

inline bool Socket::Ioctl(int cmd, int* value)
{
    return ioctlsocket(Handle(), cmd, reinterpret_cast<u_long*>(value)) == 0;
}

inline bool Socket::SetBlocking(bool value)
{
    return SocketSetNonblocking(Handle(), !value) == 0;
}

inline bool Socket::GetBlocking(bool* value)
{
    int n = SocketGetNonblocking(Handle(), value);
    *value = !(*value);
    return n == 0;
}

inline bool Socket::Bind(const SocketAddress& address)
{
    return CheckError(bind(Handle(), address.Address(), address.Length()), "Bind");
}

inline bool Socket::GetReuseAddress(bool* value)
{
    return GetOption(SOL_SOCKET, SO_REUSEADDR, value);
}

inline bool Socket::SetReuseAddress(bool value)
{
    return SetOption(SOL_SOCKET, SO_REUSEADDR, value);
}

inline bool Socket::SetKeepAlive(bool onoff)
{
    return SetOption(SOL_SOCKET, SO_KEEPALIVE, onoff);
}

inline bool Socket::GetKeepAlive(bool* onoff)
{
    return GetOption(SOL_SOCKET, SO_KEEPALIVE, onoff);
}

inline bool Socket::SetTcpNoDelay(bool onoff)
{
    return SetOption(IPPROTO_TCP, TCP_NODELAY, onoff);
}

inline bool Socket::GetTcpNoDelay(bool* onoff)
{
    return GetOption(IPPROTO_TCP, TCP_NODELAY, onoff);
}

inline bool Socket::IsReadable()
{
    struct timeval tv = {0, 0};
    return WaitReadable(&tv);
}

inline bool Socket::IsWriteable()
{
    struct timeval tv = {0, 0};
    return WaitWriteable(&tv);
}

inline int Socket::GetLastError()
{
    return SocketGetLastError();
}

inline std::string Socket::GetErrorString(int error)
{
    return SocketGetErrorString(error);
}

inline std::string Socket::GetLastErrorString()
{
    return SocketGetErrorString(GetLastError());
}

inline void Socket::SetLastError(int error)
{
    SocketSetLastError(error);
}

inline void Socket::VerifyHandle(int fd)
{
    assert(fd != kInvalidHandle);
}

inline bool Socket::IsInterruptedAndRestart(bool restart)
{
    return restart && GetLastError() == SOCKET_ERROR_CODE(EINTR);
}

/////////////////////////////////////////////////////////////////////////////
// ListenerSocket members

inline ListenerSocket::ListenerSocket()
{
}

inline ListenerSocket::ListenerSocket(int af, int type, int protocol) :
    Socket(socket(af, type, protocol))
{
}

inline bool ListenerSocket::Listen(int backlog)
{
    return CheckError(listen(Handle(), backlog), "Listen");
}

// include all inline functions
/////////////////////////////////////////////////////////////////////////////
// StreamSocket members

inline StreamSocket::StreamSocket(int af, int protocol)
{
    if (!DataSocket::Create(af, SOCK_STREAM, protocol))
        throw SocketError("StreamSocket");
}

inline bool StreamSocket::Create(sa_family_t af, int protocol)
{
    return Socket::Create(af, SOCK_STREAM, protocol);
}

inline bool StreamSocket::Shutdown()
{
    return CheckError(SocketShutdown(Handle()), "Shutdown");
}

inline bool StreamSocket::ShutdownSend()
{
    return CheckError(SocketShutdownSend(Handle()), "ShutdownSend");
}

inline bool StreamSocket::ShutdownReceive()
{
    return CheckError(SocketShutdownReceive(Handle()), "ShutdownReceive");
}

/////////////////////////////////////////////////////////////////////////////
// DatagramSocket members

inline bool DatagramSocket::Create(int af, int protocol)
{
    return Socket::Create(af, SOCK_DGRAM, protocol);
}

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_NET_SOCKET_H_