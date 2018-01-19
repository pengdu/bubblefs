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
// limitations under the License

// mesos/3rdparty/libprocess/include/process/process.hpp
// mesos/3rdparty/libprocess/include/process/dispatch.hpp
// mesos/3rdparty/libprocess/include/process/async.hpp
// mesos/3rdparty/libprocess/include/process/deferred.hpp
// mesos/3rdparty/libprocess/include/process/executor.hpp
// mesos/3rdparty/libprocess/include/process/defer.hpp
// mesos/3rdparty/libprocess/include/process/run.hpp
// mesos/3rdparty/libprocess/include/process/loop.hpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_PROCESS_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_PROCESS_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <map>
#include <queue>
#include <string>
#include <vector>

#include <process/address.hpp>
#include <process/authenticator.hpp>
#include <process/clock.hpp>
#include <process/event.hpp>
#include <process/filter.hpp>
#include <process/firewall.hpp>
#include <process/http.hpp>
#include <process/message.hpp>
#include <process/mime.hpp>
#include <process/owned.hpp>
#include <process/pid.hpp>

#include <stout/duration.hpp>
#include <stout/hashmap.hpp>
#include <stout/lambda.hpp>
#include <stout/option.hpp>
#include <stout/synchronized.hpp>

// Processes (aka Actors)
// An actor in libprocess is called a process (not to be confused by an operating system process).
// A process receives events that it processes one at a time. 
// Because a process is only processing one event at a time there's no need for synchronization within the process.
// There are a few ways to create an event for a process, the most important including:
// 1. You can send() a process a message.
// 2. You can do a function dispatch() on a process.
// 3. You can send an http::Request to a process. 
//    every process is also an HTTP-based service that you can communicate with using the HTTP protocol.

namespace bubblefs {
namespace mymesos {
namespace process {

// Forward declaration.
class EventQueue;
class Gate;
class Logging;
class Sequence;

namespace firewall {

/**
 * Install a list of firewall rules which are used to forbid incoming
 * HTTP requests.
 *
 * The rules will be applied in the provided order to each incoming
 * request. If any rule forbids the request, no more rules are applied
 * and a "403 Forbidden" response will be returned containing the reason
 * from the rule.
 *
 * **NOTE**: if a request is forbidden, the request's handler is
 * not notified.
 *
 * @see process::firewall::FirewallRule
 *
 * @param rules List of rules which will be applied to all incoming
 *     HTTP requests.
 */
void install(std::vector<Owned<FirewallRule>>&& rules);

} // namespace firewall {

class ProcessBase : public EventConsumer
{
public:
  explicit ProcessBase(const std::string& id = "");

  virtual ~ProcessBase();

  const UPID& self() const { return pid; }

protected:
  /**
   * Invoked when an event is serviced.
   */
  virtual void serve(Event&& event)
  {
    std::move(event).consume(this);
  }

  // Callbacks used to consume (i.e., handle) a specific event.
  void consume(MessageEvent&& event) override;
  void consume(DispatchEvent&& event) override;
  void consume(HttpEvent&& event) override;
  void consume(ExitedEvent&& event) override;
  void consume(TerminateEvent&& event) override;

  /**
   * Invoked when a process gets spawned.
   */
  virtual void initialize() {}

  /**
   * Invoked when a process is terminated.
   *
   * **NOTE**: this does not get invoked automatically if
   * `process::ProcessBase::consume(TerminateEvent&&)` is overridden.
   */
  virtual void finalize() {}

  /**
   * Invoked when a linked process has exited.
   *
   * For local linked processes (i.e., when the linker and linkee are
   * part of the same OS process), this can be used to reliably detect
   * when the linked process has exited.
   *
   * For remote linked processes, this indicates that the persistent
   * TCP connection between the linker and the linkee has failed
   * (e.g., linkee process died, a network error occurred). In this
   * situation, the remote linkee process might still be running.
   *
   * @see process::ProcessBase::link
   */
  virtual void exited(const UPID&) {}

  /**
   * Invoked when a linked process can no longer be monitored.
   *
   * TODO(neilc): This is not implemented.
   *
   * @see process::ProcessBase::link
   */
  virtual void lost(const UPID&) {}

  /**
   * Sends the message to the specified `UPID`. Prefer the rvalue
   * reference overloads if the data can be moved in.
   *
   * @see process::Message
   */
  void send(
      const UPID& to,
      const std::string& name,
      const char* data = nullptr,
      size_t length = 0);

  void send(
      const UPID& to,
      std::string&& name);

  void send(
      const UPID& to,
      std::string&& name,
      std::string&& data);

  /**
   * Describes the behavior of the `link` call when the target `pid`
   * points to a remote process. This enum has no effect if the target
   * `pid` points to a local process.
   */
  enum class RemoteConnection
  {
    /**
     * If a persistent socket to the target `pid` does not exist,
     * a new link is created. If a persistent socket already exists,
     * `link` will subscribe this process to the existing link.
     *
     * This is the default behavior.
     */
    REUSE,

    /**
     * If a persistent socket to the target `pid` does not exist,
     * a new link is created. If a persistent socket already exists,
     * `link` create a new socket connection with the target `pid`
     * and *atomically* swap the existing link with the new link.
     *
     * Existing linkers will remain linked, albeit via the new socket.
     */
    RECONNECT,
  };

  /**
   * Links with the specified `UPID`.
   *
   * Linking with a process from within the same OS process is
   * guaranteed to give you perfect monitoring of that process.
   *
   * Linking to a remote process establishes a persistent TCP
   * connection to the remote libprocess instance that hosts that
   * process. If the TCP connection fails, the true state of the
   * remote linked process cannot be determined; we handle this
   * situation by generating an ExitedEvent.
   */
  UPID link(
      const UPID& pid,
      const RemoteConnection remote = RemoteConnection::REUSE);

  /**
   * Any function which takes a "from" `UPID` and a message body as
   * arguments.
   *
   * The default consume implementation for message events invokes
   * installed message handlers, or delegates the message to another
   * process. A message handler always takes precedence over delegating.
   *
   * @see process::ProcessBase::install
   * @see process::ProcessBase::delegate
   */
  typedef lambda::function<void(const UPID&, const std::string&)>
  MessageHandler;

  /**
   * Sets up a handler for messages with the specified name.
   */
  void install(
      const std::string& name,
      const MessageHandler& handler)
  {
    handlers.message[name] = handler;
  }

  /**
   * @copydoc process::ProcessBase::install
   */
  template <typename T>
  void install(
      const std::string& name,
      void (T::*method)(const UPID&, const std::string&))
  {
    // Note that we use dynamic_cast here so a process can use
    // multiple inheritance if it sees so fit (e.g., to implement
    // multiple callback interfaces).
    MessageHandler handler =
      lambda::bind(method, dynamic_cast<T*>(this), lambda::_1, lambda::_2);
    install(name, handler);
  }

  /**
   * Delegates incoming messages, with the specified name, to the `UPID`.
   */
  void delegate(const std::string& name, const UPID& pid)
  {
    delegates[name] = pid;
  }

  /**
   * Any function which takes a `process::http::Request` and returns a
   * `process::http::Response`.
   *
   * The default consume implementation for HTTP events invokes
   * installed HTTP handlers.
   *
   * @see process::ProcessBase::route
   */
  typedef lambda::function<Future<http::Response>(const http::Request&)>
  HttpRequestHandler;

  // Options to control the behavior of a route.
  struct RouteOptions
  {
    RouteOptions()
      : requestStreaming(false) {}

    // Set to true if the endpoint supports request streaming.
    // Default: false.
    bool requestStreaming;
  };

  /**
   * Sets up a handler for HTTP requests with the specified name.
   *
   * @param name The endpoint or URL to route.
   *     Must begin with a `/`.
   */
  void route(
      const std::string& name,
      const Option<std::string>& help,
      const HttpRequestHandler& handler,
      const RouteOptions& options = RouteOptions());

  /**
   * @copydoc process::ProcessBase::route
   */
  template <typename T>
  void route(
      const std::string& name,
      const Option<std::string>& help,
      Future<http::Response> (T::*method)(const http::Request&),
      const RouteOptions& options = RouteOptions())
  {
    // Note that we use dynamic_cast here so a process can use
    // multiple inheritance if it sees so fit (e.g., to implement
    // multiple callback interfaces).
    HttpRequestHandler handler =
      lambda::bind(method, dynamic_cast<T*>(this), lambda::_1);
    route(name, help, handler, options);
  }

  /**
   * Any function which takes a `process::http::Request` and an
   * `Option<Principal>` and returns a `process::http::Response`.
   * This type is meant to be used for the endpoint handlers of
   * authenticated HTTP endpoints.
   *
   * If the handler is called and the principal is set,
   * this implies two things:
   *   1) The realm that the handler's endpoint is installed into
   *      requires authentication.
   *   2) The HTTP request has been successfully authenticated.
   *
   * If the principal is not set, then the endpoint's
   * realm does not require authentication.
   *
   * The default consume implementation for HTTP events invokes
   * installed HTTP handlers.
   *
   * @see process::ProcessBase::route
   */
  typedef lambda::function<Future<http::Response>(
      const http::Request&,
      const Option<http::authentication::Principal>&)>
          AuthenticatedHttpRequestHandler;

  // TODO(arojas): Consider introducing an `authentication::Realm` type.
  void route(
      const std::string& name,
      const std::string& realm,
      const Option<std::string>& help,
      const AuthenticatedHttpRequestHandler& handler,
      const RouteOptions& options = RouteOptions());

  /**
   * @copydoc process::ProcessBase::route
   */
  template <typename T>
  void route(
      const std::string& name,
      const std::string& realm,
      const Option<std::string>& help,
      Future<http::Response> (T::*method)(
          const http::Request&,
          const Option<http::authentication::Principal>&),
      const RouteOptions& options = RouteOptions())
  {
    // Note that we use dynamic_cast here so a process can use
    // multiple inheritance if it sees so fit (e.g., to implement
    // multiple callback interfaces).
    AuthenticatedHttpRequestHandler handler =
      lambda::bind(method, dynamic_cast<T*>(this), lambda::_1, lambda::_2);
    route(name, realm, help, handler, options);
  }

  /**
   * Sets up the default HTTP request handler to provide the static
   * asset(s) at the specified _absolute_ path for the specified name.
   *
   * For example, assuming the process named "server" invoked
   * `provide("name", "path")`, then an HTTP request for `/server/name`
   * would return the asset found at "path". If the specified path is a
   * directory then an HTTP request for `/server/name/file` would return
   * the asset found at `/path/file`.
   *
   * The `Content-Type` header of the HTTP response will be set to the
   * specified type given the file extension, which can be changed via
   * the optional `types` parameter.
   */
  void provide(
      const std::string& name,
      const std::string& path,
      const std::map<std::string, std::string>& types = mime::types)
  {
    // TODO(benh): Check that name is only alphanumeric (i.e., has no
    // '/') and that path is absolute.
    Asset asset;
    asset.path = path;
    asset.types = types;
    assets[name] = asset;
  }

  /**
   * Returns the number of events of the given type currently on the
   * event queue. MUST be invoked from within the process itself in
   * order to safely examine events.
   */
  template <typename T>
  size_t eventCount();

private:
  friend class SocketManager;
  friend class ProcessManager;
  friend void* schedule(void*);

  // Process states.
  //
  // Transitioning from BLOCKED to READY also requires enqueueing the
  // process in the run queue otherwise the events will never be
  // processed!
  enum class State
  {
    BOTTOM, // Uninitialized but events may be enqueued.
    BLOCKED, // Initialized, no events enqueued.
    READY, // Initialized, events enqueued.
    TERMINATING // Initialized, no more events will be enqueued.
  };

  std::atomic<State> state = ATOMIC_VAR_INIT(State::BOTTOM);

  // Flag for indicating that a terminate event has been injected.
  std::atomic<bool> termination = ATOMIC_VAR_INIT(false);

  // Enqueue the specified message, request, or function call.
  void enqueue(Event* event);

  // Delegates for messages.
  std::map<std::string, UPID> delegates;

  // Definition of an HTTP endpoint. The endpoint can be
  // associated with an authentication realm, in which case:
  //
  //  (1) `realm` and `authenticatedHandler` will be set.
  //      Libprocess will perform HTTP authentication for
  //      all requests to this endpoint (by default during
  //      HttpEvent consumption). The authentication principal
  //      will be passed to the `authenticatedHandler`.
  //
  //  Otherwise, if the endpoint is not associated with an
  //  authentication realm:
  //
  //  (2) Only `handler` will be set, and no authentication
  //      takes place.
  struct HttpEndpoint
  {
    Option<HttpRequestHandler> handler;

    Option<std::string> realm;
    Option<AuthenticatedHttpRequestHandler> authenticatedHandler;
    RouteOptions options;
  };

  // Handlers for messages and HTTP requests.
  struct {
    hashmap<std::string, MessageHandler> message;
    hashmap<std::string, HttpEndpoint> http;

    // Used for delivering HTTP requests in the correct order.
    // Initialized lazily to avoid ProcessBase requiring
    // another Process!
    Owned<Sequence> httpSequence;
  } handlers;

  // Definition of a static asset.
  struct Asset
  {
    std::string path;
    std::map<std::string, std::string> types;
  };

  // Continuation for `consume(HttpEvent&&)`.
  Future<http::Response> _consume(
      const HttpEndpoint& endpoint,
      const std::string& name,
      const Owned<http::Request>& request);

  // JSON representation of process. MUST be invoked from within the
  // process itself in order to safely examine events.
  operator JSON::Object();

  // Static assets(s) to provide.
  std::map<std::string, Asset> assets;

  // Queue of received events. We employ the PIMPL idiom here and use
  // a pointer so we can hide the implementation of `EventQueue`.
  std::unique_ptr<EventQueue> events;

  // NOTE: this is a shared pointer to a _pointer_, hence this is not
  // responsible for the ProcessBase itself.
  std::shared_ptr<ProcessBase*> reference;

  std::shared_ptr<Gate> gate;

  // Whether or not the runtime should delete this process after it
  // has terminated. Note that failure to spawn the process will leave
  // the process unmanaged and thus it may leak!
  bool manage = false;

  // Process PID.
  UPID pid;
};

// You create a process like any other class in C++ but extending from Process. 
// Process uses the curiously recurring template pattern (CRTP) to simplify types 
// for some of it's methods (you'll see this with Process::self() below).
// class FooProcess : public Process<FooProcess> {};
// Practically you can think of a process as a combination of a thread and an object, 
// except creating/spawning a process is very cheap (no actual thread gets created,
// and no stack gets allocated).
// FooProcess process; spawn(process); terminate(process); terminate(process);
// A process CAN NOT be deleted until after doing a wait(), 
// otherwise you might release resources that the library is still using! 
// To simplify memory management you can ask the library to delete the process 
// for you after it has completely terminated. You do this by invoking spawn() and passing true as the second argument
// spawn(process, true); // <-- `process` will be automatically deleted!
// 

template <typename T>
class Process : public virtual ProcessBase {
public:
  virtual ~Process() {}

  // A process is uniquely identifiable by it's process id which can be any arbitrary string 
  // (but only one process can be spawned at a time with the same id). 
  // The PID and UPID types encapsulate both the process id as well as the network address 
  // for the process, e.g., the IP and port where the process can be reached 
  // if libprocess was initialized with an IPv4 or IPv6 network address. 
  // You can get the PID of a process by calling it's self() method.
  // A UPID is the "untyped" base class of PID.
  // If you turn on logging you might see a PID/UPID printed out as id@ip:port.
  /**
   * Returns the `PID` of the process.
   *
   * Valid even before calling spawn.
   */
  PID<T> self() const { return PID<T>(static_cast<const T*>(this)); }

protected:
  // Useful typedefs for dispatch/delay/defer to self()/this.
  typedef T Self;
  typedef T This;
};


/**
 * Initialize the library.
 *
 * **NOTE**: `libprocess` uses Google's `glog` and you can specify options
 * for it (e.g., a logging directory) via environment variables.
 *
 * @param delegate Process to receive root HTTP requests.
 * @param readwriteAuthenticationRealm The authentication realm that read-write
 *     libprocess-level HTTP endpoints will be installed under, if any.
 *     If this realm is not specified, read-write endpoints will be installed
 *     without authentication.
 * @param readonlyAuthenticationRealm The authentication realm that read-only
 *     libprocess-level HTTP endpoints will be installed under, if any.
 *     If this realm is not specified, read-only endpoints will be installed
 *     without authentication.
 * @return `true` if this was the first invocation of `process::initialize()`,
 *     or `false` if it was not the first invocation.
 *
 * @see [glog](https://google-glog.googlecode.com/svn/trunk/doc/glog.html)
 */
bool initialize(
    const Option<std::string>& delegate = None(),
    const Option<std::string>& readwriteAuthenticationRealm = None(),
    const Option<std::string>& readonlyAuthenticationRealm = None());


/**
 * Clean up the library.
 *
 * @param finalize_wsa Whether the Windows socket stack should be cleaned
 *     up for the entire process. Has no effect outside of Windows.
 */
void finalize(bool finalize_wsa = false);


/**
 * Get the request absolutePath path with delegate prefix.
 */
std::string absolutePath(const std::string& path);


/**
 * Returns the socket address associated with this instance of the library.
 */
network::inet::Address address();


/**
 * Return the PID associated with the global logging process.
 */
PID<Logging> logging();


/**
 * Returns the number of worker threads the library has created. A
 * worker thread is a thread that runs a process (i.e., calls
 * `ProcessBase::serve`).
 */
long workers();


/**
 * Spawn a new process.
 *
 * @param process Process to be spawned.
 * @param manage Whether process should get deleted by the runtime
 *     after terminating.
 */
UPID spawn(ProcessBase* process, bool manage = false);

inline UPID spawn(ProcessBase& process, bool manage = false)
{
  return spawn(&process, manage);
}

template <typename T>
PID<T> spawn(T* t, bool manage = false)
{
  // We save the pid before spawn is called because it's possible that
  // the process has already been deleted after spawn returns (e.g.,
  // if 'manage' is true).
  PID<T> pid(t);

  if (!spawn(static_cast<ProcessBase*>(t), manage)) {
    return PID<T>();
  }

  return pid;
}

template <typename T>
PID<T> spawn(T& t, bool manage = false)
{
  return spawn(&t, manage);
}


/**
 * Sends a `TerminateEvent` to the given process.
 *
 * **NOTE**: currently, terminate only works for local processes (in the
 * future we plan to make this more explicit via the use of a `PID`
 * instead of a `UPID`).
 *
 * @param pid The process to terminate.
 * @param inject Whether the message should be injected ahead of all other
 *     messages queued up for that process.
 *
 * @see process::TerminateEvent
 */
void terminate(const UPID& pid, bool inject = true);
void terminate(const ProcessBase& process, bool inject = true);
void terminate(const ProcessBase* process, bool inject = true);


/**
 * Wait for the process to exit for no more than the specified seconds.
 *
 * @param PID ID of the process.
 * @param secs Max time to wait, 0 implies wait forever.
 *
 * @return true if a process was actually waited upon.
 */
bool wait(const UPID& pid, const Duration& duration = Seconds(-1));
bool wait(const ProcessBase& process, const Duration& duration = Seconds(-1));
bool wait(const ProcessBase* process, const Duration& duration = Seconds(-1));


/**
 * Sends a message with data without a return address.
 *
 * @param to Receiver of the message.
 * @param name Name of the message.
 * @param data Data to send (gets copied).
 * @param length Length of data.
 */
void post(const UPID& to,
          const std::string& name,
          const char* data = nullptr,
          size_t length = 0);


void post(const UPID& from,
          const UPID& to,
          const std::string& name,
          const char* data = nullptr,
          size_t length = 0);


/**
 * @copydoc process::terminate
 */
inline void terminate(const ProcessBase& process, bool inject)
{
  terminate(process.self(), inject);
}


/**
 * @copydoc process::terminate
 */
inline void terminate(const ProcessBase* process, bool inject)
{
  terminate(process->self(), inject);
}


/**
 * @copydoc process::wait
 */
inline bool wait(const ProcessBase& process, const Duration& duration)
{
  return process::wait(process.self(), duration); // Explicit to disambiguate.
}


/**
 * @copydoc process::wait
 */
inline bool wait(const ProcessBase* process, const Duration& duration)
{
  return process::wait(process->self(), duration); // Explicit to disambiguate.
}


// Per thread process pointer.
extern thread_local ProcessBase* __process__;

// NOTE: Methods in this namespace should only be used in tests to
// inject arbitrary events.
namespace inject {

/**
 * Simulates disconnection of the link between 'from' and 'to' by
 * sending an `ExitedEvent` to 'to'.
 *
 * @see process::ExitedEvent
 */
bool exited(const UPID& from, const UPID& to);

} // namespace inject 

// Processes and the Asynchronous Pimpl Pattern
// It's tedious to require everyone to have to explicitly 
// spawn(), terminate(), and wait() for a process. Having everyone call dispatch() 
// when they really just want to invoke a function (albeit asynchronously) is unfortnate too! 
// To alleviate these burdenes, a common pattern that is used is to wrap a process 
// within another class that performs the spawn(), terminate(), wait(), and dispatch()'s for you
// This is similar to the Pimpl pattern, except we need to spawn() and terminate()/wait() 
// and rather than synchronously invoking the underlying object 
// we're asynchronously invoking the underlying object using dispatch().
// 
// The dispatch mechanism enables you to "schedule" a method to get
// invoked on a process. The result of that method invocation is
// accessible via the future that is returned by the dispatch method
// (note, however, that it might not be the _same_ future as the one
// returned from the method, if the method even returns a future, see
// below). Assuming some class 'Fibonacci' has a (visible) method
// named 'compute' that takes an integer, N (and returns the Nth
// fibonacci number) you might use dispatch like so:
//
// PID<Fibonacci> pid = spawn(new Fibonacci(), true); // Use the GC.
// Future<int> f = dispatch(pid, &Fibonacci::compute, 10);
//
// Because the pid argument is "typed" we can ensure that methods are
// only invoked on processes that are actually of that type. Providing
// this mechanism for varying numbers of function types and arguments
// requires support for variadic templates, slated to be released in
// C++11. Until then, we use the Boost preprocessor macros to
// accomplish the same thing (albeit less cleanly). See below for
// those definitions.
//
// Dispatching is done via a level of indirection. The dispatch
// routine itself creates a promise that is passed as an argument to a
// partially applied 'dispatcher' function (defined below). The
// dispatcher routines get passed to the actual process via an
// internal routine called, not surprisingly, 'dispatch', defined
// below:

namespace internal {

// The internal dispatch routine schedules a function to get invoked
// within the context of the process associated with the specified pid
// (first argument), unless that process is no longer valid. Note that
// this routine does not expect anything in particular about the
// specified function (second argument). The semantics are simple: the
// function gets applied/invoked with the process as its first
// argument.
void dispatch(
    const UPID& pid,
    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f,
    const Option<const std::type_info*>& functionType = None());


// NOTE: This struct is used by the public `dispatch(const UPID& pid, F&& f)`
// function. See comments there for the reason why we need this.
template <typename R>
struct Dispatch;


// Partial specialization for callable objects returning `void` to be dispatched
// on a process.
// NOTE: This struct is used by the public `dispatch(const UPID& pid, F&& f)`
// function. See comments there for the reason why we need this.
template <>
struct Dispatch<void>
{
  template <typename F>
  void operator()(const UPID& pid, F&& f)
  {
    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f_(
        new lambda::CallableOnce<void(ProcessBase*)>(
            lambda::partial(
                [](typename std::decay<F>::type&& f, ProcessBase*) {
                  std::move(f)();
                },
                std::forward<F>(f),
                lambda::_1)));

    internal::dispatch(pid, std::move(f_));
  }
};


// Partial specialization for callable objects returning `Future<R>` to be
// dispatched on a process.
// NOTE: This struct is used by the public `dispatch(const UPID& pid, F&& f)`
// function. See comments there for the reason why we need this.
template <typename R>
struct Dispatch<Future<R>>
{
  template <typename F>
  Future<R> operator()(const UPID& pid, F&& f)
  {
    std::unique_ptr<Promise<R>> promise(new Promise<R>());
    Future<R> future = promise->future();

    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f_(
        new lambda::CallableOnce<void(ProcessBase*)>(
            lambda::partial(
                [](std::unique_ptr<Promise<R>> promise,
                   typename std::decay<F>::type&& f,
                   ProcessBase*) {
                  promise->associate(std::move(f)());
                },
                std::move(promise),
                std::forward<F>(f),
                lambda::_1)));

    internal::dispatch(pid, std::move(f_));

    return future;
  }
};


// Dispatches a callable object returning `R` on a process.
// NOTE: This struct is used by the public `dispatch(const UPID& pid, F&& f)`
// function. See comments there for the reason why we need this.
template <typename R>
struct Dispatch
{
  template <typename F>
  Future<R> operator()(const UPID& pid, F&& f)
  {
    std::unique_ptr<Promise<R>> promise(new Promise<R>());
    Future<R> future = promise->future();

    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f_(
        new lambda::CallableOnce<void(ProcessBase*)>(
            lambda::partial(
                [](std::unique_ptr<Promise<R>> promise,
                   typename std::decay<F>::type&& f,
                   ProcessBase*) {
                  promise->set(std::move(f)());
                },
                std::move(promise),
                std::forward<F>(f),
                lambda::_1)));

    internal::dispatch(pid, std::move(f_));

    return future;
  }
};

} // namespace internal {


// Okay, now for the definition of the dispatch routines
// themselves. For each routine we provide the version in C++11 using
// variadic templates so the reader can see what the Boost
// preprocessor macros are effectively providing. Using C++11 closures
// would shorten these definitions even more.
//
// First, definitions of dispatch for methods returning void:

template <typename T>
void dispatch(const PID<T>& pid, void (T::*method)())
{
  std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(
      new lambda::CallableOnce<void(ProcessBase*)>(
          [=](ProcessBase* process) {
            assert(process != nullptr);
            T* t = dynamic_cast<T*>(process);
            assert(t != nullptr);
            (t->*method)();
          }));

  internal::dispatch(pid, std::move(f), &typeid(method));
}

template <typename T>
void dispatch(const Process<T>& process, void (T::*method)())
{
  dispatch(process.self(), method);
}

template <typename T>
void dispatch(const Process<T>* process, void (T::*method)())
{
  dispatch(process->self(), method);
}

// Due to a bug (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933)
// with variadic templates and lambdas, we still need to do
// preprocessor expansions.

// The following assumes base names for type and variable are `A` and `a`.
#define FORWARD(Z, N, DATA) std::forward<A ## N>(a ## N)
#define MOVE(Z, N, DATA) std::move(a ## N)
#define DECL(Z, N, DATA) typename std::decay<A ## N>::type&& a ## N

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  void dispatch(                                                        \
      const PID<T>& pid,                                                \
      void (T::*method)(ENUM_PARAMS(N, P)),                             \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(        \
        new lambda::CallableOnce<void(ProcessBase*)>(                   \
            lambda::partial(                                            \
                [method](ENUM(N, DECL, _), ProcessBase* process) {      \
                  assert(process != nullptr);                           \
                  T* t = dynamic_cast<T*>(process);                     \
                  assert(t != nullptr);                                 \
                  (t->*method)(ENUM(N, MOVE, _));                       \
                },                                                      \
                ENUM(N, FORWARD, _),                                    \
                lambda::_1)));                                          \
                                                                        \
    internal::dispatch(pid, std::move(f), &typeid(method));             \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  void dispatch(                                                        \
      const Process<T>& process,                                        \
      void (T::*method)(ENUM_PARAMS(N, P)),                             \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    dispatch(process.self(), method, ENUM(N, FORWARD, _));              \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  void dispatch(                                                        \
      const Process<T>* process,                                        \
      void (T::*method)(ENUM_PARAMS(N, P)),                             \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    dispatch(process->self(), method, ENUM(N, FORWARD, _));             \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE


// Next, definitions of methods returning a future:

template <typename R, typename T>
Future<R> dispatch(const PID<T>& pid, Future<R> (T::*method)())
{
  std::unique_ptr<Promise<R>> promise(new Promise<R>());
  Future<R> future = promise->future();

  std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(
      new lambda::CallableOnce<void(ProcessBase*)>(
          lambda::partial(
              [=](std::unique_ptr<Promise<R>> promise, ProcessBase* process) {
                assert(process != nullptr);
                T* t = dynamic_cast<T*>(process);
                assert(t != nullptr);
                promise->associate((t->*method)());
              },
              std::move(promise),
              lambda::_1)));

  internal::dispatch(pid, std::move(f), &typeid(method));

  return future;
}

template <typename R, typename T>
Future<R> dispatch(const Process<T>& process, Future<R> (T::*method)())
{
  return dispatch(process.self(), method);
}

template <typename R, typename T>
Future<R> dispatch(const Process<T>* process, Future<R> (T::*method)())
{
  return dispatch(process->self(), method);
}

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const PID<T>& pid,                                                \
      Future<R> (T::*method)(ENUM_PARAMS(N, P)),                        \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    std::unique_ptr<Promise<R>> promise(new Promise<R>());              \
    Future<R> future = promise->future();                               \
                                                                        \
    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(        \
        new lambda::CallableOnce<void(ProcessBase*)>(                   \
            lambda::partial(                                            \
                [method](std::unique_ptr<Promise<R>> promise,           \
                         ENUM(N, DECL, _),                              \
                         ProcessBase* process) {                        \
                  assert(process != nullptr);                           \
                  T* t = dynamic_cast<T*>(process);                     \
                  assert(t != nullptr);                                 \
                  promise->associate(                                   \
                      (t->*method)(ENUM(N, MOVE, _)));                  \
                },                                                      \
                std::move(promise),                                     \
                ENUM(N, FORWARD, _),                                    \
                lambda::_1)));                                          \
                                                                        \
    internal::dispatch(pid, std::move(f), &typeid(method));             \
                                                                        \
    return future;                                                      \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const Process<T>& process,                                        \
      Future<R> (T::*method)(ENUM_PARAMS(N, P)),                        \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    return dispatch(process.self(), method, ENUM(N, FORWARD, _));       \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const Process<T>* process,                                        \
      Future<R> (T::*method)(ENUM_PARAMS(N, P)),                        \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    return dispatch(process->self(), method, ENUM(N, FORWARD, _));      \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE


// Next, definitions of methods returning a value.

template <typename R, typename T>
Future<R> dispatch(const PID<T>& pid, R (T::*method)())
{
  std::unique_ptr<Promise<R>> promise(new Promise<R>());
  Future<R> future = promise->future();

  std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(
      new lambda::CallableOnce<void(ProcessBase*)>(
          lambda::partial(
              [=](std::unique_ptr<Promise<R>> promise, ProcessBase* process) {
                assert(process != nullptr);
                T* t = dynamic_cast<T*>(process);
                assert(t != nullptr);
                promise->set((t->*method)());
              },
              std::move(promise),
              lambda::_1)));

  internal::dispatch(pid, std::move(f), &typeid(method));

  return future;
}

template <typename R, typename T>
Future<R> dispatch(const Process<T>& process, R (T::*method)())
{
  return dispatch(process.self(), method);
}

template <typename R, typename T>
Future<R> dispatch(const Process<T>* process, R (T::*method)())
{
  return dispatch(process->self(), method);
}

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const PID<T>& pid,                                                \
      R (T::*method)(ENUM_PARAMS(N, P)),                                \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    std::unique_ptr<Promise<R>> promise(new Promise<R>());              \
    Future<R> future = promise->future();                               \
                                                                        \
    std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f(        \
        new lambda::CallableOnce<void(ProcessBase*)>(                   \
            lambda::partial(                                            \
                [method](std::unique_ptr<Promise<R>> promise,           \
                         ENUM(N, DECL, _),                              \
                         ProcessBase* process) {                        \
                  assert(process != nullptr);                           \
                  T* t = dynamic_cast<T*>(process);                     \
                  assert(t != nullptr);                                 \
                  promise->set((t->*method)(ENUM(N, MOVE, _)));         \
                },                                                      \
                std::move(promise),                                     \
                ENUM(N, FORWARD, _),                                    \
                lambda::_1)));                                          \
                                                                        \
    internal::dispatch(pid, std::move(f), &typeid(method));             \
                                                                        \
    return future;                                                      \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const Process<T>& process,                                        \
      R (T::*method)(ENUM_PARAMS(N, P)),                                \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    return dispatch(process.self(), method, ENUM(N, FORWARD, _));       \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> dispatch(                                                   \
      const Process<T>* process,                                        \
      R (T::*method)(ENUM_PARAMS(N, P)),                                \
      ENUM_BINARY_PARAMS(N, A, &&a))                                    \
  {                                                                     \
    return dispatch(process->self(), method, ENUM(N, FORWARD, _));      \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

#undef DECL
#undef MOVE
#undef FORWARD

// We use partial specialization of
//   - internal::Dispatch<void> vs
//   - internal::Dispatch<Future<R>> vs
//   - internal::Dispatch
// in order to determine whether R is void, Future or other types.
template <typename F, typename R = typename result_of<F()>::type>
auto dispatch(const UPID& pid, F&& f)
  -> decltype(internal::Dispatch<R>()(pid, std::forward<F>(f)))
{
  return internal::Dispatch<R>()(pid, std::forward<F>(f));
}

// Provides an abstraction for asynchronously executing a function
// (note the declarations are here and definitions below since
// defining and declaring below will require defining the default
// argument when declaring these as friends in AsyncExecutor which is
// brittle).

template <typename F>
Future<typename result_of<F()>::type> async(
    const F& f,
    typename std::enable_if<!std::is_void<typename result_of<F()>::type>::value>::type* = nullptr); // NOLINT(whitespace/line_length)


template <typename F>
Future<Nothing> async(
    const F& f,
    typename std::enable_if<std::is_void<typename result_of<F()>::type>::value>::type* = nullptr); // NOLINT(whitespace/line_length)


#define TEMPLATE(Z, N, DATA)                                            \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<typename result_of<F(ENUM_PARAMS(N, A))>::type> async( \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<!std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type* = nullptr); /* NOLINT(whitespace/line_length) */ \
                                                                        \
                                                                        \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<Nothing> async(                                                \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type* = nullptr); // NOLINT(whitespace/line_length)

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE


// TODO(vinod): Merge this into ExecutorProcess.
class AsyncExecutorProcess : public Process<AsyncExecutorProcess>
{
private:
  friend class AsyncExecutor;

  AsyncExecutorProcess() : ProcessBase(ID::generate("__async_executor__")) {}
  virtual ~AsyncExecutorProcess() {}

  // Not copyable, not assignable.
  AsyncExecutorProcess(const AsyncExecutorProcess&);
  AsyncExecutorProcess& operator=(const AsyncExecutorProcess&);

  template <
      typename F,
      typename std::enable_if<
          !std::is_void<typename result_of<F()>::type>::value, int>::type = 0>
  typename result_of<F()>::type execute(const F& f)
  {
    terminate(self()); // Terminate process after function returns.
    return f();
  }

  template <
      typename F,
      typename std::enable_if<
          std::is_void<typename result_of<F()>::type>::value, int>::type = 0>
  Nothing execute(const F& f)
  {
    terminate(self()); // Terminate process after function returns.
    f();
    return Nothing();
  }

#define TEMPLATE(Z, N, DATA)                                            \
  template <                                                            \
      typename F,                                                       \
      ENUM_PARAMS(N, typename A),                                       \
      typename std::enable_if<                                          \
          !std::is_void<                                                \
              typename result_of<F(ENUM_PARAMS(N, A))>::type>::value,   \
          int>::type = 0>                                               \
  typename result_of<F(ENUM_PARAMS(N, A))>::type execute(       \
      const F& f, ENUM_BINARY_PARAMS(N, A, a))                          \
  {                                                                     \
    terminate(self()); /* Terminate process after function returns. */  \
    return f(ENUM_PARAMS(N, a));                                        \
  }                                                                     \
                                                                        \
  template <                                                            \
      typename F,                                                       \
      ENUM_PARAMS(N, typename A),                                       \
      typename std::enable_if<                                          \
          std::is_void<                                                 \
              typename result_of<F(ENUM_PARAMS(N, A))>::type>::value,   \
          int>::type = 0>                                               \
  Nothing execute(                                                      \
      const F& f, ENUM_BINARY_PARAMS(N, A, a))                          \
  {                                                                     \
    terminate(self()); /* Terminate process after function returns. */  \
    f(ENUM_PARAMS(N, a));                                               \
    return Nothing();                                                   \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE
};


// Acts like a function call but runs within an asynchronous execution
// context such as an Executor or a ProcessBase (enforced because only
// an executor or the 'defer' routines are allowed to create them).
template <typename F>
struct Deferred : std::function<F>
{
private:
  friend class Executor;

  template <typename G> friend struct _Deferred;

  // TODO(benh): Consider removing these in favor of having these
  // functions return _Deferred.
  template <typename T>
  friend Deferred<void()>
  defer(const PID<T>& pid, void (T::*method)());

  template <typename R, typename T>
  friend Deferred<Future<R>()>
  defer(const PID<T>& pid, Future<R> (T::*method)());

  template <typename R, typename T>
  friend Deferred<Future<R>()>
  defer(const PID<T>& pid, R (T::*method)());

  /*implicit*/ Deferred(const std::function<F>& f) : std::function<F>(f) {}
};


// We need an intermediate "deferred" type because when constructing a
// Deferred we won't always know the underlying function type (for
// example, if we're being passed a std::bind or a lambda). A lambda
// won't always implicitly convert to a std::function so instead we
// hold onto the functor type F and let the compiler invoke the
// necessary cast operator (below) when it actually has determined
// what type is needed. This is similar in nature to how std::bind
// works with its intermediate _Bind type (which the pre-C++11
// implementation relied on).
template <typename F>
struct _Deferred
{
  // We expect that conversion operators are invoked on rvalues only,
  // as _Deferred is supposed to be used directly as a result of defer call.
  operator Deferred<void()>() &&
  {
    // The 'pid' differentiates an already dispatched functor versus
    // one which still needs to be dispatched (which is done
    // below). We have to delay wrapping the dispatch (for example, in
    // defer.hpp) as long as possible because we don't always know
    // what type the functor is or is going to be cast to (i.e., a
    // std::bind might can be cast to functions that do or do not take
    // arguments which will just be dropped when invoking the
    // underlying bound function).
    if (pid.isNone()) {
      return std::function<void()>(std::forward<F>(f));
    }

    // We need to explicitly copy the members otherwise we'll
    // implicitly copy 'this' which might not exist at invocation.
    Option<UPID> pid_ = pid;
    F&& f_ = std::forward<F>(f);

    return std::function<void()>(
        [=]() {
          dispatch(pid_.get(), f_);
        });
  }

  operator std::function<void()>() &&
  {
    if (pid.isNone()) {
      return std::function<void()>(std::forward<F>(f));
    }

    Option<UPID> pid_ = pid;
    F&& f_ = std::forward<F>(f);

    return std::function<void()>(
        [=]() {
          dispatch(pid_.get(), f_);
        });
  }

  operator lambda::CallableOnce<void()>() &&
  {
    if (pid.isNone()) {
      return lambda::CallableOnce<void()>(std::forward<F>(f));
    }

    Option<UPID> pid_ = pid;

    return lambda::CallableOnce<void()>(
        lambda::partial(
            [pid_](typename std::decay<F>::type&& f_) {
              dispatch(pid_.get(), std::move(f_));
            },
            std::forward<F>(f)));
  }

  template <typename R>
  operator Deferred<R()>() &&
  {
    if (pid.isNone()) {
      return std::function<R()>(std::forward<F>(f));
    }

    Option<UPID> pid_ = pid;
    F&& f_ = std::forward<F>(f);

    return std::function<R()>(
        [=]() {
          return dispatch(pid_.get(), f_);
        });
  }

  template <typename R>
  operator std::function<R()>() &&
  {
    if (pid.isNone()) {
      return std::function<R()>(std::forward<F>(f));
    }

    Option<UPID> pid_ = pid;
    F&& f_ = std::forward<F>(f);

    return std::function<R()>(
        [=]() {
          return dispatch(pid_.get(), f_);
        });
  }

  template <typename R>
  operator lambda::CallableOnce<R()>() &&
  {
    if (pid.isNone()) {
      return lambda::CallableOnce<R()>(std::forward<F>(f));
    }

    Option<UPID> pid_ = pid;

    return lambda::CallableOnce<R()>(
        lambda::partial(
          [pid_](typename std::decay<F>::type&& f_) {
            return dispatch(pid_.get(), std::move(f_));
          },
          std::forward<F>(f)));
  }

// Expands to lambda::_$(N+1). N is zero-based, and placeholders are one-based.
#define PLACEHOLDER(Z, N, DATA) CAT(lambda::_, INC(N))

// This assumes type and variable base names are `P` and `p` respectively.
#define FORWARD(Z, N, DATA) std::forward<P ## N>(p ## N)

  // Due to a bug (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933)
  // with variadic templates and lambdas, we still need to do
  // preprocessor expansions. In addition, due to a bug with clang (or
  // libc++) we can't use std::bind with a std::function so we have to
  // explicitly use the std::function<R(P...)>::operator() (see
  // http://stackoverflow.com/questions/20097616/stdbind-to-a-stdfunction-crashes-with-clang).
#define TEMPLATE(Z, N, DATA)                                             \
  template <ENUM_PARAMS(N, typename P)>                                  \
  operator Deferred<void(ENUM_PARAMS(N, P))>() &&                        \
  {                                                                      \
    if (pid.isNone()) {                                                  \
      return std::function<void(ENUM_PARAMS(N, P))>(std::forward<F>(f)); \
    }                                                                    \
                                                                         \
    Option<UPID> pid_ = pid;                                             \
    F&& f_ = std::forward<F>(f);                                         \
                                                                         \
    return std::function<void(ENUM_PARAMS(N, P))>(                       \
        [=](ENUM_BINARY_PARAMS(N, P, p)) {                               \
          std::function<void()> f__([=]() {                              \
            f_(ENUM_PARAMS(N, p));                                       \
          });                                                            \
          dispatch(pid_.get(), f__);                                     \
        });                                                              \
  }                                                                      \
                                                                         \
  template <ENUM_PARAMS(N, typename P)>                                  \
  operator std::function<void(ENUM_PARAMS(N, P))>() &&                   \
  {                                                                      \
    if (pid.isNone()) {                                                  \
      return std::function<void(ENUM_PARAMS(N, P))>(std::forward<F>(f)); \
    }                                                                    \
                                                                         \
    Option<UPID> pid_ = pid;                                             \
    F&& f_ = std::forward<F>(f);                                         \
                                                                         \
    return std::function<void(ENUM_PARAMS(N, P))>(                       \
        [=](ENUM_BINARY_PARAMS(N, P, p)) {                               \
          std::function<void()> f__([=]() {                              \
            f_(ENUM_PARAMS(N, p));                                       \
          });                                                            \
          dispatch(pid_.get(), f__);                                     \
        });                                                              \
  }                                                                      \
                                                                         \
  template <ENUM_PARAMS(N, typename P)>                                  \
  operator lambda::CallableOnce<void(ENUM_PARAMS(N, P))>() &&            \
  {                                                                      \
    if (pid.isNone()) {                                                  \
      return lambda::CallableOnce<void(ENUM_PARAMS(N, P))>(              \
          std::forward<F>(f));                                           \
    }                                                                    \
                                                                         \
    Option<UPID> pid_ = pid;                                             \
                                                                         \
    return lambda::CallableOnce<void(ENUM_PARAMS(N, P))>(                \
        lambda::partial(                                                 \
            [pid_](typename std::decay<F>::type&& f_,                    \
                   ENUM_BINARY_PARAMS(N, P, &&p)) {                      \
              lambda::CallableOnce<void()> f__(                          \
                  lambda::partial(std::move(f_), ENUM(N, FORWARD, _)));  \
              dispatch(pid_.get(), std::move(f__));                      \
            },                                                           \
            std::forward<F>(f),                                          \
            ENUM(N, PLACEHOLDER, _)));                                   \
  }

  REPEAT_FROM_TO(1, 3, TEMPLATE, _) // Args A0 -> A1.
#undef TEMPLATE

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R, ENUM_PARAMS(N, typename P)>                     \
  operator Deferred<R(ENUM_PARAMS(N, P))>() &&                          \
  {                                                                     \
    if (pid.isNone()) {                                                 \
      return std::function<R(ENUM_PARAMS(N, P))>(std::forward<F>(f));   \
    }                                                                   \
                                                                        \
    Option<UPID> pid_ = pid;                                            \
    F&& f_ = std::forward<F>(f);                                        \
                                                                        \
    return std::function<R(ENUM_PARAMS(N, P))>(                         \
        [=](ENUM_BINARY_PARAMS(N, P, p)) {                              \
          std::function<R()> f__([=]() {                                \
            return f_(ENUM_PARAMS(N, p));                               \
          });                                                           \
          return dispatch(pid_.get(), f__);                             \
        });                                                             \
  }                                                                     \
                                                                        \
  template <typename R, ENUM_PARAMS(N, typename P)>                     \
  operator std::function<R(ENUM_PARAMS(N, P))>() &&                     \
  {                                                                     \
    if (pid.isNone()) {                                                 \
      return std::function<R(ENUM_PARAMS(N, P))>(std::forward<F>(f));   \
    }                                                                   \
                                                                        \
    Option<UPID> pid_ = pid;                                            \
    F&& f_ = std::forward<F>(f);                                        \
                                                                        \
    return std::function<R(ENUM_PARAMS(N, P))>(                         \
        [=](ENUM_BINARY_PARAMS(N, P, p)) {                              \
          std::function<R()> f__([=]() {                                \
            return f_(ENUM_PARAMS(N, p));                               \
          });                                                           \
          return dispatch(pid_.get(), f__);                             \
        });                                                             \
  }                                                                     \
                                                                        \
  template <typename R, ENUM_PARAMS(N, typename P)>                     \
  operator lambda::CallableOnce<R(ENUM_PARAMS(N, P))>() &&              \
  {                                                                     \
    if (pid.isNone()) {                                                 \
      return lambda::CallableOnce<R(ENUM_PARAMS(N, P))>(                \
          std::forward<F>(f));                                          \
    }                                                                   \
                                                                        \
    Option<UPID> pid_ = pid;                                            \
                                                                        \
    return lambda::CallableOnce<R(ENUM_PARAMS(N, P))>(                  \
        lambda::partial(                                                \
            [pid_](typename std::decay<F>::type&& f_,                   \
                   ENUM_BINARY_PARAMS(N, P, &&p)) {                     \
              lambda::CallableOnce<R()> f__(                            \
                  lambda::partial(std::move(f_), ENUM(N, FORWARD, _))); \
              return dispatch(pid_.get(), std::move(f__));              \
        },                                                              \
        std::forward<F>(f),                                             \
        ENUM(N, PLACEHOLDER, _)));                                      \
  }

  REPEAT_FROM_TO(1, 3, TEMPLATE, _) // Args A0 -> A1.
#undef TEMPLATE

#undef FORWARD
#undef PLACEHOLDER

private:
  friend class Executor;

  template <typename G>
  friend _Deferred<G> defer(const UPID& pid, G&& g);

// This assumes type and variable base names are `A` and `a` respectively.
#define FORWARD(Z, N, DATA) std::forward<A ## N>(a ## N)

#define TEMPLATE(Z, N, DATA)                                             \
  template <typename T,                                                  \
            ENUM_PARAMS(N, typename P),                                  \
            ENUM_PARAMS(N, typename A)>                                  \
  friend auto defer(const PID<T>& pid,                                   \
                    void (T::*method)(ENUM_PARAMS(N, P)),                \
                    ENUM_BINARY_PARAMS(N, A, &&a))                       \
    -> _Deferred<decltype(                                               \
           lambda::partial(                                              \
               &std::function<void(ENUM_PARAMS(N, P))>::operator(),      \
               std::function<void(ENUM_PARAMS(N, P))>(),                 \
               ENUM(N, FORWARD, _)))>;

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

#define TEMPLATE(Z, N, DATA)                                             \
  template <typename R,                                                  \
            typename T,                                                  \
            ENUM_PARAMS(N, typename P),                                  \
            ENUM_PARAMS(N, typename A)>                                  \
  friend auto defer(const PID<T>& pid,                                   \
                    Future<R> (T::*method)(ENUM_PARAMS(N, P)),           \
                    ENUM_BINARY_PARAMS(N, A, &&a))                       \
    -> _Deferred<decltype(                                               \
           lambda::partial(                                              \
               &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(), \
               std::function<Future<R>(ENUM_PARAMS(N, P))>(),            \
               ENUM(N, FORWARD, _)))>;

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

#define TEMPLATE(Z, N, DATA)                                             \
  template <typename R,                                                  \
            typename T,                                                  \
            ENUM_PARAMS(N, typename P),                                  \
            ENUM_PARAMS(N, typename A)>                                  \
  friend auto defer(const PID<T>& pid,                                   \
                    R (T::*method)(ENUM_PARAMS(N, P)),                   \
                    ENUM_BINARY_PARAMS(N, A, &&a))                       \
    -> _Deferred<decltype(                                               \
         lambda::partial(                                                \
           &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(),     \
           std::function<Future<R>(ENUM_PARAMS(N, P))>(),                \
           ENUM(N, FORWARD, _)))>;

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE
#undef FORWARD

  _Deferred(const UPID& pid, F&& f) : pid(pid), f(std::forward<F>(f)) {}

  /*implicit*/ _Deferred(F&& f) : f(std::forward<F>(f)) {}

  Option<UPID> pid;
  F f;
};

// This is a wrapper around AsyncExecutorProcess.
class AsyncExecutor
{
private:
  // Declare async functions as friends.
  template <typename F>
  friend Future<typename result_of<F()>::type> async(
      const F& f,
      typename std::enable_if<!std::is_void<typename result_of<F()>::type>::value>::type*); // NOLINT(whitespace/line_length)

  template <typename F>
  friend Future<Nothing> async(
      const F& f,
      typename std::enable_if<std::is_void<typename result_of<F()>::type>::value>::type*); // NOLINT(whitespace/line_length)

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  friend Future<typename result_of<F(ENUM_PARAMS(N, A))>::type> async( \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<!std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type*); /* NOLINT(whitespace/line_length) */ \
                                                                        \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  friend Future<Nothing> async(                                         \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type*); // NOLINT(whitespace/line_length)

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

  AsyncExecutor()
  {
    process = spawn(new AsyncExecutorProcess(), true); // Automatically GC.
  }

  virtual ~AsyncExecutor() {}

  // Not copyable, not assignable.
  AsyncExecutor(const AsyncExecutor&);
  AsyncExecutor& operator=(const AsyncExecutor&);

  template <typename F>
  Future<typename result_of<F()>::type> execute(
      const F& f,
      typename std::enable_if<!std::is_void<typename result_of<F()>::type>::value>::type* = nullptr) // NOLINT(whitespace/line_length)
  {
    // Need to disambiguate overloaded method.
    typename result_of<F()>::type(AsyncExecutorProcess::*method)(const F&) =
      &AsyncExecutorProcess::execute<F>;

    return dispatch(process, method, f);
  }

  template <typename F>
  Future<Nothing> execute(
      const F& f,
      typename std::enable_if<std::is_void<typename result_of<F()>::type>::value>::type* = nullptr) // NOLINT(whitespace/line_length)
  {
    // Need to disambiguate overloaded method.
    Nothing(AsyncExecutorProcess::*method)(const F&) =
      &AsyncExecutorProcess::execute<F>;

    return dispatch(process, method, f);
  }

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<typename result_of<F(ENUM_PARAMS(N, A))>::type> execute( \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<!std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type* = nullptr) /* NOLINT(whitespace/line_length) */ \
  {                                                                     \
    /* Need to disambiguate overloaded method. */                       \
    typename result_of<F(ENUM_PARAMS(N, A))>::type(AsyncExecutorProcess::*method)(const F&, ENUM_PARAMS(N, A)) = /* NOLINT(whitespace/line_length) */ \
      &AsyncExecutorProcess::execute<F, ENUM_PARAMS(N, A)>;             \
                                                                        \
    return dispatch(process, method, f, ENUM_PARAMS(N, a));             \
  }                                                                     \
                                                                        \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<Nothing> execute(                                              \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type* = nullptr) /* NOLINT(whitespace/line_length) */ \
  {                                                                     \
    /* Need to disambiguate overloaded method. */                       \
    Nothing(AsyncExecutorProcess::*method)(const F&, ENUM_PARAMS(N, A)) = \
      &AsyncExecutorProcess::execute<F, ENUM_PARAMS(N, A)>;             \
                                                                        \
    return dispatch(process, method, f, ENUM_PARAMS(N, a));             \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

  PID<AsyncExecutorProcess> process;
};


// Provides an abstraction for asynchronously executing a function.
template <typename F>
Future<typename result_of<F()>::type> async(
    const F& f,
    typename std::enable_if<!std::is_void<typename result_of<F()>::type>::value>::type*) // NOLINT(whitespace/line_length)
{
  return AsyncExecutor().execute(f);
}


template <typename F>
Future<Nothing> async(
    const F& f,
    typename std::enable_if<std::is_void<typename result_of<F()>::type>::value>::type*) // NOLINT(whitespace/line_length)
{
  return AsyncExecutor().execute(f);
}


#define TEMPLATE(Z, N, DATA)                                            \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<typename result_of<F(ENUM_PARAMS(N, A))>::type> async( \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<!std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type*) /* NOLINT(whitespace/line_length) */ \
  {                                                                     \
    return AsyncExecutor().execute(f, ENUM_PARAMS(N, a));               \
  }                                                                     \
                                                                        \
  template <typename F, ENUM_PARAMS(N, typename A)>                     \
  Future<Nothing> async(                                                \
      const F& f,                                                       \
      ENUM_BINARY_PARAMS(N, A, a),                                      \
      typename std::enable_if<std::is_void<typename result_of<F(ENUM_PARAMS(N, A))>::type>::value>::type*) /* NOLINT(whitespace/line_length) */ \
  {                                                                     \
    return AsyncExecutor().execute(f, ENUM_PARAMS(N, a));               \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE
  

// Provides an abstraction that can take a standard function object
// and defer or asynchronously execute it without needing a process.
// Each converted function object will get execute serially with respect
// to one another when invoked.
class Executor
{
public:
  Executor() : process(ID::generate("__executor__"))
  {
    spawn(process);
  }

  ~Executor()
  {
    terminate(process);
    wait(process);
  }

  void stop()
  {
    terminate(process);

    // TODO(benh): Note that this doesn't wait because that could
    // cause a deadlock ... thus, the semantics here are that no more
    // dispatches will occur after this function returns but one may
    // be occurring concurrently.
  }

  template <typename F>
  _Deferred<F> defer(F&& f)
  {
    return _Deferred<F>(process.self(), std::forward<F>(f));
  }


private:
  // Not copyable, not assignable.
  Executor(const Executor&);
  Executor& operator=(const Executor&);

  ProcessBase process;


public:
  template <
      typename F,
      typename R = typename result_of<F()>::type,
      typename std::enable_if<!std::is_void<R>::value, int>::type = 0>
  auto execute(F&& f)
    -> decltype(dispatch(process, std::function<R()>(std::forward<F>(f))))
  {
    // NOTE: Currently we cannot pass a mutable lambda into `dispatch()`
    // because it would be captured by copy, so we convert `f` into a
    // `std::function` to bypass this restriction.
    return dispatch(process, std::function<R()>(std::forward<F>(f)));
  }

  // NOTE: This overload for `void` returns `Future<Nothing>` so we can
  // chain. This follows the same behavior as `async()`.
  template <
      typename F,
      typename R = typename result_of<F()>::type,
      typename std::enable_if<std::is_void<R>::value, int>::type = 0>
  Future<Nothing> execute(F&& f)
  {
    // NOTE: Currently we cannot pass a mutable lambda into `dispatch()`
    // because it would be captured by copy, so we convert `f` into a
    // `std::function` to bypass this restriction. This wrapper also
    // avoids `f` being evaluated when it is a nested bind.
    // TODO(chhsiao): Capture `f` by forwarding once we switch to C++14.
    return dispatch(
        process,
        std::bind(
            [](const std::function<R()>& f_) { f_(); return Nothing(); },
            std::function<R()>(std::forward<F>(f))));
  }
};


// Per thread executor pointer. We use a pointer to lazily construct the
// actual executor.
extern thread_local Executor* _executor_;

#define __executor__                                                    \
  (_executor_ == nullptr ? _executor_ = new Executor() : _executor_)


// The defer mechanism is very similar to the dispatch mechanism (see
// dispatch.hpp), however, rather than scheduling the method to get
// invoked, the defer mechanism returns a 'Deferred' object that when
// invoked does the underlying dispatch.

// First, definitions of defer for methods returning void:

template <typename T>
Deferred<void()> defer(const PID<T>& pid, void (T::*method)())
{
  return Deferred<void()>([=]() { dispatch(pid, method); });
}


template <typename T>
Deferred<void()> defer(const Process<T>& process, void (T::*method)())
{
  return defer(process.self(), method);
}


template <typename T>
Deferred<void()> defer(const Process<T>* process, void (T::*method)())
{
  return defer(process->self(), method);
}


// Due to a bug (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933)
// with variadic templates and lambdas, we still need to do
// preprocessor expansions. In addition, due to a bug with clang (or
// libc++) we can't use std::bind with a std::function so we have to
// explicitly use the std::function<R(P...)>::operator() (see
// http://stackoverflow.com/questions/20097616/stdbind-to-a-stdfunction-crashes-with-clang).

// This assumes that type and variable base names are `A` and `a` respectively.
#define FORWARD_A(Z, N, DATA) std::forward<A ## N>(a ## N)

// This assumes that type and variable base names are `P` and `p` respectively.
#define FORWARD_P(Z, N, DATA) std::forward<P ## N>(p ## N)

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const PID<T>& pid,                                         \
             void (T::*method)(ENUM_PARAMS(N, P)),                      \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> _Deferred<decltype(                                              \
         lambda::partial(                                               \
           &std::function<void(ENUM_PARAMS(N, P))>::operator(),         \
           std::function<void(ENUM_PARAMS(N, P))>(),                    \
           ENUM(N, FORWARD_A, _)))>                                     \
  {                                                                     \
    std::function<void(ENUM_PARAMS(N, P))> f(                           \
        [=](ENUM_BINARY_PARAMS(N, P, &&p)) {                            \
          dispatch(pid, method, ENUM(N, FORWARD_P, _));                 \
        });                                                             \
    return lambda::partial(                                             \
        &std::function<void(ENUM_PARAMS(N, P))>::operator(),            \
        std::move(f),                                                   \
        ENUM(N, FORWARD_A, _));                                         \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const Process<T>& process,                                 \
             void (T::*method)(ENUM_PARAMS(N, P)),                      \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> decltype(defer(process.self(), method, ENUM(N, FORWARD_A, _)))   \
  {                                                                     \
    return defer(process.self(), method, ENUM(N, FORWARD_A, _));        \
  }                                                                     \
                                                                        \
  template <typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const Process<T>* process,                                 \
             void (T::*method)(ENUM_PARAMS(N, P)),                      \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> decltype(defer(process->self(), method, ENUM(N, FORWARD_A, _)))  \
  {                                                                     \
    return defer(process->self(), method, ENUM(N, FORWARD_A, _));       \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE


// Next, definitions of defer for methods returning future:

template <typename R, typename T>
Deferred<Future<R>()> defer(const PID<T>& pid, Future<R> (T::*method)())
{
  return Deferred<Future<R>()>([=]() { return dispatch(pid, method); });
}

template <typename R, typename T>
Deferred<Future<R>()> defer(const Process<T>& process, Future<R> (T::*method)())
{
  return defer(process.self(), method);
}

template <typename R, typename T>
Deferred<Future<R>()> defer(const Process<T>* process, Future<R> (T::*method)())
{
  return defer(process->self(), method);
}

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const PID<T>& pid,                                         \
             Future<R> (T::*method)(ENUM_PARAMS(N, P)),                 \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> _Deferred<decltype(                                              \
         lambda::partial(                                               \
           &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(),    \
           std::function<Future<R>(ENUM_PARAMS(N, P))>(),               \
           ENUM(N, FORWARD_A, _)))>                                     \
  {                                                                     \
    std::function<Future<R>(ENUM_PARAMS(N, P))> f(                      \
        [=](ENUM_BINARY_PARAMS(N, P, &&p)) {                            \
          return dispatch(pid, method, ENUM(N, FORWARD_P, _));          \
        });                                                             \
    return lambda::partial(                                             \
        &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(),       \
        std::move(f),                                                   \
        ENUM(N, FORWARD_A, _));                                         \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const Process<T>& process,                                 \
             Future<R> (T::*method)(ENUM_PARAMS(N, P)),                 \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> decltype(defer(process.self(), method, ENUM(N, FORWARD_A, _)))   \
  {                                                                     \
    return defer(process.self(), method, ENUM(N, FORWARD_A, _));        \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const Process<T>* process,                                 \
             Future<R> (T::*method)(ENUM_PARAMS(N, P)),                 \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> decltype(defer(process->self(), method, ENUM(N, FORWARD_A, _)))  \
  {                                                                     \
    return defer(process->self(), method, ENUM(N, FORWARD_A, _));       \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE


// Finally, definitions of defer for methods returning a value:

template <typename R, typename T>
Deferred<Future<R>()> defer(const PID<T>& pid, R (T::*method)())
{
  return Deferred<Future<R>()>([=]() { return dispatch(pid, method); });
}

template <typename R, typename T>
Deferred<Future<R>()> defer(const Process<T>& process, R (T::*method)())
{
  return defer(process.self(), method);
}

template <typename R, typename T>
Deferred<Future<R>()> defer(const Process<T>* process, R (T::*method)())
{
  return defer(process->self(), method);
}

#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto defer(const PID<T>& pid,                                         \
             R (T::*method)(ENUM_PARAMS(N, P)),                         \
             ENUM_BINARY_PARAMS(N, A, &&a))                             \
    -> _Deferred<decltype(                                              \
         lambda::partial(                                               \
           &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(),    \
           std::function<Future<R>(ENUM_PARAMS(N, P))>(),               \
           ENUM(N, FORWARD_A, _)))>                                     \
  {                                                                     \
    std::function<Future<R>(ENUM_PARAMS(N, P))> f(                      \
        [=](ENUM_BINARY_PARAMS(N, P, &&p)) {                            \
          return dispatch(pid, method, ENUM(N, FORWARD_P, _));          \
        });                                                             \
    return lambda::partial(                                             \
        &std::function<Future<R>(ENUM_PARAMS(N, P))>::operator(),       \
        std::move(f),                                                   \
        ENUM(N, FORWARD_A, _));                                         \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto                                                                  \
  defer(const Process<T>& process,                                      \
        R (T::*method)(ENUM_PARAMS(N, P)),                              \
        ENUM_BINARY_PARAMS(N, A, &&a))                                  \
    -> decltype(defer(process.self(), method, ENUM(N, FORWARD_A, _)))   \
  {                                                                     \
    return defer(process.self(), method, ENUM(N, FORWARD_A, _));        \
  }                                                                     \
                                                                        \
  template <typename R,                                                 \
            typename T,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  auto                                                                  \
  defer(const Process<T>* process,                                      \
        R (T::*method)(ENUM_PARAMS(N, P)),                              \
        ENUM_BINARY_PARAMS(N, A, &&a))                                  \
    -> decltype(defer(process->self(), method, ENUM(N, FORWARD_A, _)))  \
  {                                                                     \
    return defer(process->self(), method, ENUM(N, FORWARD_A, _));       \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE

#undef FORWARD_A
#undef FORWARD_P


// Now we define defer calls for functors (with and without a PID):

template <typename F>
_Deferred<F> defer(const UPID& pid, F&& f)
{
  return _Deferred<F>(pid, std::forward<F>(f));
}


template <typename F>
_Deferred<F> defer(F&& f)
{
  if (__process__ != nullptr) {
    return defer(__process__->self(), std::forward<F>(f));
  }

  return __executor__->defer(std::forward<F>(f));
}

template <typename R>
class ThunkProcess : public Process<ThunkProcess<R>>
{
public:
  ThunkProcess(std::shared_ptr<lambda::function<R()>> _thunk,
               std::shared_ptr<Promise<R>> _promise)
    : ProcessBase(ID::generate("__thunk__")),
      thunk(_thunk),
      promise(_promise) {}

  virtual ~ThunkProcess() {}

protected:
  void serve(Event&& event) override
  {
    promise->set((*thunk)());
  }

private:
  std::shared_ptr<lambda::function<R()>> thunk;
  std::shared_ptr<Promise<R>> promise;
};

} // namespace internal {


template <typename R>
Future<R> run(R (*method)())
{
  std::shared_ptr<lambda::function<R()>> thunk(
      new lambda::function<R()>(
          lambda::bind(method)));

  std::shared_ptr<Promise<R>> promise(new Promise<R>());
  Future<R> future = promise->future();

  terminate(spawn(new internal::ThunkProcess<R>(thunk, promise), true));

  return future;
}


#define TEMPLATE(Z, N, DATA)                                            \
  template <typename R,                                                 \
            ENUM_PARAMS(N, typename P),                                 \
            ENUM_PARAMS(N, typename A)>                                 \
  Future<R> run(                                                        \
      R (*method)(ENUM_PARAMS(N, P)),                                   \
      ENUM_BINARY_PARAMS(N, A, a))                                      \
  {                                                                     \
    std::shared_ptr<lambda::function<R()>> thunk(                       \
        new lambda::function<R()>(                                      \
            lambda::bind(method, ENUM_PARAMS(N, a))));                  \
                                                                        \
    std::shared_ptr<Promise<R>> promise(new Promise<R>());              \
    Future<R> future = promise->future();                               \
                                                                        \
    terminate(spawn(new internal::ThunkProcess<R>(thunk, promise), true)); \
                                                                        \
    return future;                                                      \
  }

  REPEAT_FROM_TO(1, 13, TEMPLATE, _) // Args A0 -> A11.
#undef TEMPLATE
  

// Provides an asynchronous "loop" abstraction. This abstraction is
// helpful for code that would have synchronously been written as a
// loop but asynchronously ends up being a recursive set of functions
// which depending on the compiler may result in a stack overflow
// (i.e., a compiler that can't do sufficient tail call optimization
// may add stack frames for each recursive call).
//
// The loop abstraction takes an optional PID `pid` and uses it as the
// execution context to run the loop. The implementation does a
// `defer` on this `pid` to "pop" the stack when it needs to
// asynchronously recurse. This also lets callers synchronize
// execution with other code dispatching and deferring using `pid`. If
// `None` is passed for `pid` then no `defer` is done and the stack
// will still "pop" but be restarted from the execution context
// wherever the blocked future is completed. This is usually very safe
// when that blocked future will be completed by the IO thread, but
// should not be used if it's completed by another process (because
// you'll block that process until the next time the loop blocks).
//
// The two functions passed to the loop represent the loop "iterate"
// step and the loop "body" step respectively. Each invocation of
// "iterate" returns the next value and the "body" returns whether or
// not to continue looping (as well as any other processing necessary
// of course). You can think of this synchronously as:
//
//     bool condition = true;
//     do {
//       condition = body(iterate());
//     } while (condition);
//
// Asynchronously using recursion this might look like:
//
//     Future<Nothing> loop()
//     {
//       return iterate()
//         .then([](T t) {
//            return body(t)
//             .then([](bool condition) {
//               if (condition) {
//                 return loop();
//               } else {
//                 return Nothing();
//               }
//             });
//         });
//     }
//
// And asynchronously using `pid` as the execution context:
//
//     Future<Nothing> loop()
//     {
//       return iterate()
//         .then(defer(pid, [](T t) {
//            return body(t)
//             .then(defer(pid, [](bool condition) {
//               if (condition) {
//                 return loop();
//               } else {
//                 return Nothing();
//               }
//             }));
//         }));
//     }
//
// And now what this looks like using `loop`:
//
//     loop(pid,
//          []() {
//            return iterate();
//          },
//          [](T t) {
//            return body(t);
//          });
//
// One difference between the `loop` version of the "body" versus the
// other non-loop examples above is the return value is not `bool` or
// `Future<bool>` but rather `ControlFlow<V>` or
// `Future<ControlFlow<V>>`. This enables you to return values out of
// the loop via a `Break(...)`, for example:
//
//     loop(pid,
//          []() {
//            return iterate();
//          },
//          [](T t) {
//            if (finished(t)) {
//              return Break(SomeValue());
//            }
//            return Continue();
//          });
template <typename Iterate,
          typename Body,
          typename T = typename internal::unwrap<typename result_of<Iterate()>::type>::type, // NOLINT(whitespace/line_length)
          typename CF = typename internal::unwrap<typename result_of<Body(T)>::type>::type, // NOLINT(whitespace/line_length)
          typename V = typename CF::ValueType>
Future<V> loop(const Option<UPID>& pid, Iterate&& iterate, Body&& body);


// A helper for `loop` which creates a Process for us to provide an
// execution context for running the loop.
template <typename Iterate,
          typename Body,
          typename T = typename internal::unwrap<typename result_of<Iterate()>::type>::type, // NOLINT(whitespace/line_length)
          typename CF = typename internal::unwrap<typename result_of<Body(T)>::type>::type, // NOLINT(whitespace/line_length)
          typename V = typename CF::ValueType>
Future<V> loop(Iterate&& iterate, Body&& body)
{
  // Have libprocess own and free the new `ProcessBase`.
  UPID process = spawn(new ProcessBase(), true);

  return loop<Iterate, Body, T, CF, V>(
      process,
      std::forward<Iterate>(iterate),
      std::forward<Body>(body))
    .onAny([=]() {
      terminate(process);
      // NOTE: no need to `wait` or `delete` since the `spawn` above
      // put libprocess in control of garbage collection.
    });
}


// Generic "control flow" construct that is leveraged by
// implementations such as `loop`. At a high-level a `ControlFlow`
// represents some control flow statement such as `continue` or
// `break`, however, these statements can both have values or be
// value-less (i.e., these are meant to be composed "functionally" so
// the representation of `break` captures a value that "exits the
// current function" but the representation of `continue` does not).
//
// The pattern here is to define the type/representation of control
// flow statements within the `ControlFlow` class (e.g.,
// `ControlFlow::Continue` and `ControlFlow::Break`) but also provide
// "syntactic sugar" to make it easier to use at the call site (e.g.,
// the functions `Continue()` and `Break(...)`).
template <typename T>
class ControlFlow
{
public:
  using ValueType = T;

  enum class Statement
  {
    CONTINUE,
    BREAK
  };

  class Continue
  {
  public:
    Continue() = default;

    template <typename U>
    operator ControlFlow<U>() const
    {
      return ControlFlow<U>(ControlFlow<U>::Statement::CONTINUE, None());
    }
  };

  class Break
  {
  public:
    Break(T t) : t(std::move(t)) {}

    template <typename U>
    operator ControlFlow<U>() const &
    {
      return ControlFlow<U>(ControlFlow<U>::Statement::BREAK, t);
    }

    template <typename U>
    operator ControlFlow<U>() &&
    {
      return ControlFlow<U>(ControlFlow<U>::Statement::BREAK, std::move(t));
    }

  private:
    T t;
  };

  ControlFlow(Statement s, Option<T> t) : s(s), t(std::move(t)) {}

  Statement statement() const { return s; }

  T& value() & { return t.get(); }
  const T& value() const & { return t.get(); }
  T&& value() && { return t.get(); }
  const T&& value() const && { return t.get(); }

private:
  Statement s;
  Option<T> t;
};


// Provides "syntactic sugar" for creating a `ControlFlow::Continue`.
struct Continue
{
  Continue() = default;

  template <typename T>
  operator ControlFlow<T>() const
  {
    return typename ControlFlow<T>::Continue();
  }
};


// Provides "syntactic sugar" for creating a `ControlFlow::Break`.
template <typename T>
typename ControlFlow<typename std::decay<T>::type>::Break Break(T&& t)
{
  return typename ControlFlow<typename std::decay<T>::type>::Break(
      std::forward<T>(t));
}


inline ControlFlow<Nothing>::Break Break()
{
  return ControlFlow<Nothing>::Break(Nothing());
}


namespace internal {

template <typename Iterate, typename Body, typename T, typename R>
class Loop : public std::enable_shared_from_this<Loop<Iterate, Body, T, R>>
{
public:
  template <typename Iterate_, typename Body_>
  static std::shared_ptr<Loop> create(
      const Option<UPID>& pid,
      Iterate_&& iterate,
      Body_&& body)
  {
    return std::shared_ptr<Loop>(
        new Loop(
            pid,
            std::forward<Iterate_>(iterate),
            std::forward<Body_>(body)));
  }

  std::shared_ptr<Loop> shared()
  {
    // Must fully specify `shared_from_this` because we're templated.
    return std::enable_shared_from_this<Loop>::shared_from_this();
  }

  std::weak_ptr<Loop> weak()
  {
    return std::weak_ptr<Loop>(shared());
  }

  Future<R> start()
  {
    auto self = shared();
    auto weak_self = weak();

    // Propagating discards:
    //
    // When the caller does a discard we need to propagate it to
    // either the future returned from `iterate` or the future
    // returned from `body`. One easy way to do this would be to add
    // an `onAny` callback for every future returned from `iterate`
    // and `body`, but that would be a slow memory leak that would
    // grow over time, especially if the loop was actually
    // infinite. Instead, we capture the current future that needs to
    // be discarded within a `discard` function that we'll invoke when
    // we get a discard. Because there is a race setting the `discard`
    // function and reading it out to invoke we have to synchronize
    // access using a mutex. An alternative strategy would be to use
    // something like `atomic_load` and `atomic_store` with
    // `shared_ptr` so that we can swap the current future(s)
    // atomically.
    promise.future().onDiscard([weak_self]() {
      auto self = weak_self.lock();
      if (self) {
        // We need to make a copy of the current `discard` function so
        // that we can invoke it outside of the `synchronized` block
        // in the event that discarding invokes causes the `onAny`
        // callbacks that we have added in `run` to execute which may
        // deadlock attempting to re-acquire `mutex`!
        std::function<void()> f = []() {};
        synchronized (self->mutex) {
          f = self->discard;
        }
        f();
      }
    });

    if (pid.isSome()) {
      // Start the loop using `pid` as the execution context.
      dispatch(pid.get(), [self]() {
        self->run(self->iterate());
      });

      // TODO(benh): Link with `pid` so that we can discard or abandon
      // the promise in the event `pid` terminates and didn't discard
      // us so that we can avoid any leaks (memory or otherwise).
    } else {
      run(iterate());
    }

    return promise.future();
  }

  void run(Future<T> next)
  {
    auto self = shared();

    // Reset `discard` so that we're not delaying cleanup of any
    // captured futures longer than necessary.
    //
    // TODO(benh): Use `WeakFuture` in `discard` functions instead.
    synchronized (mutex) {
      discard = []() {};
    }

    while (next.isReady()) {
      Future<ControlFlow<R>> flow = body(next.get());
      if (flow.isReady()) {
        switch (flow->statement()) {
          case ControlFlow<R>::Statement::CONTINUE: {
            next = iterate();
            continue;
          }
          case ControlFlow<R>::Statement::BREAK: {
            promise.set(flow->value());
            return;
          }
        }
      } else {
        auto continuation = [self](const Future<ControlFlow<R>>& flow) {
          if (flow.isReady()) {
            switch (flow->statement()) {
              case ControlFlow<R>::Statement::CONTINUE: {
                self->run(self->iterate());
                break;
              }
              case ControlFlow<R>::Statement::BREAK: {
                self->promise.set(flow->value());
                break;
              }
            }
          } else if (flow.isFailed()) {
            self->promise.fail(flow.failure());
          } else if (flow.isDiscarded()) {
            self->promise.discard();
          }
        };

        if (pid.isSome()) {
          flow.onAny(defer(pid.get(), continuation));
        } else {
          flow.onAny(continuation);
        }

        if (!promise.future().hasDiscard()) {
          synchronized (mutex) {
            self->discard = [=]() mutable { flow.discard(); };
          }
        }

        // There's a race between when a discard occurs and the
        // `discard` function gets invoked and therefore we must
        // explicitly always do a discard. In addition, after a
        // discard occurs we'll need to explicitly do discards for
        // each new future that blocks.
        if (promise.future().hasDiscard()) {
          flow.discard();
        }

        return;
      }
    }

    auto continuation = [self](const Future<T>& next) {
      if (next.isReady()) {
        self->run(next);
      } else if (next.isFailed()) {
        self->promise.fail(next.failure());
      } else if (next.isDiscarded()) {
        self->promise.discard();
      }
    };

    if (pid.isSome()) {
      next.onAny(defer(pid.get(), continuation));
    } else {
      next.onAny(continuation);
    }

    if (!promise.future().hasDiscard()) {
      synchronized (mutex) {
        discard = [=]() mutable { next.discard(); };
      }
    }

    // See comment above as to why we need to explicitly discard
    // regardless of the path the if statement took above.
    if (promise.future().hasDiscard()) {
      next.discard();
    }
  }

protected:
  Loop(const Option<UPID>& pid, const Iterate& iterate, const Body& body)
    : pid(pid), iterate(iterate), body(body) {}

  Loop(const Option<UPID>& pid, Iterate&& iterate, Body&& body)
    : pid(pid), iterate(std::move(iterate)), body(std::move(body)) {}

private:
  const Option<UPID> pid;
  Iterate iterate;
  Body body;
  Promise<R> promise;

  // In order to discard the loop safely we capture the future that
  // needs to be discarded within the `discard` function and reading
  // and writing that function with a mutex.
  std::mutex mutex;
  std::function<void()> discard = []() {};
};

} // namespace internal {


template <typename Iterate, typename Body, typename T, typename CF, typename V>
Future<V> loop(const Option<UPID>& pid, Iterate&& iterate, Body&& body)
{
  using Loop = internal::Loop<
    typename std::decay<Iterate>::type,
    typename std::decay<Body>::type,
    T,
    V>;

  std::shared_ptr<Loop> loop = Loop::create(
      pid,
      std::forward<Iterate>(iterate),
      std::forward<Body>(body));

  return loop->start();
}

} // namespace process
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_PROCESS_H_