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

// mesos/3rdparty/libprocess/include/process/event.hpp
// mesos/3rdparty/libprocess/src/event_queue.hpp

#ifndef BUBBLEFS_UTILS_MESOS_PROCESS_EVENT_QUEUE_H_
#define BUBBLEFS_UTILS_MESOS_PROCESS_EVENT_QUEUE_H_

#include <deque>
#include <mutex>
#include <string>

#ifdef LOCK_FREE_EVENT_QUEUE
#include <concurrentqueue.h>
#endif // LOCK_FREE_EVENT_QUEUE

#include <process/event.hpp>
#include <process/http.hpp>

#include <stout/json.hpp>
#include <stout/stringify.hpp>
#include <stout/synchronized.hpp>

// Process Event Queue (aka Mailbox)
// Each process has a queue of incoming Event's that it processes one at a time. 
// Other actor implementations often call this queue the "mailbox".
// There are 5 different kinds of events that can be enqueued for a process:
// 1. MessageEvent: a Message has been received.
// 2. DispatchEvent: a method on the process has been "dispatched".
// 3. HttpEvent: an http::Request has been received.
// 4. ExitedEvent: another process which has been linked has terminated.
// 5. TerminateEvent: the process has been requested to terminate.
// An event is serviced one at a time by invoking the process' serve() method 
// which by default invokes the process' visit() method corresponding to the underlying event type. 
// Most actors don't need to override the implementation of serve() or visit() 
// but can rely on higher-level abstractions that simplify serving the event 
// (e.g., route(), which make it easy to set up handlers for an HttpEvent, discussed below in HTTP).

namespace bubblefs {
namespace mymesos {
namespace process {

// Forward declarations.
class ProcessBase;
struct MessageEvent;
struct DispatchEvent;
struct HttpEvent;
struct ExitedEvent;
struct TerminateEvent;


struct EventVisitor
{
  virtual ~EventVisitor() {}
  virtual void visit(const MessageEvent&) {}
  virtual void visit(const DispatchEvent&) {}
  virtual void visit(const HttpEvent&) {}
  virtual void visit(const ExitedEvent&) {}
  virtual void visit(const TerminateEvent&) {}
};


struct EventConsumer
{
  virtual ~EventConsumer() {}
  virtual void consume(MessageEvent&&) {}
  virtual void consume(DispatchEvent&&) {}
  virtual void consume(HttpEvent&&) {}
  virtual void consume(ExitedEvent&&) {}
  virtual void consume(TerminateEvent&&) {}
};


struct Event
{
  virtual ~Event() {}

  virtual void visit(EventVisitor* visitor) const = 0;
  virtual void consume(EventConsumer* consumer) && = 0;

  template <typename T>
  bool is() const
  {
    bool result = false;
    struct IsVisitor : EventVisitor
    {
      explicit IsVisitor(bool* _result) : result(_result) {}
      virtual void visit(const T&) { *result = true; }
      bool* result;
    } visitor(&result);
    visit(&visitor);
    return result;
  }

  template <typename T>
  const T& as() const
  {
    const T* result = nullptr;
    struct AsVisitor : EventVisitor
    {
      explicit AsVisitor(const T** _result) : result(_result) {}
      virtual void visit(const T& t) { *result = &t; }
      const T** result;
    } visitor(&result);
    visit(&visitor);
    if (result == nullptr) {
      ABORT("Attempting to \"cast\" event incorrectly!");
    }
    return *result;
  }

  // JSON representation for an Event.
  operator JSON::Object() const;
};

// A MessageEvent gets enqueued for a process when it gets sent a Message, 
// either locally or remotely. You use send() to send a message from within a process 
// and post() to send a message from outside a process. 
// A post() sends a message without a return address 
// because there is no process to reply to. 
// PID<ServerProcess> server = spawn(new ServerProcess(), true);
// PID<ClientProcess> client = spawn(new ClientProcess(server), true);
// wait(server);
// wait(client);

struct MessageEvent : Event
{
  explicit MessageEvent(Message&& _message)
    : message(std::move(_message)) {}

  MessageEvent(
      const UPID& from,
      const UPID& to,
      const std::string& name,
      const char* data,
      size_t length)
    : message{name, from, to, std::string(data, length)} {}

  MessageEvent(
      const UPID& from,
      const UPID& to,
      std::string&& name,
      std::string&& data)
    : message{std::move(name), from, to, std::move(data)} {}

  MessageEvent(MessageEvent&& that) = default;
  MessageEvent(const MessageEvent& that) = delete;
  MessageEvent& operator=(MessageEvent&&) = default;
  MessageEvent& operator=(const MessageEvent&) = delete;

  void visit(EventVisitor* visitor) const override
  {
    visitor->visit(*this);
  }

  void consume(EventConsumer* consumer) && override
  {
    consumer->consume(std::move(*this));
  }

  Message message;
};


struct HttpEvent : Event
{
  HttpEvent(
      http::Request* _request,
      Promise<http::Response>* _response)
    : request(_request),
      response(_response) {}

  HttpEvent(HttpEvent&&) = default;
  HttpEvent(const HttpEvent&) = delete;
  HttpEvent& operator=(HttpEvent&&) = default;
  HttpEvent& operator=(const HttpEvent&) = delete;

  virtual ~HttpEvent()
  {
    if (response) {
      // Fail the response in case it wasn't set.
      response->set(http::InternalServerError());
    }
  }

  void visit(EventVisitor* visitor) const override
  {
    visitor->visit(*this);
  }

  void consume(EventConsumer* consumer) && override
  {
    consumer->consume(std::move(*this));
  }

  std::unique_ptr<http::Request> request;
  std::unique_ptr<Promise<http::Response>> response;
};


struct DispatchEvent : Event
{
  DispatchEvent(
      const UPID& _pid,
      std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> _f,
      const Option<const std::type_info*>& _functionType)
    : pid(_pid),
      f(std::move(_f)),
      functionType(_functionType)
  {}

  DispatchEvent(DispatchEvent&&) = default;
  DispatchEvent(const DispatchEvent&) = delete;
  DispatchEvent& operator=(DispatchEvent&&) = default;
  DispatchEvent& operator=(const DispatchEvent&) = delete;

  void visit(EventVisitor* visitor) const override
  {
    visitor->visit(*this);
  }

  void consume(EventConsumer* consumer) && override
  {
    consumer->consume(std::move(*this));
  }

  // PID receiving the dispatch.
  UPID pid;

  // Function to get invoked as a result of this dispatch event.
  std::unique_ptr<lambda::CallableOnce<void(ProcessBase*)>> f;

  Option<const std::type_info*> functionType;
};


struct ExitedEvent : Event
{
  explicit ExitedEvent(const UPID& _pid)
    : pid(_pid) {}

  ExitedEvent(ExitedEvent&&) = default;
  ExitedEvent(const ExitedEvent&) = delete;
  ExitedEvent& operator=(ExitedEvent&&) = default;
  ExitedEvent& operator=(const ExitedEvent&) = delete;

  void visit(EventVisitor* visitor) const override
  {
    visitor->visit(*this);
  }

  void consume(EventConsumer* consumer) && override
  {
    consumer->consume(std::move(*this));
  }

  UPID pid;
};


struct TerminateEvent : Event
{
  TerminateEvent(const UPID& _from, bool _inject)
    : from(_from), inject(_inject) {}

  TerminateEvent(TerminateEvent&&) = default;
  TerminateEvent(const TerminateEvent&) = delete;
  TerminateEvent& operator=(TerminateEvent&&) = default;
  TerminateEvent& operator=(const TerminateEvent&) = delete;

  void visit(EventVisitor* visitor) const override
  {
    visitor->visit(*this);
  }

  void consume(EventConsumer* consumer) && override
  {
    consumer->consume(std::move(*this));
  }

  UPID from;
  bool inject;
};


inline Event::operator JSON::Object() const
{
  JSON::Object object;

  struct Visitor : EventVisitor
  {
    explicit Visitor(JSON::Object* _object) : object(_object) {}

    virtual void visit(const MessageEvent& event)
    {
      object->values["type"] = "MESSAGE";

      const Message& message = event.message;

      object->values["name"] = message.name;
      object->values["from"] = stringify(message.from);
      object->values["to"] = stringify(message.to);
      object->values["body"] = message.body;
    }

    virtual void visit(const HttpEvent& event)
    {
      object->values["type"] = "HTTP";

      const http::Request& request = *event.request;

      object->values["method"] = request.method;
      object->values["url"] = stringify(request.url);
    }

    virtual void visit(const DispatchEvent& event)
    {
      object->values["type"] = "DISPATCH";
    }

    virtual void visit(const ExitedEvent& event)
    {
      object->values["type"] = "EXITED";
    }

    virtual void visit(const TerminateEvent& event)
    {
      object->values["type"] = "TERMINATE";
    }

    JSON::Object* object;
  } visitor(&object);

  visit(&visitor);

  return object;
}  
  
// A _multiple_ producer (MP) _single_ consumer (SC) event queue for a
// process. Note that we don't _enforce_ the MP/SC semantics during
// runtime but we have explicitly separated out the `Producer`
// interface and the `Consumer` interface in order to help avoid
// incorrect usage.
//
// Notable semantics:
//
//   * Consumers _must_ call `empty()` before calling
//     `dequeue()`. Failing to do so may result in undefined behavior.
//
//   * After a consumer calls `decomission()` they _must_ not call any
//     thing else (not even `empty()` and especially not
//     `dequeue()`). Doing so is undefined behavior.
//
// Notes on the lock-free implementation:
//
// The SC requirement is necessary for the lock-free implementation
// because the underlying queue does not provide linearizability which
// means events can be dequeued "out of order". Usually this is not a
// problem, after all, in most circumstances we won't know the order
// in which events might be enqueued in the first place. However, this
// can be a very bad problem if a single process attempts to enqueue
// two events in a different process AND THOSE EVENTS ARE
// REORDERED. To ensure this will never be the case we give every
// event a sequence number. That way an event from the same process
// will always have a happens-before relationship with respect to the
// events that they enqueue because they'll have distinct sequence
// numbers.
//
// This makes the consumer implementation more difficult because the
// consumer might need to "reorder" events as it reads them out. To do
// this efficiently we require only a single consumer, which fits well
// into the actor model because there will only ever be a single
// thread consuming an actors events at a time.
class EventQueue
{
public:
  EventQueue() : producer(this), consumer(this) {}

  class Producer
  {
  public:
    void enqueue(Event* event) { queue->enqueue(event); }

  private:
    friend class EventQueue;

    Producer(EventQueue* queue) : queue(queue) {}

    EventQueue* queue;
  } producer;

  class Consumer
  {
  public:
    Event* dequeue() { return queue->dequeue(); }
    bool empty() { return queue->empty(); }
    void decomission() { queue->decomission(); }
    template <typename T>
    size_t count() { return queue->count<T>(); }
    operator JSON::Array() { return queue->operator JSON::Array(); }

  private:
    friend class EventQueue;

    Consumer(EventQueue* queue) : queue(queue) {}

    EventQueue* queue;
  } consumer;

private:
  friend class Producer;
  friend class Consumer;

#ifndef LOCK_FREE_EVENT_QUEUE
  void enqueue(Event* event)
  {
    bool enqueued = false;
    synchronized (mutex) {
      if (comissioned) {
        events.push_back(event);
        enqueued = true;
      }
    }

    if (!enqueued) {
      delete event;
    }
  }

  Event* dequeue()
  {
    Event* event = nullptr;

    synchronized (mutex) {
      if (events.size() > 0) {
        Event* event = events.front();
        events.pop_front();
        return event;
      }
    }

    // Semantics are the consumer _must_ call `empty()` before calling
    // `dequeue()` which means an event must be present.
    return CHECK_NOTNULL(event);
  }

  bool empty()
  {
    synchronized (mutex) {
      return events.size() == 0;
    }
  }

  void decomission()
  {
    synchronized (mutex) {
      comissioned = false;
      while (!events.empty()) {
        Event* event = events.front();
        events.pop_front();
        delete event;
      }
    }
  }

  template <typename T>
  size_t count()
  {
    synchronized (mutex) {
      return std::count_if(
          events.begin(),
          events.end(),
          [](const Event* event) {
            return event->is<T>();
          });
    }
  }

  operator JSON::Array()
  {
    JSON::Array array;
    synchronized (mutex) {
      foreach (Event* event, events) {
        array.values.push_back(JSON::Object(*event));
      }
    }
    return array;
  }

  std::mutex mutex;
  std::deque<Event*> events;
  bool comissioned = true;
#else // LOCK_FREE_EVENT_QUEUE
  void enqueue(Event* event)
  {
    Item item = {sequence.fetch_add(1), event};
    if (comissioned.load()) {
      queue.enqueue(std::move(item));
    } else {
      sequence.fetch_sub(1);
      delete event;
    }
  }

  Event* dequeue()
  {
    // NOTE: for performance reasons we don't check `comissioned` here
    // so it's possible that we'll loop forever if a consumer called
    // `decomission()` and then subsequently called `dequeue()`.
    Event* event = nullptr;
    do {
      // Given the nature of the concurrent queue implementation it's
      // possible that we'll need to try to dequeue multiple times
      // until it returns an event even though we know there is an
      // event because the semantics are that we shouldn't call
      // `dequeue()` before calling `empty()`.
      event = try_dequeue();
    } while (event == nullptr);
    return event;
  }

  bool empty()
  {
    // NOTE: for performance reasons we don't check `comissioned` here
    // so it's possible that we'll return true when in fact we've been
    // decomissioned and you shouldn't attempt to dequeue anything.
    return (sequence.load() - next) == 0;
  }

  void decomission()
  {
    comissioned.store(true);
    while (!empty()) {
      // NOTE: we use `try_dequeue()` here because we might be racing
      // with `enqueue()` where they've already incremented `sequence`
      // so we think there are more items to dequeue but they aren't
      // actually going to enqueue anything because they've since seen
      // `comissioned` is true. We'll attempt to dequeue with
      // `try_dequeue()` and eventually they'll decrement `sequence`
      // and so `empty()` will return true and we'll bail.
      Event* event = try_dequeue();
      if (event != nullptr) {
        delete event;
      }
    }
  }

  template <typename T>
  size_t count()
  {
    // Try and dequeue more elements first!
    queue.try_dequeue_bulk(std::back_inserter(items), SIZE_MAX);

    return std::count_if(
        items.begin(),
        items.end(),
        [](const Item& item) {
          if (item.event != nullptr) {
            return item.event->is<T>();
          }
          return false;
        });
  }

  operator JSON::Array()
  {
    // Try and dequeue more elements first!
    queue.try_dequeue_bulk(std::back_inserter(items), SIZE_MAX);

    JSON::Array array;
    foreach (const Item& item, items) {
      if (item.event != nullptr) {
        array.values.push_back(JSON::Object(*item.event));
      }
    }

    return array;
  }

  struct Item
  {
    uint64_t sequence;
    Event* event;
  };

  Event* try_dequeue()
  {
    // The general algoritm here is as follows: we bulk dequeue as
    // many items from the concurrent queue as possible. We then look
    // for the `next` item in the sequence hoping that it's at the
    // beginning of `items` but because the `queue` is not
    // linearizable it might be "out of order". If we find it out of
    // order we effectively dequeue it but leave it in `items` so as
    // not to incur any costly rearrangements/compactions in
    // `items`. We'll later pop the out of order items once they get
    // to the front.

    // Start by popping any items that we effectively dequeued but
    // didn't remove from `items` so as not to incur costly
    // rearragements/compactions.
    while (!items.empty() && next > items.front().sequence) {
      items.pop_front();
    }

    // Optimistically let's hope that the next item is at the front of
    // `item`. If so, pop the item, increment `next`, and return the
    // event.
    if (!items.empty() && items.front().sequence == next) {
      Event* event = items.front().event;
      items.pop_front();
      next += 1;
      return event;
    }

    size_t index = 0;

    do {
      // Now look for a potentially out of order item. If found,
      //  signifiy the item has been dequeued by nulling the event
      //  (necessary for the implementation of `count()` and `operator
      //  JSON::Array()`) and return the event.
      for (; index < items.size(); index++) {
        if (items[index].sequence == next) {
          Event* event = items[index].event;
          items[index].event = nullptr;
          next += 1;
          return event;
        }
      }

      // If we can bulk dequeue more items then keep looking for the
      // out of order event!
      //
      // NOTE: we use the _small_ value of `4` to dequeue here since
      // in the presence of enough events being enqueued we could end
      // up spending a LONG time dequeuing here! Since the next event
      // in the sequence should really be close to the top of the
      // queue we use a small value to dequeue.
      //
      // The intuition here is this: the faster we can return the next
      // event the faster that event can get processed and the faster
      // it might generate other events that can get processed in
      // parallel by other threads and the more work we get done.
    } while (queue.try_dequeue_bulk(std::back_inserter(items), 4) != 0);

    return nullptr;
  }

  // Underlying queue of items.
  moodycamel::ConcurrentQueue<Item> queue;

  // Counter to represent the item sequence. Note that we use a
  // unsigned 64-bit integer which means that even if we were adding
  // one item to the queue every nanosecond we'd be able to run for
  // 18,446,744,073,709,551,615 nanoseconds or ~585 years! ;-)
  std::atomic<uint64_t> sequence = ATOMIC_VAR_INIT(0);

  // Counter to represent the next item we expect to dequeue. Note
  // that we don't need to make this be atomic because only a single
  // consumer is ever reading or writing this variable!
  uint64_t next = 0;

  // Collection of bulk dequeued items that may be out of order. Note
  // that like `next` this will only ever be read/written by a single
  // consumer.
  //
  // The use of a deque was explicit because it is implemented as an
  // array of arrays (or vector of vectors) which usually gives good
  // performance for appending to the back and popping from the front
  // which is exactly what we need to do. To avoid any performance
  // issues that might be incurred we do not remove any items from the
  // middle of the deque (see comments in `try_dequeue()` above for
  // more details).
  std::deque<Item> items;

  // Whether or not the event queue has been decomissioned. This must
  // be atomic as it can be read by a producer even though it's only
  // written by a consumer.
  std::atomic<bool> comissioned = ATOMIC_VAR_INIT(true);
#endif // LOCK_FREE_EVENT_QUEUE
};

} // namespace process
} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_PROCESS_EVENT_QUEUE_H_