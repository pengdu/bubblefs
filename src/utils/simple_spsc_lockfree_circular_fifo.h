/* 
* Not any company's property but Public-Domain
* Do with source-code as you will. No requirement to keep this
* header if need to use it/change it/ or do whatever with it
*
* Note that there is No guarantee that this code will work 
* and I take no responsibility for this code and any problems you
* might get if using it. 
*
* Code & platform dependent issues with it was originally 
* published at http://www.kjellkod.cc/threadsafecircularqueue
* 2012-16-19  @author Kjell Hedström, hedstrom@kjellkod.cc */

// should be mentioned the thinking of what goes where
// it is a "controversy" whether what is tail and what is head
// http://en.wikipedia.org/wiki/FIFO#Head_or_tail_first

// KjellKod/lock-free-wait-free-circularfifo/src/circularfifo_memory_relaxed_aquire_release_padded.hpp

#ifndef BUBBLEFS_UTILS_SIMPLE_SPSC_LOCKFREE_CIRCULAR_FIFO_H_
#define BUBBLEFS_UTILS_SIMPLE_SPSC_LOCKFREE_CIRCULAR_FIFO_H_

#include <atomic>
#include <cstddef>

namespace bubblefs {
namespace mysimple {
  
template<typename Element, size_t Size> 
class SPSCLockFreeCircularFifo{
public:
  typedef char cache_line[64];
  enum { Capacity = Size + 1 } __attribute__((aligned(64)));

  SPSCLockFreeCircularFifo() : _tail(0), _head(0){}   
  virtual ~SPSCLockFreeCircularFifo() {}

  bool push(const Element& item); // pushByMOve?
  bool pop(Element& item);

  bool wasEmpty() const;
  bool wasFull() const;
  bool isLockFree() const;

private:
  size_t increment(size_t idx) const { return (idx + 1) % Capacity; }

  cache_line _pad_storage;
  /*alignas(64)*/ Element _array[Capacity];
  cache_line _pad_tail;
  /*alignas(64)*/ std::atomic <size_t>  _tail;  
  cache_line  _pad_head;
  /*alignas(64)*/ std::atomic<size_t>   _head; // head(output) index
};

template<typename Element, size_t Size>
bool SPSCLockFreeCircularFifo<Element, Size>::push(const Element& item)
{       
  // Since the Producer thread is the only thread calling push(),
  // it is guaranteed that the tail value will be the latest.
  // No cross-thread synchronization is needed for the load.
  const auto current_tail = _tail.load(std::memory_order_relaxed); 
  const auto next_tail = increment(current_tail); 
  // Like memory read barrier, following reads get the new updates, not be re-ordered before the atomic ops.
  if(next_tail != _head.load(std::memory_order_acquire))                           
  {     
    // The item is saved in the position pointed to by the not-yet-synchronized current_tail. 
    // This happens-before the release_store.
    _array[current_tail] = item;
    // Like memory write barrier, previous writes set the new updates, not be re-ordered after the atomic ops.
    _tail.store(next_tail, std::memory_order_release); 
    return true;
  }
  
  return false; // full queue

}

// Pop by Consumer can only update the head (load with relaxed, store with release)
//     the tail must be accessed with at least aquire
template<typename Element, size_t Size>
bool SPSCLockFreeCircularFifo<Element, Size>::pop(Element& item)
{
  // Since the Consumer thread is the only thread calling pop(),
  // it is guaranteed that the head value will be the latest. 
  // No cross-thread synchronization is needed for the load.
  const auto current_head = _head.load(std::memory_order_relaxed);
  // Like memory read barrier.
  if(current_head == _tail.load(std::memory_order_acquire)) 
    return false; // empty queue
  // This happens-before the release_store.
  item = _array[current_head];
  // Like memory write barrier.
  _head.store(increment(current_head), std::memory_order_release); 
  return true;
}

template<typename Element, size_t Size>
bool SPSCLockFreeCircularFifo<Element, Size>::wasEmpty() const
{
  // snapshot with acceptance of that this comparison operation is not atomic
  return (_head.load() == _tail.load()); 
}

// snapshot with acceptance that this comparison is not atomic
template<typename Element, size_t Size>
bool SPSCLockFreeCircularFifo<Element, Size>::wasFull() const
{
  const auto next_tail = increment(_tail.load()); // aquire, we dont know who call
  return (next_tail == _head.load());
}

template<typename Element, size_t Size>
bool SPSCLockFreeCircularFifo<Element, Size>::isLockFree() const
{
  return (_tail.is_lock_free() && _head.is_lock_free());
}

} // namespace mysimple
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SIMPLE_SPSC_LOCKFREE_CIRCULAR_FIFO_H_