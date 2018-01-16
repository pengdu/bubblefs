/**
 *  Copyright (c) 2015 by Contributors
 */

// ps-lite/include/ps/internal/customer.h
// ps-lite/src/customer.cc

#ifndef BUBBLEFS_UTILS_PSLITE_CUSTOMER_H_
#define BUBBLEFS_UTILS_PSLITE_CUSTOMER_H_

#include <mutex>
#include <vector>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <memory>
#include "utils/pslite_message.h"
#include "utils/pslite_threadsafe_queue.h"
#include "utils/pslite_postoffice.h"

namespace bubblefs {
namespace mypslite {

/**
 * \brief The object for communication.
 *
 * As a sender, a customer tracks the responses for each request sent.
 *
 * It has its own receiving thread which is able to process any message received
 * from a remote node with `msg.meta.customer_id` equal to this customer's id
 */
class Customer {
 public:
  /**
   * \brief the handle for a received message
   * \param recved the received message
   */
  using RecvHandle = std::function<void(const Message& recved)>;

  /**
   * \brief constructor
   * \param id the unique id, any received message with
   * \param recv_handle the functino for processing a received message
   */
  Customer(int id, const RecvHandle& recv_handle);

  /**
   * \brief desconstructor
   */
  ~Customer();

  /**
   * \brief return the unique id
   */
  int id() { return id_; }

  /**
   * \brief get a timestamp for a new request. threadsafe
   * \param recver the receive node id of this request
   * \return the timestamp of this request
   */
  int NewRequest(int recver);


  /**
   * \brief wait until the request is finished. threadsafe
   * \param timestamp the timestamp of the request
   */
  void WaitRequest(int timestamp);

  /**
   * \brief return the number of responses received for the request. threadsafe
   * \param timestamp the timestamp of the request
   */
  int NumResponse(int timestamp);

  /**
   * \brief add a number of responses to timestamp
   */
  void AddResponse(int timestamp, int num = 1);

  /**
   * \brief accept a received message from \ref Van. threadsafe
   * \param recved the received the message
   */
  void Accept(const Message& recved) { recv_queue_.Push(recved); }

 private:
  /**
   * \brief the thread function
   */
  void Receiving();

  int id_;

  RecvHandle recv_handle_;
  ThreadsafeQueue<Message> recv_queue_;
  std::unique_ptr<std::thread> recv_thread_;

  std::mutex tracker_mu_;
  std::condition_variable tracker_cond_;
  std::vector<std::pair<int, int>> tracker_;

  DISALLOW_COPY_AND_ASSIGN(Customer);
};

const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();

Customer::Customer(int id, const Customer::RecvHandle& recv_handle)
    : id_(id), recv_handle_(recv_handle) {
  Postoffice::Get()->AddCustomer(this);
  recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
}

Customer::~Customer() {
  Postoffice::Get()->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  recv_thread_->join();
}

int Customer::NewRequest(int recver) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  int num = Postoffice::Get()->GetNodeIDs(recver).size();
  tracker_.push_back(std::make_pair(num, 0));
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp].first == tracker_[timestamp].second;
    });
}

int Customer::NumResponse(int timestamp) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp].second;
}

void Customer::AddResponse(int timestamp, int num) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  tracker_[timestamp].second += num;
}

void Customer::Receiving() {
  while (true) {
    Message recv;
    recv_queue_.WaitAndPop(&recv);
    if (!recv.meta.control.empty() &&
        recv.meta.control.cmd == Control::TERMINATE) {
      break;
    }
    recv_handle_(recv);
    if (!recv.meta.request) {
      std::lock_guard<std::mutex> lk(tracker_mu_);
      tracker_[recv.meta.timestamp].second++;
      tracker_cond_.notify_all();
    }
  }
}

}  // namespace mypslite
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PSLITE_CUSTOMER_H_