/**
 *  Copyright (c) 2015 by Contributors
 */

// ps-lite/include/ps/internal/postoffice.h
// // ps-lite/src/postoffice.cc

#ifndef BUBBLEFS_UTILS_PSLITE_POSTOFFICE_H_
#define BUBBLEFS_UTILS_PSLITE_POSTOFFICE_H_

#include <mutex>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <thread>
#include <chrono>
#include "platform/base_error.h"
#include "utils/pslite_message.h"
#include "utils/pslite_base.h"
#include "utils/pslite_customer.h"
#include "utils/pslite_env.h"
#include "utils/pslite_range.h"
#include "utils/pslite_van.h"

/*
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &
 
# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    ${bin} ${arg} &
done
 
# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} ${arg} &
done
*/

namespace bubblefs {
namespace mypslite {
/**
 * \brief the center of the system
 */
class Postoffice {
 public:
  /**
   * \brief return the singleton object
   */
  static Postoffice* Get() {
    static Postoffice e; return &e;
  }
  /** \brief get the van */
  Van* van() { return van_; }
  /**
   * \brief start the system
   *
   * This function will block until every nodes are started.
   * \param argv0 the program name, used for logging.
   * \param do_barrier whether to block until every nodes are started.
   */
  void Start(const char* argv0, const bool do_barrier);
  /**
   * \brief terminate the system
   *
   * All nodes should call this function before existing. 
   * \param do_barrier whether to do block until every node is finalized, default true.
   */
  void Finalize(const bool do_barrier = true);
  /**
   * \brief add an customer to the system. threadsafe
   */
  void AddCustomer(Customer* customer);
  /**
   * \brief remove a customer by given it's id. threasafe
   */
  void RemoveCustomer(Customer* customer);
  /**
   * \brief get the customer by id, threadsafe
   * \param id the customer id
   * \param timeout timeout in sec
   * \return return nullptr if doesn't exist and timeout
   */
  Customer* GetCustomer(int id, int timeout = 0) const;
  /**
   * \brief get the id of a node (group), threadsafe
   *
   * if it is a  node group, return the list of node ids in this
   * group. otherwise, return {node_id}
   */
  const std::vector<int>& GetNodeIDs(int node_id) const {
    const auto it = node_ids_.find(node_id);
    PANIC_ENFORCE(it != node_ids_.cend(), "node %d doesn't exist", node_id);
    return it->second;
  }
  /**
   * \brief return the key ranges of all server nodes
   */
  const std::vector<Range>& GetServerKeyRanges();
  /**
   * \brief the template of a callback
   */
  using Callback = std::function<void()>;
  /**
   * \brief Register a callback to the system which is called after Finalize()
   *
   * The following codes are equal
   * \code {cpp}
   * RegisterExitCallback(cb);
   * Finalize();
   * \endcode
   *
   * \code {cpp}
   * Finalize();
   * cb();
   * \endcode
   * \param cb the callback function
   */
  void RegisterExitCallback(const Callback& cb) {
    exit_callback_ = cb;
  }
  /**
   * \brief convert from a worker rank into a node id
   * \param rank the worker rank
   */
  static inline int WorkerRankToID(int rank) {
    return rank * 2 + 9;
  }
  /**
   * \brief convert from a server rank into a node id
   * \param rank the server rank
   */
  static inline int ServerRankToID(int rank) {
    return rank * 2 + 8;
  }
  /**
   * \brief convert from a node id into a server or worker rank
   * \param id the node id
   */
  static inline int IDtoRank(int id) {
#ifdef _MSC_VER
#undef max
#endif
    return std::max((id - 8) / 2, 0);
  }
  /** \brief Returns the number of worker nodes */
  int num_workers() const { return num_workers_; }
  /** \brief Returns the number of server nodes */
  int num_servers() const { return num_servers_; }
  /** \brief Returns the rank of this node in its group
   *
   * Each worker will have a unique rank within [0, NumWorkers()). So are
   * servers. This function is available only after \ref Start has been called.
   */
  int my_rank() const { return IDtoRank(van_->my_node().id); }
  /** \brief Returns true if this node is a worker node */
  int is_worker() const { return is_worker_; }
  /** \brief Returns true if this node is a server node. */
  int is_server() const { return is_server_; }
  /** \brief Returns true if this node is a scheduler node. */
  int is_scheduler() const { return is_scheduler_; }
  /** \brief Returns the verbose level. */
  int verbose() const { return verbose_; }
  /** \brief Return whether this node is a recovery node */
  bool is_recovery() const { return van_->my_node().is_recovery; }
  /**
   * \brief barrier
   * \param node_id the barrier group id
   */
  void Barrier(int node_id);
  /**
   * \brief process a control message, called by van
   * \param the received message
   */
  void Manage(const Message& recv);
  /**
   * \brief update the heartbeat record map
   * \param node_id the \ref Node id
   * \param t the last received heartbeat time
   */
  void UpdateHeartbeat(int node_id, time_t t) {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    heartbeats_[node_id] = t;
  }
  /**
   * \brief get node ids that haven't reported heartbeats for over t seconds
   * \param t timeout in sec
   */
  std::vector<int> GetDeadNodes(int t = 60);

 private:
  Postoffice();
  ~Postoffice() { delete van_; }
  Van* van_;
  mutable std::mutex mu_;
  std::unordered_map<int, Customer*> customers_;
  std::unordered_map<int, std::vector<int>> node_ids_;
  std::vector<Range> server_key_ranges_;
  bool is_worker_, is_server_, is_scheduler_;
  int num_servers_, num_workers_;
  bool barrier_done_;
  int verbose_;
  std::mutex barrier_mu_;
  std::condition_variable barrier_cond_;
  std::mutex heartbeat_mu_;
  std::unordered_map<int, time_t> heartbeats_;
  Callback exit_callback_;
  /** \brief Holding a shared_ptr to prevent it from being destructed too early */
  std::shared_ptr<Environment> env_ref_;
  time_t start_time_;
  DISALLOW_COPY_AND_ASSIGN(Postoffice);
};

/** \brief verbose log */
#define MYPS_VLOG(x) LOG_IF(INFO, x <= Postoffice::Get()->verbose())

Postoffice::Postoffice() {
  van_ = Van::Create("zmq");
  env_ref_ = Environment::_GetSharedRef();
  const char* val = NULL;
  val = Environment::Get()->find("DMLC_NUM_WORKER");
  num_workers_ = atoi(val);
  val =  Environment::Get()->find("DMLC_NUM_WORKER");
  num_servers_ = atoi(val);
  val = Environment::Get()->find("DMLC_NUM_WORKER");
  std::string role(val);
  is_worker_ = role == "worker";
  is_server_ = role == "server";
  is_scheduler_ = role == "scheduler";
  verbose_ = GetEnv("PS_VERBOSE", 0);
}

void Postoffice::Start(const char* argv0, const bool do_barrier) {
  // init glog
  if (argv0) {
    //dmlc::InitLogging(argv0);
  }

  // init node info.
  for (int i = 0; i < num_workers_; ++i) {
    int id = WorkerRankToID(i);
    for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
            kWorkerGroup + kScheduler,
            kWorkerGroup + kServerGroup + kScheduler}) {
      node_ids_[g].push_back(id);
    }
  }

  for (int i = 0; i < num_servers_; ++i) {
    int id = ServerRankToID(i);
    for (int g : {id, kServerGroup, kWorkerGroup + kServerGroup,
            kServerGroup + kScheduler,
            kWorkerGroup + kServerGroup + kScheduler}) {
      node_ids_[g].push_back(id);
    }
  }

  for (int g : {kScheduler, kScheduler + kServerGroup + kWorkerGroup,
          kScheduler + kWorkerGroup, kScheduler + kServerGroup}) {
    node_ids_[g].push_back(kScheduler);
  }

  // start van
  van_->Start();

  // record start time
  start_time_ = time(NULL);

  // do a barrier here
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
}

void Postoffice::Finalize(const bool do_barrier) {
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
  van_->Stop();
  if (exit_callback_) exit_callback_();
}


void Postoffice::AddCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  int id = customer->id();
  PANIC_ENFORCE_EQ(customers_.count(id), (size_t)0); //" already exists";
  customers_[id] = customer;
}


void Postoffice::RemoveCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  int id = customer->id();
  customers_.erase(id);
}


Customer* Postoffice::GetCustomer(int id, int timeout) const {
  Customer* obj = nullptr;
  for (int i = 0; i < timeout*1000+1; ++i) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      const auto it = customers_.find(id);
      if (it != customers_.end()) {
        obj = it->second;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return obj;
}

void Postoffice::Barrier(int node_group) {
  if (GetNodeIDs(node_group).size() <= 1) return;
  auto role = van_->my_node().role;
  if (role == Node::SCHEDULER) {
    PANIC_ENFORCE(node_group & kScheduler, "node_group has no kScheduler");
  } else if (role == Node::WORKER) {
    PANIC_ENFORCE(node_group & kWorkerGroup, "node_group has no kWorkerGroup");
  } else if (role == Node::SERVER) {
    PANIC_ENFORCE(node_group & kServerGroup, "node_group has no kServerGroup");
  }

  std::unique_lock<std::mutex> ulk(barrier_mu_);
  barrier_done_ = false;
  Message req;
  req.meta.recver = kScheduler;
  req.meta.request = true;
  req.meta.control.cmd = Control::BARRIER;
  req.meta.control.barrier_group = node_group;
  req.meta.timestamp = van_->GetTimestamp();
  PANIC_ENFORCE_GT(van_->Send(req), 0);

  barrier_cond_.wait(ulk, [this] {
      return barrier_done_;
    });
}

const std::vector<Range>& Postoffice::GetServerKeyRanges() {
  if (server_key_ranges_.empty()) {
    for (int i = 0; i < num_servers_; ++i) {
      server_key_ranges_.push_back(Range(
          kMaxKey / num_servers_ * i,
          kMaxKey / num_servers_ * (i+1)));
    }
  }
  return server_key_ranges_;
}

void Postoffice::Manage(const Message& recv) {
  PANIC_ENFORCE(!recv.meta.control.empty(), "meta.control is empty()");
  const auto& ctrl = recv.meta.control;
  if (ctrl.cmd == Control::BARRIER && !recv.meta.request) {
    barrier_mu_.lock();
    barrier_done_ = true;
    barrier_mu_.unlock();
    barrier_cond_.notify_all();
  }
}

std::vector<int> Postoffice::GetDeadNodes(int t) {
  std::vector<int> dead_nodes;
  if (!van_->IsReady() || t == 0) return dead_nodes;

  time_t curr_time = time(NULL);
  const auto& nodes = is_scheduler_
    ? GetNodeIDs(kWorkerGroup + kServerGroup)
    : GetNodeIDs(kScheduler);
  {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    for (int r : nodes) {
      auto it = heartbeats_.find(r);
      if ((it == heartbeats_.end() || it->second + t < curr_time)
            && start_time_ + t < curr_time) {
        dead_nodes.push_back(r);
      }
    }
  }
  return dead_nodes;
}

}  // namespace mypslite
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PSLITE_POSTOFFICE_H_