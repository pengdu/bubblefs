/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/net_dag_utils.h
// caffe2/caffe2/core/net_dag_utils.cc

#ifndef CAFFE2_CORE_NET_DAG_UTILS_H_
#define CAFFE2_CORE_NET_DAG_UTILS_H_

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <stack>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/caffe2_simple_queue.h"
#include "platform/base_error.h"
#include "platform/types.h"
#include "platform/caffe2_timer.h"
#include "utils/caffe2_blob.h"
#include "utils/caffe2_observer.h"
#include "utils/caffe2_operator.h"
#include "utils/caffe2_net.h"
#include "utils/caffe2_tensor.h"
#include "utils/caffe2_workspace.h"
#include "utils/caffe2_proto_caffe2.h"
#include "utils/caffe2_simple_queue.h"

namespace bubblefs {
namespace mycaffe2 {
namespace dag_utils {

struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  std::vector<int> children_;
  std::vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
  bool is_chain_start_ = false;
};

struct OpGraphNode {
  std::vector<int> children_;
  std::vector<int> parents_;
  int visited_inputs = 0;
  int num_orig_parents;
};

using ExecutionChains = std::unordered_map<int, std::vector<int>>;

ExecutionChains computeChains(std::vector<OperatorNode>& orig_nodes);

ExecutionChains singleChains(std::vector<OperatorNode>& nodes);

std::vector<OperatorNode> prepareOperatorNodes(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws);

std::vector<OpGraphNode> prepareChainGraphNodes(
    const std::vector<dag_utils::OperatorNode>& operator_nodes,
    const std::vector<std::vector<int>>& execution_chains);

namespace {
void prune(int node_idx, std::vector<OpGraphNode>& nodes) {
  // Ancestor table for tracking the visited nodes
  std::vector<bool> ancestors(nodes.size(), false);
  // stack element is pair of <curr_node, previous_node>
  std::stack<std::pair<int, int>> nodes_stack;
  // initialize the prev_node to be -1
  nodes_stack.push(std::make_pair(node_idx, -1));

  while (!nodes_stack.empty()) {
    const auto& node_pair = nodes_stack.top();
    int curr = node_pair.first;
    int prev = node_pair.second;

    // If the node has already been visited, pop curr out of
    // stack and clean up the ancestor table
    PANIC_ENFORCE(curr < ancestors.size(), "Out of bound access");
    if (ancestors[curr]) {
      ancestors[curr] = false;
      nodes_stack.pop();
      continue;
    }

    // Check if this has a parent that can be pruned:
    //  if parent is not the previous node visited and is
    //  an ancestor of the current traversar, it can be
    //  pruned.
    if (prev >= 0) {
      std::vector<int> new_parents;
      for (auto parent : nodes[curr].parents_) {
        if (parent != prev && ancestors[parent]) {
          // We can prune this one
          nodes[parent].children_.erase(
              std::remove(
                  nodes[parent].children_.begin(),
                  nodes[parent].children_.end(),
                  curr),
              nodes[parent].children_.end());
        } else {
          new_parents.push_back(parent);
        }
      }
      nodes[curr].parents_ = new_parents;
    }

    ancestors[curr] = true;

    // Descend -- but only once from each node
    if (nodes[curr].visited_inputs == nodes[curr].num_orig_parents) {
      const auto& children = nodes[curr].children_;
      for (auto child : children) {
        nodes[child].visited_inputs++;
        nodes_stack.push(std::make_pair(child, curr));
      }
    }
  }
}

/**
 * Prune redundant dependencies to improve chaining.
 * TODO: t15868555 This algorithm is fast but can miss dependencies.
 */
std::vector<OpGraphNode> pruneOpNodeGraph(
    const std::vector<OperatorNode>& nodes) {
  Timer t;
  std::vector<OpGraphNode> pruned;

  // Create a separate list of pruned operatornodes used
  // for the chaining computation. Because of the unique_ptr
  // in the OperatorNode, we cannot do a copy but have to
  // copy just the fields we need.
  for (auto& node : nodes) {
    OpGraphNode nd;
    nd.children_ = node.children_;
    nd.parents_ = node.parents_;
    nd.num_orig_parents = nd.parents_.size();
    pruned.push_back(nd);
  }

  for (int i = 0; i < pruned.size(); ++i) {
    if (pruned[i].parents_.size() == 0) {
      prune(i, pruned);
    }
  }

  //LOG(INFO) << "Operator graph pruning prior to chain compute took: " << t.Seconds() << " secs";
  return pruned;
}

void updateOperatorNodes(
    std::vector<OperatorNode>& nodes,
    const ExecutionChains& chains) {
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    if (chains.find(i) != chains.end()) {
      node.is_chain_start_ = true;
    } else {
      node.is_chain_start_ = false;
    }
    node.runtime_parent_count_ = 0;
  }
}
} // namespace

ExecutionChains computeChains(std::vector<OperatorNode>& orig_nodes) {
  const std::vector<OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
  std::vector<int> initial_frontier;
  for (int idx = 0; idx < nodes.size(); ++idx) {
    if (nodes[idx].parents_.size() == 0) {
      initial_frontier.push_back(idx);
    }
  }

  // We need to construct the node_seen_count to know how many inner edges each
  // node has.
  std::unordered_map<int, int> node_seen_count;

  for (int root_index : initial_frontier) {
    const auto& root = nodes[root_index];
    std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
    depth_stack.push(make_pair(root_index, root.children_.begin()));
    node_seen_count[root_index]++;
    PANIC_ENFORCE(
        node_seen_count[root_index] == 1,
        "root node visit count must be == 1 ");

    while (depth_stack.size() > 0) {
      auto cur = depth_stack.top();
      depth_stack.pop();
      if (cur.second != nodes[cur.first].children_.end()) {
        int node_index = *cur.second;
        node_seen_count[node_index]++;
        cur.second++;
        depth_stack.push(cur);
        if (node_seen_count[node_index] == 1) {
          // Visit each child only once.
          depth_stack.push(
              make_pair(node_index, nodes[node_index].children_.begin()));
        }
      }
    }
  }
  // Now, we compute the set of execution chains An execution chain is
  // a linear set of nodes that can be executed on a single stream
  // (e.g. a chain of single input, single output operators)
  ExecutionChains chains;
  std::unordered_set<int> seen_nodes;
  std::vector<int> chain;
  std::pair<int, std::vector<int>::const_iterator> cur;
  std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
  auto check_current_for_chaining = [&]() -> bool {
    return (
        node_seen_count[cur.first] == 1 &&
        (chain.size() == 0 ||
         (
             // A chain of operators is executed without additional
             // synchronization by calling RunAsync sequentially on each
             // operator and passing the same stream id on each call.
             // RunAsync may schedule an async computation on device.
             // In order to be scheduled on the same chain two operators
             // (parent and dependent) need to satisfy one of:
             //  1. Parent op does not have an async part
             //  2. Parent op has async part _and_
             //     both ops are on the same device _and_
             //     dependent op can be executed as an async dependency

             !orig_nodes[chain.back()].operator_->HasAsyncPart() ||
             (IsSameDevice(
                  orig_nodes[cur.first].operator_->device_option(),
                  orig_nodes[chain.back()].operator_->device_option()) &&
              orig_nodes[cur.first].operator_->SupportsAsyncScheduling()))));
  };
  auto commit_chain = [&]() {
    if (chain.size() > 0) {
      PANIC_ENFORCE(
          chains.insert({chain.front(), chain}).second,
          "Chain was already added.");
      //VLOG(2) << "Added chain: " << chain.front() << "with elements";
      for (auto ch : chain) {
        //VLOG(2) << ch << ", ";
      }
      chain.clear();
    }
  };
  auto depth_traverse = [&]() {
    while (cur.second != nodes[cur.first].children_.end() &&
           seen_nodes.find(*cur.second) != seen_nodes.end()) {
      cur.second++;
    }

    if (cur.second != nodes[cur.first].children_.end()) {
      auto next = std::make_pair(*cur.second, nodes[*cur.second].children_.begin());
      depth_stack.push(cur);
      depth_stack.push(next);
    }
  };
  for (int root_index : initial_frontier) {
    depth_stack.push(
        make_pair(root_index, nodes[root_index].children_.begin()));
    while (depth_stack.size() > 0) {
      cur = depth_stack.top();
      depth_stack.pop();
      if (seen_nodes.find(cur.first) == seen_nodes.end()) {
        seen_nodes.insert(cur.first);
        // Has one child, can be candidate for chain or can be added to the
        // previous chain.
        if (nodes[cur.first].children_.size() == 1) {
          if (check_current_for_chaining()) {
            // Add oneself to the current chain.
            //VLOG(1) << "Adding to existing chain" << cur.first;
            chain.push_back(cur.first);
            int index = *nodes[cur.first].children_.begin();
            depth_stack.push(make_pair(index, nodes[index].children_.begin()));
          } else {
            // Can't belong to the previous chain, commit previous chain and
            // start a new one.
            commit_chain();
            chain.push_back(cur.first);
            int index = *nodes[cur.first].children_.begin();
            depth_stack.push(make_pair(index, nodes[index].children_.begin()));
          }
        } else if (
            nodes[cur.first].children_.size() == 0 &&
            check_current_for_chaining()) {
          // Add current node to the current chain and commit.
          chain.push_back(cur.first);
          commit_chain();
        } else {
          // Node has more than one child.
          commit_chain();
          // Add current node as an independent chain since it won't be a part
          // of a bigger chain.
          chain.push_back(cur.first);
          commit_chain();
          depth_traverse();
        }
      } else {
        // This node has been seen before, we will only traverse its children.
        // Commit any pending chains and continue traversing.
        commit_chain();
        depth_traverse();
      }
    } // End while

    // Check if this if is even needed.
    commit_chain();
  }
  PANIC_ENFORCE(
      seen_nodes.size() == nodes.size(),
      "Haven't seen all the nodes, expected number of nodes , but seen only",
      nodes.size());

  updateOperatorNodes(orig_nodes, chains);
  return chains;
}

ExecutionChains singleChains(std::vector<OperatorNode>& nodes) {
  ExecutionChains chains;
  for (auto i = 0; i < nodes.size(); ++i) {
    chains[i] = {i};
  }
  updateOperatorNodes(nodes, chains);
  return chains;
}

std::vector<OperatorNode> prepareOperatorNodes(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  std::vector<OperatorNode> operator_nodes(net_def->op.size());
  std::map<string, int> blob_creator;
  std::map<string, std::set<int>> blob_readers;
  bool net_def_has_device_option = net_def->device_option;
  // Initialize the operators
  for (int idx = 0; idx < net_def->op.size(); ++idx) {
    const OperatorDef& op_def = net_def->op(idx);
    //VLOG(1) << "Creating operator #" << idx << ": " << op_def.name() << ": " << op_def.type();
    if (!op_def.device_option && net_def_has_device_option) {
      OperatorDef temp_def(op_def);
      temp_def.device_option->CopyFrom(net_def->device_option());
      operator_nodes[idx].operator_ = CreateOperator(temp_def, ws, idx);
    } else {
      auto op = CreateOperator(op_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
      operator_nodes[idx].operator_ = std::move(op);
    }
    // Check the inputs, and set up parents if necessary. This addressese the
    // read after write case.
    auto checkInputs =
        [&](const google::protobuf::RepeatedPtrField<std::string>& inputs) {
          for (const string& input : inputs) {
            if (blob_creator.count(input) == 0) {
              //VLOG(1) << "Input " << input << " not produced by this net. "<< "Assuming it is pre-existing.";
            } else {
              int parent = blob_creator[input];
              //VLOG(1) << "op dependency (RaW " << input << "): " << parent << "->" << idx;
              operator_nodes[idx].parents_.push_back(parent);
              operator_nodes[parent].children_.push_back(idx);
            }
            // Add the current idx to the readers of this input.
            blob_readers[input].insert(idx);
          }
        };
    checkInputs(op_def.input());
    checkInputs(op_def.control_input());

    // Check the outputs.
    for (const string& output : op_def.output()) {
      if (blob_creator.count(output) != 0) {
        // This addresses the write after write case - we will assume that all
        // writes are inherently sequential.
        int waw_parent = blob_creator[output];
        //VLOG(1) << "op dependency (WaW " << output << "): " << waw_parent << "->" << idx;
        operator_nodes[idx].parents_.push_back(waw_parent);
        operator_nodes[waw_parent].children_.push_back(idx);
      }
      // This addresses the write after read case - we will assume that writes
      // should only occur after all previous reads are finished.
      for (const int war_parent : blob_readers[output]) {
        //VLOG(1) << "op dependency (WaR " << output << "): " << war_parent << "->" << idx;
        operator_nodes[idx].parents_.push_back(war_parent);
        operator_nodes[war_parent].children_.push_back(idx);
      }
      // Renew the creator of the output name.
      blob_creator[output] = idx;
      // The write would create an implicit barrier that all earlier readers of
      // this output is now parents of the current op, and future writes would
      // not need to depend on these earlier readers. Thus, we can clear up the
      // blob readers.
      blob_readers[output].clear();
    }
  }

  // Now, make sure that the parent list and the children list do not contain
  // duplicated items.
  for (int i = 0; i < operator_nodes.size(); ++i) {
    auto& node = operator_nodes[i];
    // Sort, remove duplicates, and delete self dependency.
    auto& p = node.parents_;
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    p.erase(std::remove(p.begin(), p.end(), i), p.end());
    // Do the same for the children vector.
    auto& c = node.children_;
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    c.erase(std::remove(c.begin(), c.end(), i), c.end());
  }

  return operator_nodes;
}

std::vector<OpGraphNode> prepareChainGraphNodes(
    const std::vector<dag_utils::OperatorNode>& operator_nodes,
    const std::vector<std::vector<int>>& execution_chains) {
  std::unordered_map<int, int> op_to_chain_idx;
  for (int chain_idx = 0; chain_idx < execution_chains.size(); ++chain_idx) {
    const auto& chain_indices = execution_chains[chain_idx];
    for (const auto& chain_op_idx : chain_indices) {
      PANIC_ENFORCE(!op_to_chain_idx.count(chain_op_idx), "");
      op_to_chain_idx[chain_op_idx] = chain_idx;
    }
  }

  std::vector<OpGraphNode> chain_nodes(execution_chains.size());
  for (int op_idx = 0; op_idx < operator_nodes.size(); ++op_idx) {
    PANIC_ENFORCE(op_to_chain_idx.count(op_idx), "");
    auto chain_idx = op_to_chain_idx[op_idx];
    auto& chain = chain_nodes[chain_idx];
    auto& op_node = operator_nodes[op_idx];

    for (const auto& child_idx : op_node.children_) {
      PANIC_ENFORCE(op_to_chain_idx.count(child_idx), "");
      auto child_chain_idx = op_to_chain_idx[child_idx];
      if (child_chain_idx != chain_idx) {
        auto it = std::find(
            chain.children_.begin(), chain.children_.end(), child_chain_idx);
        if (it == chain.children_.end()) {
          chain.children_.push_back(child_chain_idx);
        }
      }
    }

    for (const auto& parent_idx : op_node.parents_) {
      PANIC_ENFORCE(op_to_chain_idx.count(parent_idx),"");
      auto parent_chain_idx = op_to_chain_idx[parent_idx];
      if (parent_chain_idx != chain_idx) {
        auto it = std::find(
            chain.parents_.begin(), chain.parents_.end(), parent_chain_idx);
        if (it == chain.parents_.end()) {
          chain.parents_.push_back(parent_chain_idx);
        }
      }
    }
  }

  return chain_nodes;
}

} // namespace dag_utils
} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_DAG_UTILS_H_