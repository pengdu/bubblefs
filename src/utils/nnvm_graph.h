/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nnvm as well as basic data structure.
 */

// nnvm/include/nnvm/graph.h
// nnvm/src/core/graph.cc

#ifndef BUBBLEFS_UTILS_NNVM_GRAPH_H_
#define BUBBLEFS_UTILS_NNVM_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "utils/nnvm_base.h"
#include "utils/nnvm_node.h"
#include "utils/nnvm_symbolic.h"

namespace bubblefs {
namespace mynnvm {

class IndexedGraph;

/*!
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph {
 public:
  /*! \brief outputs of the computation graph. */
  std::vector<NodeEntry> outputs;
  /*!
   * \brief attributes of a graph
   *  Note that attribute is shared pointer and can be shared across graphs.
   *
   *  It is highly recommended to keep each attribute immutable.
   *  It is also safe to implement an copy-on-write semnatics.
   *
   *  Copy when shared_ptr.unique is not true, while reuse original space
   *  when shared_ptr.unique is true.
   */
  std::unordered_map<std::string, std::shared_ptr<any> > attrs;
  /*!
   * \brief Get the immutable attribute from attrs.
   * \param attr_name the name of the attribute
   * \return the reference to corresponding attribute
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline const T& GetAttr(const std::string& attr_name) const;
  /*!
   * \brief Get a move copy of the attribute, implement copy on write semantics.
   *  The content is moved if the reference counter of shared_ptr is 1.
   *  The attribute is erased from attrs after the call.
   *
   * \param attr_name the name of the attribute
   * \return a new copy of the corresponding attribute.
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline T MoveCopyAttr(const std::string& attr_name);
  /*!
   * \brief get a indexed graph of current graph, if not exist, create it on demand
   * \return The indexed graph.
   * \sa IndexedGraph
   */
  const IndexedGraph& indexed_graph() const;

 private:
  // internal structure of indexed graph
  mutable std::shared_ptr<const IndexedGraph> indexed_graph_;
};

/*!
 * \brief Auxiliary data structure to index a graph.
 *  It maps Nodes in the graph to consecutive integers node_id.
 *  It also maps IndexedGraph::NodeEntry to consecutive integer entry_id.
 *  This allows storing properties of Node and NodeEntry into
 *  compact vector and quickly access them without resorting to hashmap.
 *
 *  The node_id and entry_rptr are the same as the JSON graph produced by SaveJSON Pass.
 */
class IndexedGraph {
 public:
  /*! \brief represents a data in the graph */
  struct NodeEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t node_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief version of the node */
    uint32_t version;
  };
  /*! \brief Node data structure in IndexedGraph */
  struct Node {
    /*! \brief pointer to the source node */
    const mynnvm::Node* source;
    /*! \brief inputs to the node */
    array_view<NodeEntry> inputs;
    /*! \brief control flow dependencies to the node */
    array_view<uint32_t> control_deps;
    /*! \brief weak reference to node */
    std::weak_ptr<Node> weak_ref;
  };
  /*! \return number of nodes in the graph */
  inline size_t num_nodes() const {
    return nodes_.size();
  }
  /*! \return total number of NodeEntry in the graph */
  inline size_t num_node_entries() const {
    return entry_rptr_.back();
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param node_id The node index
   * \param index the output index
   * \return the unique index.
   */
  inline uint32_t entry_id(uint32_t node_id, uint32_t index) const {
    return entry_rptr_[node_id] + index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const NodeEntry& e) const {
    return entry_rptr_[e.node_id] + e.index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given NodeEntry.
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const mynnvm::NodeEntry& e) const {
    return entry_rptr_[node_id(e.node.get())] + e.index;
  }
  
  /*!
   * \brief Get the corresponding node id for a given Node in the IndexedGraph.
   * \param node The Node to query for index.
   * \return the node index.
   */
  inline uint32_t node_id(const mynnvm::Node* node) const {
    return node2index_.at(node);
  }
  /*!
   * \brief Get the corresponding Node structure for a given node_id.
   * \param node_id The node id
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](uint32_t node_id) const {
    return nodes_[node_id];
  }
  /*!
   * \brief Get the corresponding Node structure
   * \param node The pointer to the Node structure
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](const mynnvm::Node* node) const {
    return nodes_[node_id(node)];
  }
  /*! \return list of argument nodes */
  inline const std::vector<uint32_t>& input_nodes() const {
    return input_nodes_;
  }
  /*! \return list of mutable nodes */
  inline const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return mutable_input_nodes_;
  }
  /*! \return list of output entries */
  inline const std::vector<NodeEntry>& outputs() const {
    return outputs_;
  }

  /*! \return whether a node is existed in the indexed graph */
  inline bool exist(const mynnvm::Node* node) const {
    return node2index_.count(node);
  }

  // disalllow copy assign
  IndexedGraph(const IndexedGraph&) = delete;

 private:
  friend class Graph;
  /*!
   * \brief Constructor an IndexedGraph from normal Graph
   * \param other The source graph.
   */
  explicit IndexedGraph(const Graph& other);
  // Node pointers in CSR structure.
  std::vector<Node> nodes_;
  // Index to all input nodes.
  std::vector<uint32_t> input_nodes_;
  // Index to all mutable input nodes.
  std::unordered_set<uint32_t> mutable_input_nodes_;
  // space to store the outputs entries
  std::vector<NodeEntry> outputs_;
  // mapping from node to index.
  std::unordered_map<const mynnvm::Node*, uint32_t> node2index_;
  // CSR pointer of node entries
  std::vector<size_t> entry_rptr_;
  // space to store input entries of each
  std::vector<NodeEntry> input_entries_;
  // control flow dependencies
  std::vector<uint32_t> control_deps_;
};

/*!
 * \brief perform a Post Order DFS visit to each node in the graph.
 *  This order is deterministic and is also topoligical sorted.
 * \param heads The heads in the graph.
 * \param fvisit a function of type std::function<void(const std::shared_ptr<Node>&)>
 * \tparam FVisit The function type to perform the visit.
 */
template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads, FVisit fvisit);

// inline function implementations
template<typename T>
inline const T& Graph::GetAttr(const std::string& attr_name) const {
  auto it = attrs.find(attr_name);
  PANIC_ENFORCE(it != attrs.end(), "Cannot find attribute "); // "Cannot find attribute " << attr_name << " in the graph";
  return mynnvm::get<T>(*it->second);
}

template<typename T>
inline T Graph::MoveCopyAttr(const std::string& attr_name) {
  auto it = attrs.find(attr_name);
  PANIC_ENFORCE(it != attrs.end(), "Cannot find attribute "); // "Cannot find attribute " << attr_name << " in the graph";
  std::shared_ptr<any> sptr = it->second;
  attrs.erase(it);
  if (sptr.unique()) {
    return std::move(mynnvm::get<T>(*sptr));
  } else {
    return mynnvm::get<T>(*sptr);
  }
}

template <typename GNode, typename HashType,
           typename FVisit, typename HashFunc,
          typename InDegree, typename GetInput>
void PostOrderDFSVisit(const std::vector<GNode>& heads,
                       FVisit fvisit,
                       HashFunc hash,
                       InDegree indegree,
                       GetInput getinput) {
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_set<HashType> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(head_hash);
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        fvisit(back.first);
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (visited.count(input_hash) == 0) {
          stack.push_back(std::make_pair(input, 0));
          visited.insert(input_hash);
        }
      }
    }
  }
}

template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads,
                     FVisit fvisit) {
  typedef const NodePtr* GNode;
  std::vector<GNode> head_nodes(heads.size());
  std::transform(heads.begin(), heads.end(), head_nodes.begin(),
                 [](const NodeEntry& e)->GNode {
                   return &e.node;
                 });
  PostOrderDFSVisit<GNode, Node*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->Node* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t {  // InDegree
        if (!(*n)) return 0;
        return (*n)->inputs.size() + (*n)->control_deps.size();
      },
      [](GNode n, uint32_t index)->GNode {  // GetInput
        if (index < (*n)->inputs.size()) {
          return &(*n)->inputs.at(index).node;
        } else {
          return &(*n)->control_deps.at(index - (*n)->inputs.size());
        }
      });
}

const IndexedGraph& Graph::indexed_graph() const {
  if (indexed_graph_ == nullptr) {
    indexed_graph_.reset(new IndexedGraph(*this));
  }
  return *indexed_graph_;
}

// implement constructor from graph
/*
IndexedGraph::IndexedGraph(const Graph &g) {
  entry_rptr_.push_back(0);
  std::vector<size_t> inputs_rptr{0}, control_rptr{0};

  DFSVisit(g.outputs, [this, &inputs_rptr, &control_rptr]
             (const NodePtr& n) {
      PANIC_ENFORCE_LT(nodes_.size(), std::numeric_limits<uint32_t>::max());
      uint32_t nid = static_cast<uint32_t>(nodes_.size());
      // nodes_
      IndexedGraph::Node new_node;
      new_node.source = n.get();
      new_node.weak_ref = n;
      nodes_.emplace_back(std::move(new_node));
      // arg_nodes_
      if (n->is_variable()) {
        input_nodes_.push_back(nid);
      }
      // node2index_
      node2index_[n.get()] = nid;
      // entry rptr
      entry_rptr_.push_back(entry_rptr_.back() + n->num_outputs());
      // input entries
      for (const auto& e : n->inputs) {
        auto it = node2index_.find(e.node.get());
        PANIC_ENFORCE(it != node2index_.end() && it->first == e.node.get(), "");
        input_entries_.emplace_back(NodeEntry{it->second, e.index, e.version});
      }
      inputs_rptr.push_back(input_entries_.size());
      // control deps
      for (const auto& nptr : n->control_deps) {
        auto it = node2index_.find(nptr.get());
        PANIC_ENFORCE(it != node2index_.end() && it->first == nptr.get(), "");
        control_deps_.push_back(it->second);
      }
      control_rptr.push_back(control_deps_.size());
  });

  for (const auto& e : g.outputs) {
    outputs_.emplace_back(NodeEntry{
        node2index_.at(e.node.get()), e.index, e.version});
  }

  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  // setup array view
  // input_entries_ and control_rptr must not change after this step.
  const NodeEntry* iptr = mydmlc::BeginPtr(input_entries_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].inputs = array_view<NodeEntry>(
        iptr + inputs_rptr[nid], iptr + inputs_rptr[nid + 1]);
    if (nodes_[nid].source->op() != nullptr &&
        fmutate_inputs.count(nodes_[nid].source->op())) {
      for (uint32_t i : fmutate_inputs[nodes_[nid].source->op()](nodes_[nid].source->attrs)) {
        mutable_input_nodes_.insert(nodes_[nid].inputs[i].node_id);
      }
    }
  }
  const uint32_t* cptr = mydmlc::BeginPtr(control_deps_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].control_deps = array_view<uint32_t>(
        cptr + control_rptr[nid], cptr + control_rptr[nid + 1]);
  }
}
*/

}  // namespace mynnvm
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_NNVM_GRAPH_H_