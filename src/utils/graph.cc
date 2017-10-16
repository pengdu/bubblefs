/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/graph/graph.cc

#include "utils/graph.h"
#include <vector>
#include "platform/logging.h"
#include "utils/errors.h"
#include "utils/map_util.h"
#include "utils/strcat.h"

namespace bubblefs {

const int Graph::kControlSlot = -1;

class NodeProperties {
 public:
  //NodeProperties(const NodeDef& node_def,
  //               const DataTypeSlice inputs, const DataTypeSlice outputs)
  //    : node_def(node_def),
  //      input_types(inputs.begin(), inputs.end()),
  //      output_types(outputs.begin(), outputs.end()) {}
  NodeProperties(const NodeDef& node_def)
      : node_def(node_def) { }
        
  NodeDef node_def;
  const DataTypeVector input_types;
  const DataTypeVector output_types;
};

// Node

#define REF_CLASS(key, value) \
  {key, value}, { "Ref" key, value }

const std::unordered_map<string, Node::NodeClass>& Node::kNodeClassTable =
    *new std::unordered_map<string, Node::NodeClass>({
        // Keep in same order as NodeClass values
        REF_CLASS("Switch", NC_SWITCH),
        REF_CLASS("Merge", NC_MERGE),
        REF_CLASS("Enter", NC_ENTER),
        REF_CLASS("Exit", NC_EXIT),
        REF_CLASS("NextIteration", NC_NEXT_ITERATION),
        {"LoopCond", NC_LOOP_COND},
        {"ControlTrigger", NC_CONTROL_TRIGGER},
        {"_Send", NC_SEND},
        {"_HostSend", NC_HOST_SEND},
        {"_Recv", NC_RECV},
        {"_HostRecv", NC_HOST_RECV},
        {"Const", NC_CONSTANT},
        {"HostConst", NC_CONSTANT},
        {"Variable", NC_VARIABLE},
        {"VariableV2", NC_VARIABLE},
        REF_CLASS("Identity", NC_IDENTITY),
        {"GetSessionHandle", NC_GET_SESSION_HANDLE},
        {"GetSessionHandleV2", NC_GET_SESSION_HANDLE},
        {"GetSessionTensor", NC_GET_SESSION_TENSOR},
        {"DeleteSessionTensor", NC_DELETE_SESSION_TENSOR},
        {"Size", NC_METADATA},
        {"Shape", NC_METADATA},
        {"Rank", NC_METADATA},
    });

#undef REF_CLASS

Node::NodeClass Node::GetNodeClassForOp(const string& ts) {
  auto it = kNodeClassTable.find(ts);
  if (it != kNodeClassTable.end()) {
    return it->second;
  } else {
    return NC_OTHER;
  }
}

string Node::DebugString() const {
  string ret = strings::StrCat("{name:'", name(), "' id:", id_);
  if (IsSource()) {
    strings::StrAppend(&ret, " source}");
  } else if (IsSink()) {
    strings::StrAppend(&ret, " sink}");
  } else {
    strings::StrAppend(&ret, " op device:");
    strings::StrAppend(&ret, "{", assigned_device_name(), "}");
    //strings::StrAppend(&ret, " def:{", SummarizeNode(*this), "}}");
  }
  return ret;
}

Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_index_(0) {}

void Node::Initialize(int id, int cost_id,
                      std::shared_ptr<NodeProperties> props) {
  DCHECK_EQ(id_, -1);
  DCHECK(in_edges_.empty());
  DCHECK(out_edges_.empty());
  id_ = id;
  cost_id_ = cost_id;

  props_ = std::move(props);
  // Initialize the class_ based on the type string
  //class_ = GetNodeClassForOp(props_->node_def.op());
}

void Node::Clear() {
  in_edges_.clear();
  out_edges_.clear();
  id_ = -1;
  cost_id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  assigned_device_name_index_ = 0;
}

const string& Node::name() const { return props_->node_def.name; }
const string& Node::type_string() const { return props_->node_def.op; }
const NodeDef& Node::def() const { return props_->node_def; }
//const OpDef& Node::op_def() const { return *props_->op_def; }

int32 Node::num_inputs() const { return props_->input_types.size(); }
DataType Node::input_type(int32 i) const { return props_->input_types[i]; }
const DataTypeVector& Node::input_types() const { return props_->input_types; }

int32 Node::num_outputs() const { return props_->output_types.size(); }
DataType Node::output_type(int32 o) const { return props_->output_types[o]; }
const DataTypeVector& Node::output_types() const {
  return props_->output_types;
}

const string& Node::requested_device() const { return def().device; }

gtl::iterator_range<NeighborIter> Node::out_nodes() const {
  return gtl::make_range(NeighborIter(out_edges_.begin(), false),
                         NeighborIter(out_edges_.end(), false));
}

gtl::iterator_range<NeighborIter> Node::in_nodes() const {
  return gtl::make_range(NeighborIter(in_edges_.begin(), true),
                         NeighborIter(in_edges_.end(), true));
}

void Node::MaybeCopyOnWrite() {
  // NodeProperties may be shared between Nodes. Make a copy if so.
  if (!props_.unique()) {
    props_ = std::make_shared<NodeProperties>(*props_);
  }
}

/*
AttrValue* Node::AddAttrHelper(const string& name) {
  MaybeCopyOnWrite();
  return &((*props_->node_def.mutable_attr())[name]);
}

void Node::ClearAttr(const string& name) {
  MaybeCopyOnWrite();
  (*props_->node_def.mutable_attr()).erase(name);
}
*/

void Node::set_requested_device(const string& device) {
  MaybeCopyOnWrite();
  props_->node_def.set_device(device);
}

Status Node::input_edge(int idx, const Edge** e) const {
  if (idx < 0 || idx >= num_inputs()) {
    return errors::InvalidArgument("Invalid input_edge index: ", idx, ", Node ",
                                   name(), " only has ", num_inputs(),
                                   " inputs.");
  }

  // This does a linear search over the edges.  In the common case,
  // the number of elements is small enough that this search isn't
  // expensive.  Should it become a bottleneck, one can make an
  // optimization where, if the number of edges is small, we use
  // linear iteration, and if the number of edges is large, we perform
  // an indexing step during construction that keeps an array of Edges
  // indexed by pointer.  This would keep the size of each Node small
  // in the common case but make this function faster when the number
  // of edges is large.
  for (const Edge* edge : in_edges()) {
    if (edge->dst_input() == idx) {
      *e = edge;
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find input edge ", idx, " for ", name());
}

// Returns a vector of the non-control input edges to a node, indexed by ID.
Status Node::input_edges(std::vector<const Edge*>* input_edges) const {
  input_edges->clear();
  input_edges->resize(num_inputs(), nullptr);

  for (const Edge* edge : in_edges()) {
    if (edge->IsControlEdge()) continue;
    if (edge->dst_input() < 0 || edge->dst_input() >= num_inputs()) {
      return errors::Internal("Invalid edge input number ", edge->dst_input());
    }
    if ((*input_edges)[edge->dst_input()] != nullptr) {
      return errors::Internal("Duplicate edge input number: ",
                              edge->dst_input());
    }
    (*input_edges)[edge->dst_input()] = edge;
  }

  for (int i = 0; i < num_inputs(); ++i) {
    if ((*input_edges)[i] == nullptr) {
      return errors::InvalidArgument("Missing edge input number: ", i);
    }
  }
  return Status::OK();
}

Status Node::input_node(int idx, Node** n) const {
  const Edge* e;
  TF_RETURN_IF_ERROR(input_edge(idx, &e));
  if (e == nullptr) {
    *n = nullptr;
  } else {
    *n = e->src();
  }
  return Status::OK();
}

Status Node::input_node(int idx, const Node** const_n) const {
  Node* n;
  TF_RETURN_IF_ERROR(input_node(idx, &n));
  *const_n = n;
  return Status::OK();
}


// Graph

Graph::Graph() { }

Graph::~Graph() {
  // Manually call the destructors for all the Nodes we constructed using
  // placement new.
  for (Node* node : nodes_) {
    if (node != nullptr) {
      node->~Node();
    }
  }
  for (Node* node : free_nodes_) {
    node->~Node();
  }
  // Edges have no destructor, and we arena-allocated them, so no need to
  // destroy them.
}

Node* Graph::AddNode(const NodeDef& node_def, Status* status) {
  //const OpDef* op_def;
  //status->Update(ops_.LookUpOpDef(node_def.op(), &op_def));
  //if (!status->ok()) return nullptr;

  //DataTypeVector inputs;
  //DataTypeVector outputs;
  //status->Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  //if (!status->ok()) {
  //  *status = AttachDef(*status, node_def);
  //  return nullptr;
  //}

  Node* node = AllocateNode(
      std::make_shared<NodeProperties>(node_def),
      nullptr);
  return node;
}

Node* Graph::CopyNode(Node* node) {
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());
  Node* copy = AllocateNode(node->props_, node);
  copy->set_assigned_device_name(node->assigned_device_name());

  // Since the OpDef of a function may be owned by the Graph that owns 'node',
  // relookup the OpDef in the target graph. If it differs, then clone the
  // node properties with the updated OpDef.
  //const OpDef* op_def;
  //TF_CHECK_OK(ops_.LookUpOpDef(node->type_string(), &op_def));
  //if (op_def != node->props_->op_def) {
  //  copy->MaybeCopyOnWrite();
  //  copy->props_->op_def = op_def;
  //}

  return copy;
}

void Graph::RemoveNode(Node* node) {
  DCHECK(IsValidNode(node)) << node->DebugString();
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());

  // Remove any edges involving this node.
  while (!node->in_edges_.empty()) {
    RemoveEdge(*node->in_edges_.begin());
  }
  while (!node->out_edges_.empty()) {
    RemoveEdge(*node->out_edges_.begin());
  }
  ReleaseNode(node);
}

const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y) {
  DCHECK(IsValidNode(source)) << source->DebugString();
  DCHECK(IsValidNode(dest)) << dest->DebugString();

  // source/sink must only be linked via control slots, and
  // control slots must only be linked to control slots.
  if (source == source_node() || dest == sink_node() || x == kControlSlot ||
      y == kControlSlot) {
    DCHECK_EQ(x, kControlSlot) << source->DebugString();
    DCHECK_EQ(y, kControlSlot) << dest->DebugString();
  }

  Edge* e = nullptr;
  if (free_edges_.empty()) {
    e = new (arena_.Alloc(sizeof(Edge))) Edge;  // placement new
  } else {
    e = free_edges_.back();
    free_edges_.pop_back();
  }
  e->id_ = edges_.size();
  e->src_ = source;
  e->dst_ = dest;
  e->src_output_ = x;
  e->dst_input_ = y;
  CHECK(source->out_edges_.insert(e).second);
  CHECK(dest->in_edges_.insert(e).second);
  edges_.push_back(e);
  ++num_edges_;
  return e;
}

void Graph::RemoveEdge(const Edge* e) {
  DCHECK(IsValidNode(e->src_)) << e->src_->DebugString();
  DCHECK(IsValidNode(e->dst_)) << e->dst_->DebugString();
  CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
  CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
  CHECK_EQ(e, edges_[e->id_]);
  CHECK_GT(num_edges_, 0);

  edges_[e->id_] = nullptr;

  Edge* del = const_cast<Edge*>(e);
  del->src_ = nullptr;
  del->dst_ = nullptr;
  del->id_ = -1;
  del->src_output_ = kControlSlot - 1;
  del->dst_input_ = kControlSlot - 1;
  free_edges_.push_back(del);
  --num_edges_;
}

string Graph::NewName(StringPiece prefix) {
  return strings::StrCat(prefix, "/_", name_counter_++);
}

bool Graph::IsValidNode(Node* node) const {
  if (node == nullptr) return false;
  const int id = node->id();
  if (id < 0 || static_cast<size_t>(id) >= nodes_.size()) return false;
  return nodes_[id] == node;
}

Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props,
                          const Node* cost_node) {
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new (arena_.Alloc(sizeof(Node))) Node;  // placement new
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  node->graph_ = this;
  const int id = nodes_.size();
  int cost_id = cost_node ? cost_node->cost_id() : id;
  node->Initialize(id, cost_id, std::move(props));
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}

void Graph::ReleaseNode(Node* node) {
  DCHECK(IsValidNode(node)) << node->DebugString();
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

// Ensures that 'device_name' is present in the device name table, and returns
// the index of that device name. The index is stable, and can be used in
// calls to Node::set_assigned_device_name_index().
int Graph::InternDeviceName(const string& device_name) {
  // Special case, very common.  Also, this allows us to use a single map
  // lookup below, instead of two.  The 'if (index_cell > 0)' test below
  // relies on this check.
  if (device_name.empty()) {
    return 0;
  }

  int& index_cell = device_names_map_[device_name];
  if (index_cell > 0) {
    return index_cell;
  }

  const int index = device_names_map_.size();
  index_cell = index;
  device_names_.push_back(device_name);
  return index;
}

}  // namespace bubblefs