
// caffe/include/caffe/net.hpp

#ifndef BUBBLEFS_UTILS_CAFFE_NET_H_
#define BUBBLEFS_UTILS_CAFFE_NET_H_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/caffe_blob.h"
#include "utils/caffe_layer.h"
#include "utils/caffe_proto_caffe.h"

namespace bubblefs {
namespace mycaffe {

struct Phase;
struct NetParameter;
struct NetState;
struct NetStateRule;
  
/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const std::vector<string>* stages = NULL);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *
   */
  const std::vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const std::vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const std::vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const std::vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const std::vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const std::vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const std::vector<std::vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const std::vector<std::vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const std::vector<int> & top_ids(int i) const {
    //CHECK_GE(i, 0) << "Invalid layer id";
    //CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const std::vector<int> & bottom_ids(int i) const {
    //CHECK_GE(i, 0) << "Invalid layer id";
    //CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const std::vector<std::vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const std::vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const std::vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const std::vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const std::vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const std::vector<float>& params_lr() const { return params_lr_; }
  inline const std::vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const std::vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const std::vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const std::map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const std::vector<int>& param_owners() const { return param_owners_; }
  inline const std::vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const std::vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const std::vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const std::vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const std::vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const std::vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const std::vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const std::vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const std::vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 std::map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   std::map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  std::vector<shared_ptr<Layer<Dtype> > > layers_;
  std::vector<string> layer_names_;
  std::map<string, int> layer_names_index_;
  std::vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  std::vector<shared_ptr<Blob<Dtype> > > blobs_;
  std::vector<string> blob_names_;
  std::map<string, int> blob_names_index_;
  std::vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  std::vector<vector<Blob<Dtype>*> > bottom_vecs_;
  std::vector<vector<int> > bottom_id_vecs_;
  std::vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  std::vector<vector<Blob<Dtype>*> > top_vecs_;
  std::vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  std::vector<Dtype> blob_loss_weights_;
  std::vector<vector<int> > param_id_vecs_;
  std::vector<int> param_owners_;
  std::vector<string> param_display_names_;
  std::vector<pair<int, int> > param_layer_indices_;
  std::map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  std::vector<int> net_input_blob_indices_;
  std::vector<int> net_output_blob_indices_;
  std::vector<Blob<Dtype>*> net_input_blobs_;
  std::vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  std::vector<shared_ptr<Blob<Dtype> > > params_;
  std::vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  std::vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  std::vector<float> params_lr_;
  std::vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  std::vector<float> params_weight_decay_;
  std::vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  std::vector<Callback*> before_forward_;
  std::vector<Callback*> after_forward_;
  std::vector<Callback*> before_backward_;
  std::vector<Callback*> after_backward_;

  DISALLOW_COPY_AND_ASSIGN(Net);
};

}  // namespace mycaffe
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE_NET_H_