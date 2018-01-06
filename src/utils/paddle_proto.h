/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

//syntax = "proto2";

package paddle;

// Paddle/proto/DataConfig.proto

message FileGroupConf {
  optional uint32 queue_capacity = 1 [ default = 1 ];
  // how many files to load for a load file thread
  optional int32 load_file_count = 2 [ default = 1 ];
  // how many threads to load files
  // Setting to be 5~10 is appropriate when loading files by hadoop vfs
  optional int32 load_thread_num = 3 [ default = 1 ];
};

message DataConfig {

  required string type = 1;

  // name of a text file which contains a list of file names at each line
  optional string files = 3;

  optional int32 feat_dim = 4;         // feature dimension of one frame
  repeated int32 slot_dims = 5;        // feature slot dims
  optional int32 context_len = 6;      // max neibour frame numbers
  optional uint64 buffer_capacity = 7; // the number of samples

  // part of data used in training
  // if not -1, part of train data is used in training
  optional int64 train_sample_num = 8 [ default = -1 ];

  // The number of documents processed once
  optional int32 file_load_num = 9 [ default = -1 ];
  optional bool async_load_data = 12 [ default = false ];
  /// Note the field number 10, 11 and 13 have been deprecated.
  optional bool for_test = 14
      [ default = false ]; // whether this data is for test
  optional FileGroupConf file_group_conf = 15;
  repeated int32 float_slot_dims = 16;

  /// Note the field number 17, 18 and 19 have been deprecated.

  // a list of values which will be used to create additional one dimensional
  // float
  // values slots. These one dimensional slots can be used as the weight input
  // for cost layers.
  // Currently this is only supported by ProtoDataProvider.
  repeated double constant_slots = 20;

  // for PyDataProvider.
  // Specify the load data script module name, object name and user args
  optional string load_data_module = 21;
  optional string load_data_object = 22;
  optional string load_data_args = 23;

  // for MultiDataProvider
  repeated DataConfig sub_data_configs = 24; // sub dataproviders
                                             /*
                                              * the ratio of each sub dataproviders:
                                              * e.g. sub dataprovider A's ratio is 1, B's ratio is 9, batch_size is 100,
                                              * then each mini-batch is combined by 10 instance from A and 90 instances
                                              * from B.
                                              */
  optional int32 data_ratio = 25;
  /*
   * if one of the sub dataproviders is running out of data, then
   * (1) it is "main data", then finish current pass.
   * (2) it is not "main data", then reset it, and try getNextBatch again.
   */
  optional bool is_main_data = 26 [ default = true ];

  // the usage ratio of instances. Setting to 1.0 means the use of all
  // instances.
  optional double usage_ratio = 27 [ default = 1.0 ];
};

// Paddle/proto/DataFormat.proto

/*
 If values is not empty and ids is empty, this is a dense vector.
 If values is not empty and ids is not empty, this is a sparse vector. The
 position of each value
 is specified by ids.
 If values is empty and ids is not empty, this is a sparse vector whose non-zero
 values are 1.
 The position of each 1 is specified by ids.
*/
message VectorSlot {
  repeated float values = 1 [ packed = true ];
  repeated uint32 ids = 2 [ packed = true ];
  /* For multidimensional data, for example "image width height depth" */
  repeated uint32 dims = 3 [ packed = true ];
  repeated string strs = 4;
};

/*
 SubseqSlot use to record whether VectorSlot or any other slot in future has
 subseq.
 If not all VectorSlot have subseq, we only store the one who has subseq, and
 use *slot_id* to record it.
 One vector_slots has one sequence, and it may have N subseq, thus the number of
 *lens* will be N too.
*/
message SubseqSlot {
  required uint32 slot_id = 1; // the id of slot who has subseq
  repeated uint32 lens = 2;    // lengths of sub-sequence in the slot
};

message SlotDef {
  enum SlotType {
    VECTOR_DENSE = 0;
    VECTOR_SPARSE_NON_VALUE = 1;
    VECTOR_SPARSE_VALUE = 2;
    INDEX = 3; // This can be used as label, or word id, etc.
    VAR_MDIM_DENSE = 4;
    VAR_MDIM_INDEX = 5;
    STRING = 6;
  }
  required SlotType type = 1;
  required uint32 dim =
      2; // For INDEX slots, this means the maximal index plus 1.
};

message DataHeader {
  // INDEX slot should be always after VECTOR slots.
  repeated SlotDef slot_defs = 1;
};

message DataSample {
  optional bool is_beginning = 1
      [ default = true ]; // is the beginning of a sequence
  repeated VectorSlot vector_slots = 2;
  repeated uint32 id_slots = 3 [ packed = true ];
  /* use ids of VectorSlot */
  repeated VectorSlot var_id_slots = 4;
  repeated SubseqSlot subseq_slots = 5;
};

// Paddle/proto/ModelConfig.proto

/**
 * Various structs for the configuration of a neural network
 */

message ExternalConfig {
  repeated string layer_names = 1;
  repeated string input_layer_names = 2;
  repeated string output_layer_names = 3;
}

message ActivationConfig {
  // identity: f(x) = x
  // sigmoid: f(x) = 1 / (1 + exp(-x))
  // logistic: f(x) = (1 - exp(-x)) / (1+ exp(-x))
  // softmax: y_i = f(x_i) = exp(x_i) / (\sum_i exp(x_i))
  // relu: y = max(0, x)
  required string type = 1;
};

message ConvConfig {
  // filter_size = 5, says that this layer will use
  // filters of size 5x5 pixels.
  required uint32 filter_size = 1;

  // The image data dimensionality.
  // This value must be either 1, 2, 3, or a multiple of 4.
  required uint32 channels = 2;

  // stride = 1, indicates that the distance between
  // successive filter applications should be 1 pixel.
  required uint32 stride = 3;

  // padding = 4, instructs the net to implicitly
  // pad the images with a 4-pixel border of zeros.
  required uint32 padding = 4;

  // If groups = 4 together with the filters = 32 parameter,
  // they state that this convolutional layer is to have 4
  // groups of 32 filters. Each filter will connect to 8
  // input channels.
  required uint32 groups = 5;
  required uint32 filter_channels = 6;

  // The size of output feature map.
  required uint32 output_x = 7;

  // The size of input feature map.
  required uint32 img_size = 8;

  // caffe mode for output size coherence
  required bool caffe_mode = 9 [ default = true ];

  // if filter_size_y is set , this convolutional layer will use
  // filters of size filter_size * filter_size_y pixels.
  // if filter_size_y is not set, this convolutional layer will use
  // filters of size filter_size * filter_size
  required uint32 filter_size_y = 10;
  required uint32 padding_y = 11;
  required uint32 stride_y = 12;

  // if not set, use output_x
  optional uint32 output_y = 13;

  // if not set, use img_size
  optional uint32 img_size_y = 14;

  optional uint32 dilation = 15 [ default = 1 ];
  optional uint32 dilation_y = 16 [ default = 1 ];

  optional uint32 filter_size_z = 17 [ default = 1 ];
  optional uint32 padding_z = 18 [ default = 1 ];
  optional uint32 stride_z = 19 [ default = 1 ];
  optional uint32 output_z = 20 [ default = 1 ];
  optional uint32 img_size_z = 21 [ default = 1 ];
}

message PoolConfig {
  // max or avg pooling
  required string pool_type = 1;
  required uint32 channels = 2;

  // Defines the size of the pooling region in
  // the x (equivalently, y) dimension.
  required uint32 size_x = 3;

  // Tell the net where in the input image to start the pooling.
  // start is deprecated now.
  optional uint32 start = 4;

  // Defines the stride size between successive pooling squares.
  required uint32 stride = 5 [ default = 1 ];

  // The size of output feature map.
  required uint32 output_x = 6;

  // The size of input feature map.
  required uint32 img_size = 7;

  // padding = 4, instructs the net to implicitly
  // pad the images with a 4-pixel border of zeros.
  optional uint32 padding = 8 [ default = 0 ];

  // if not set, use size_x
  optional uint32 size_y = 9;

  // if not set, use stride
  optional uint32 stride_y = 10;

  // if not set, use output_x
  optional uint32 output_y = 11;

  // if not set, use img_size
  optional uint32 img_size_y = 12;

  // if not set, use padding
  optional uint32 padding_y = 13;

  optional uint32 size_z = 14 [ default = 1 ];
  optional uint32 stride_z = 15 [ default = 1 ];
  optional uint32 output_z = 16 [ default = 1 ];
  optional uint32 img_size_z = 17 [ default = 1 ];
  optional uint32 padding_z = 18 [ default = 1 ];

  optional bool exclude_mode = 19;
}

message SppConfig {
  required ImageConfig image_conf = 1;
  required string pool_type = 2;
  required uint32 pyramid_height = 3;
}

message NormConfig {
  // rnorm or cmrnorm
  required string norm_type = 1;
  required uint32 channels = 2;

  // rnorm: this defines the size of the local regions
  // used for response normalization.
  // cmrnorm: The size parameter indicates how many
  // nearby maps to use for normalization.
  required uint32 size = 3;

  // the parameters for normalization
  // u = u / (1+scale*sum(u^2 in window))^pow
  required double scale = 4;
  required double pow = 5;

  // The size of output feature map.
  required uint32 output_x = 6;

  // The size of input feature map.
  required uint32 img_size = 7;

  // normalize with fixed window or sliding window
  // u = u / (1+scale*sum(u^2 in window))^pow
  // fixed window: shared a fixed window for each value
  // sliding window: have a different window for each value
  optional bool blocked = 8;

  // if not set, use output_x
  optional uint32 output_y = 9;

  // if not set, use img_size
  optional uint32 img_size_y = 10;
}

message BlockExpandConfig {
  required uint32 channels = 1;

  required uint32 stride_x = 2;
  required uint32 stride_y = 3;

  required uint32 padding_x = 4;
  required uint32 padding_y = 5;

  required uint32 block_x = 6;
  required uint32 block_y = 7;

  // The size of output feature map.
  required uint32 output_x = 8;
  required uint32 output_y = 9;

  // The size of input feature map.
  required uint32 img_size_x = 10;
  required uint32 img_size_y = 11;
}

message MaxOutConfig {
  required ImageConfig image_conf = 1;
  required uint32 groups = 2;
}

message RowConvConfig { required uint32 context_length = 1; }

message SliceConfig {
  required uint32 start = 1;
  required uint32 end = 2;
}

message ProjectionConfig {
  required string type = 1;
  required string name = 2;
  required uint64 input_size = 3;
  required uint64 output_size = 4;

  // For ShiftProjection
  optional int32 context_start = 5;
  optional int32 context_length = 6;
  optional bool trainable_padding = 7 [ default = false ];

  // For convolution
  optional ConvConfig conv_conf = 8;
  optional int32 num_filters = 9;

  // For IdentityOffsetProjection
  optional uint64 offset = 11 [ default = 0 ];

  // For pool
  optional PoolConfig pool_conf = 12;

  // For slice
  // Each slice output is the input[start, end)
  repeated SliceConfig slices = 13;
}

message OperatorConfig {
  required string type = 1;
  repeated int32 input_indices = 2;
  repeated uint64 input_sizes = 3;
  required uint64 output_size = 4;

  // For DotMulOperator
  optional double dotmul_scale = 5 [ default = 1.0 ];

  // For ConvOperator
  optional ConvConfig conv_conf = 6;
  optional int32 num_filters = 7;
}

message BilinearInterpConfig {
  // The size of input feature map.
  required ImageConfig image_conf = 1;
  // The size of output feature map.
  required uint32 out_size_x = 2;
  required uint32 out_size_y = 3;
}

message ImageConfig {
  // The image data dimensionality.
  // This value must be either 1, 2, 3, or a multiple of 4.
  required uint32 channels = 2;

  // The size of input feature map.
  required uint32 img_size = 8;
  optional uint32 img_size_y = 9;
  optional uint32 img_size_z = 10 [ default = 1 ];
}

message PriorBoxConfig {
  repeated uint32 min_size = 1;
  repeated uint32 max_size = 2;
  repeated float aspect_ratio = 3;
  repeated float variance = 4;
}

message PadConfig {
  required ImageConfig image_conf = 1;
  repeated uint32 pad_c = 2;
  repeated uint32 pad_h = 3;
  repeated uint32 pad_w = 4;
}

message ReshapeConfig {
  repeated uint32 height_axis = 1;
  repeated uint32 width_axis = 2;
}

message MultiBoxLossConfig {
  required uint32 num_classes = 1;
  required float overlap_threshold = 2;
  required float neg_pos_ratio = 3;
  required float neg_overlap = 4;
  required uint32 background_id = 5;
  required uint32 input_num = 6;
  optional uint32 height = 7 [ default = 1 ];
  optional uint32 width = 8 [ default = 1 ];
}

message DetectionOutputConfig {
  required uint32 num_classes = 1;
  required float nms_threshold = 2;
  required uint32 nms_top_k = 3;
  required uint32 background_id = 4;
  required uint32 input_num = 5;
  required uint32 keep_top_k = 6;
  required float confidence_threshold = 7;
  optional uint32 height = 8 [ default = 1 ];
  optional uint32 width = 9 [ default = 1 ];
}

message ClipConfig {
  required double min = 1;
  required double max = 2;
}

message ROIPoolConfig {
  required uint32 pooled_width = 1;
  required uint32 pooled_height = 2;
  required float spatial_scale = 3;
  optional uint32 height = 4 [ default = 1 ];
  optional uint32 width = 5 [ default = 1 ];
}

message ScaleSubRegionConfig {
  required ImageConfig image_conf = 1;
  required float value = 2;
}

message LayerInputConfig {
  required string input_layer_name = 1;
  optional string input_parameter_name = 2;
  optional ConvConfig conv_conf = 3;
  optional PoolConfig pool_conf = 4;
  optional NormConfig norm_conf = 5;
  optional ProjectionConfig proj_conf = 6;
  optional BlockExpandConfig block_expand_conf = 7;
  optional ImageConfig image_conf = 8;
  // If the input layer has multi-output.
  // Set the argument name.
  optional string input_layer_argument = 9;
  optional BilinearInterpConfig bilinear_interp_conf = 10;
  optional MaxOutConfig maxout_conf = 11;
  optional SppConfig spp_conf = 12;
  optional PriorBoxConfig priorbox_conf = 13;
  optional PadConfig pad_conf = 14;
  optional RowConvConfig row_conv_conf = 15;
  optional MultiBoxLossConfig multibox_loss_conf = 16;
  optional DetectionOutputConfig detection_output_conf = 17;
  optional ClipConfig clip_conf = 18;
  optional ScaleSubRegionConfig scale_sub_region_conf = 19;
  optional ROIPoolConfig roi_pool_conf = 20;
}

message LayerConfig {
  required string name = 1;
  required string type = 2;
  optional uint64 size = 3;
  // optional ActivationConfig activation = 4;
  optional string active_type = 4;
  repeated LayerInputConfig inputs = 5;
  optional string bias_parameter_name = 6;

  // This number must be a multiple of 16.
  optional uint32 num_filters = 7;

  // indicates that the biases of every filter in this layer
  // should be shared amongst all applications of that filter
  // (which is how convnets are usually trained). Setting this to
  // false will untie the biases, yielding a separate bias for
  // every location at which the filter is applied.
  optional bool shared_biases = 8 [ default = false ];

  // Valid values are ones that divide the area of the output
  // grid in this convolutional layer. For example if this layer
  // produces 32-channel 20x20 output grid, valid values of
  // partialSum are ones which divide 20*20 = 400.
  // I'll update this comments when confirmed
  optional uint32 partial_sum = 9;

  // for dropout
  optional double drop_rate = 10;

  // for HierarchicalSoftmaxLayer and NCELayer
  // the number of classes
  optional uint32 num_classes = 11;

  // the gpu device which the Layer's data in.
  // Only used by ParallelNeuralNetork. Ignored otherwise.
  optional int32 device = 12 [ default = -1 ];

  // for recurrent layer. If true, the recurrence runs from the end to the
  // beginning.
  optional bool reversed = 13 [ default = false ];

  // for lstmemory layer. Different types of nodes have different activation
  // type.
  optional string active_gate_type = 14;
  optional string active_state_type = 15;

  // For NCELayer
  // The number of random negative labels for each sample
  optional int32 num_neg_samples = 16 [ default = 10 ];

  // For NCELayer
  // The distribution for generating the random negative labels.
  // A uniform distribution will be used if not provided
  repeated double neg_sampling_dist = 17 [ packed = true ];

  // For MaxLayer
  // default: output VALUE of MaxLayer. set this flag to true for output INDEX
  // INDEX will be put in Argument::value as double values.
  optional bool output_max_index = 19 [ default = false ];

  /// The filed number 20 have been deprecated.

  // For self-normalized estimation
  optional double softmax_selfnorm_alpha = 21 [ default = 0.1 ];

  /// The filed numbers 22 and 23 have been deprecated.

  // for MDLstmLayer
  repeated bool directions = 24;

  // for CTCLayer
  optional bool norm_by_times = 25;

  // for CostLayers
  optional double coeff = 26 [ default = 1.0 ];

  // for AverageLayer
  // can be set to: 'average', 'sum' or 'squarerootn'
  optional string average_strategy = 27;

  // for error clipping
  optional double error_clipping_threshold = 28 [ default = 0.0 ];

  // for operators used by mixed layer
  repeated OperatorConfig operator_confs = 29;

  // for lambdaCost
  optional int32 NDCG_num = 30;
  optional int32 max_sort_size = 31;

  // for SlopeInterceptLayer
  optional double slope = 32;
  optional double intercept = 33;

  // for CosSimVecMatLayer and CosSimLayer
  optional double cos_scale = 34;

  // for DataNormLayer
  // can be set to: 'z-score', 'min-max' or 'decimal-scaling'
  optional string data_norm_strategy = 36;

  // for bos/eos id
  optional uint32 bos_id = 37;
  optional uint32 eos_id = 38;

  // for max id layer
  optional uint32 beam_size = 39;

  // for seqlastins layer, whether select first instead last
  optional bool select_first = 40 [ default = false ];

  // for seqlastins layer, AverageLayer, MaxLayer and ExpandLayer
  // can be set to: 'non-seq','seq'
  optional string trans_type = 41 [ default = 'non-seq' ];

  // to indicate whether selective_fc layer
  // is used in sequence generation or not
  optional bool selective_fc_pass_generation = 42 [ default = false ];

  // to indicate whether selective_fc layer take its last input to
  // selected several columns and only compute the multiplications
  // between the input matrices and the selected columns of
  // the parameter matrices of this layer.
  // if set false, selective_fc degrades into fc.
  optional bool has_selected_colums = 43 [ default = true ];

  // this parameter is for speed consideration.
  // if number of the selected columns is less than
  // sample number * selective_fc output size * selective_fc_mull_mull_ratio
  // sparse multiplication is used, otherwise, using full multiplication.
  optional double selective_fc_full_mul_ratio = 44 [ default = 0.02 ];

  // to indicate how many threads selective_fc use to to accelate
  // the plain_mul period
  // leave empty or set to 0 to disable multi-thread accleleration
  optional uint32 selective_fc_parallel_plain_mul_thread_num = 45
      [ default = 0 ];

  // for batch normalization layer
  // if set use_global_stats true, will use the loaded mean and variance.
  optional bool use_global_stats = 46;

  // use to compute moving mean and variance.
  optional double moving_average_fraction = 47 [ default = 0.9 ];

  // bias size
  optional uint32 bias_size = 48 [ default = 0 ];

  // this parameter can be used as a user-defined parameter when necessary,
  // without changing the proto file.
  // e.g., when a new layer with a user-defined parameter is implemented,
  // it can be used to pass that parameter, without modifying the proto file.
  // string type is used for flexibility: different types can be converted
  // to string and reinterpreted in the user's own layer implementation.
  optional string user_arg = 49;

  // to indicate rectangle image data
  optional uint64 height = 50;
  optional uint64 width = 51;

  // blank label used in ctc loss
  optional uint32 blank = 52 [ default = 0 ];

  // stride parameter for seqlastins layer, AverageLayer, MaxLayer, which
  // controls the scope of pooling operation. can be set > 0.
  // leave empty or set to -1 to disable this stride pooling.
  optional int32 seq_pool_stride = 53 [ default = -1 ];

  // for crop layer
  optional int32 axis = 54 [ default = 2 ];
  repeated uint32 offset = 55;
  repeated uint32 shape = 56;

  // for HuberRegressionLoss
  optional double delta = 57 [ default = 1.0 ];

  // for 3D data
  optional uint64 depth = 58 [ default = 1 ];

  // for switch order layer
  optional ReshapeConfig reshape_conf = 59;

  // for batch normalization layer
  // The small constant added to the variance to improve numeric stability.
  optional double epsilon = 60 [ default = 0.00001 ];

  // for factorization machine layer
  optional uint32 factor_size = 61;
}

message EvaluatorConfig {
  required string name = 1;
  required string type = 2;
  repeated string input_layers = 3;

  // Used by ChunkEvaluator
  // one of "IOB", "IOE", "IOBES"
  optional string chunk_scheme = 4;
  // number of chunk types other than "other"
  optional int32 num_chunk_types = 5;

  // Used by PrecisionRecallEvaluator and ClassificationErrorEvaluator
  // For multi binary labels: true if output > classification_threshold
  optional double classification_threshold = 6 [ default = 0.5 ];
  // The positive label. -1 means average precision and recall
  optional int32 positive_label = 7 [ default = -1 ];

  // load dict from this file
  optional string dict_file = 8;

  // dump result in this file
  optional string result_file = 9;

  // top # results for max id printer
  optional int32 num_results = 10 [ default = 1 ];

  // whether to delimit the sequence in the seq_text_printer
  optional bool delimited = 11 [ default = true ];

  // Used by ChunkEvaluator
  // chunk of these types are not counted
  repeated int32 excluded_chunk_types = 12;

  // Used by ClassificationErrorEvaluator
  // top # classification error
  optional int32 top_k = 13 [ default = 1 ];

  // Used by DetectionMAPEvaluator
  optional double overlap_threshold = 14 [ default = 0.5 ];

  optional int32 background_id = 15 [ default = 0 ];

  optional bool evaluate_difficult = 16 [ default = false ];

  optional string ap_type = 17 [ default = "11point" ];
}

message LinkConfig {
  required string layer_name = 1;
  required string link_name = 2;
  // If true, this link has sub-sequence
  optional bool has_subseq = 3 [ default = false ];
}

message MemoryConfig {
  required string layer_name = 1;
  required string link_name = 2;

  optional string boot_layer_name = 3;
  optional string boot_bias_parameter_name = 4;
  optional string boot_bias_active_type = 5;
  optional uint32 boot_with_const_id = 7;

  // memory is a sequence, initailized by a sequence boot layer
  optional bool is_sequence = 6 [ default = false ];
}

message GeneratorConfig {
  required uint32 max_num_frames = 1;
  required string eos_layer_name = 2;
  optional int32 num_results_per_sample = 3 [ default = 1 ];

  // for beam search
  optional int32 beam_size = 4 [ default = 1 ];

  optional bool log_prob = 5 [ default = true ];
}

message SubModelConfig {
  required string name = 1;
  repeated string layer_names = 2; // selected layers in sub model
  repeated string input_layer_names = 3;
  repeated string output_layer_names = 4;
  repeated string evaluator_names = 5;

  optional bool is_recurrent_layer_group = 6 [ default = false ];

  // If true, the recurrence runs from the end to the beginning.
  optional bool reversed = 7 [ default = false ];

  // name and link name of memory
  repeated MemoryConfig memories = 8;

  // if use recurrent layer group, all layers in submodel will postfix by
  // "_in_"+submodel.name, so we add a name pair to link between
  // root model and layer group,
  // note that these in/out layers are not input/output of the network.
  repeated LinkConfig in_links = 9;
  repeated LinkConfig out_links = 10;

  optional GeneratorConfig generator = 11;

  // the id of inlink which share info with outlinks, used in recurrent layer
  // group
  optional int32 target_inlinkid = 12;
}

message ModelConfig {
  // type of the model.
  // Currently, "nn", "recurrent_nn" and "recursive_nn" are supported
  required string type = 1 [ default = "nn" ];

  // layers should be ordered in such a way that the forward propagation
  // can be correctly executed by going from the first layer to the last layer
  repeated LayerConfig layers = 2;

  repeated ParameterConfig parameters = 3;

  // Input layers should have the same order as the data streams provided
  // by the data provider. The type of input layers should be "data"
  repeated string input_layer_names = 4;

  // For training, the type of a output layer is usually cost layer.
  // For prediction, they should be the actual output layers.
  repeated string output_layer_names = 5;

  repeated EvaluatorConfig evaluators = 6;

  repeated SubModelConfig sub_models = 8;

  // For External Machine, defining how to split a neural network
  // into multiple parts.
  optional ExternalConfig external_config = 9;
};

// Paddle/proto/OptimizerConfig.proto

message SGDConfig {
  // SGD
  // momentum: float >= 0. Parameter updates momentum.
  // decay: float >= 0. Learning rate decay over each update.
  // nesterov: boolean. Whether to apply Nesterov momentum.
  optional double momentum = 21 [ default = 0.0 ];
  optional double decay = 23 [ default = 0.0 ];
  optional bool nesterov = 24 [ default = false ];
}

message AdadeltaConfig {
  // Adadelta
  // It is recommended to leave it at the default value.
  // rho: float >= 0.
  // epsilon: float >= 0. Fuzz factor.
  // decay: float >= 0. Learning rate decay over each update.

  // reference : [Adadelta - an adaptive learning rate
  // method](http://arxiv.org/abs/1212.5701)
  optional double rho = 33 [ default = 0.90 ];
  optional double epsilon = 31 [ default = 1e-5 ];
  optional double decay = 32 [ default = 0.0 ];
}

message AdagradConfig {
  // Adagrad
  // epsilon: float >= 0.
  // decay: float >= 0. Learning rate decay over each update.

  // reference : [Adaptive Subgradient Methods for Online Learning and
  // Stochastic
  // Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  optional double epsilon = 41 [ default = 1e-5 ];
  optional double decay = 42 [ default = 0.0 ];
}

message AdamConfig {
  // Adaj
  // beta_1: float, 0 < beta < 1. Generally close to 1.
  // beta_2: float, 0 < beta < 1. Generally close to 1.
  // epsilon: float >= 0. Fuzz factor.
  // decay: float >= 0. Learning rate decay over each update.
  // reference : [Adam - A Method for Stochastic
  // Optimization](http://arxiv.org/abs/1412.6980v8)
  optional double beta_1 = 41;
  optional double beta_2 = 42;
  optional double epsilon = 43;
  optional double decay = 44;
}

message ConstLrConfig {
  // learninRate Policy
  optional double learning_rate = 1 [ default = 1.0 ];
}

message LinearLrConfig {
  // learninRate Policy
  optional double learning_rate = 1 [ default = 1.0 ];
  optional double lr_decay_a = 2;
  optional double lr_decay_b = 3;
}

message TensorProto {
  enum DataType {
    PADDLE_ELEMENT_TYPE_INT32 = 0;
    PADDLE_ELEMENT_TYPE_UINT32 = 1;
    PADDLE_ELEMENT_TYPE_INT64 = 2;
    PADDLE_ELEMENT_TYPE_UINT64 = 3;
    PADDLE_ELEMENT_TYPE_FLOAT32 = 4;
    PADDLE_ELEMENT_TYPE_FLOAT64 = 5;
  }
  optional DataType data_type = 1;
  repeated bytes content = 2;
}

message LrPolicyState {
  // learninRate Policy
  optional double learning_rate = 1 [ default = 1.0 ];
  optional double lr_decay_a = 2;
  optional double lr_decay_b = 3;
}

message SGDOptimizerState {
  optional LrPolicyState lr_state = 101;
  optional double num_sample_passed = 104;
  // state
  optional TensorProto parameter = 1;
  optional TensorProto momentums = 2;
}

message AdadeltaOptimizerState {
  // learning rate policy
  optional LrPolicyState lr_state = 101;
  optional double num_sample_passed = 104;
  // state
  optional TensorProto parameter = 1;
  optional TensorProto accum_gradient = 2;
  optional TensorProto accum_delta = 3;
  optional TensorProto update_delta = 4;
}

message AdagradOptimizerState {
  optional LrPolicyState lr_state = 101;
  optional double num_sample_passed = 104;
  // state
  optional TensorProto parameter = 1;
  optional TensorProto accum_gradient = 2;
}

message AdamOptimizerState {
  optional LrPolicyState lr_state = 101;
  optional double num_sample_passed = 104;
  // state
  optional TensorProto parameter = 1;
  optional TensorProto momentums = 2;
  optional TensorProto velocitys = 3;
}

message OptimizerConfig {
  enum Optimizer {
    SGD = 1;
    Adadelta = 2;
    Adagrad = 3;
    Adam = 4;
  }
  optional Optimizer optimizer = 1;
  optional SGDConfig sgd = 3;
  optional AdadeltaConfig adadelta = 4;
  optional AdagradConfig adagrad = 5;
  optional AdamConfig adam = 6;

  enum LrPolicy {
    Const = 0;
    Linear = 1;
  }
  optional LrPolicy lr_policy = 11;
  optional ConstLrConfig const_lr = 12;
  optional LinearLrConfig linear_lr = 13;

  // common config of optimizer
  // gradient clip when L2 exceeding value
  optional double clip_norm = 101;
  // gradient clip when L1 exceeding value
  optional double clip_value = 102;
}

// Paddle/proto/ParameterConfig.proto


This repository
Search
Pull requests
Issues
Marketplace
Explore
 @mengjiahao
 Sign out
 Watch 545
  Unstar 6,181  Fork 1,614 PaddlePaddle/Paddle
 Code  Issues 969  Pull requests 210  Projects 28  Wiki  Insights
Branch: develop Find file Copy pathPaddle/proto/ParameterConfig.proto
1d4fa24  on Aug 4, 2017
@gangliao gangliao ClangFormat for proto and cuda
3 contributors @NHZlX @reyoung @gangliao
RawBlameHistory    
84 lines (73 sloc)  3.31 KB
/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
syntax = "proto2";

package paddle;

/**
 * Configuration structure for parameter
 */

enum ParameterInitStrategy {
  PARAMETER_INIT_NORMAL = 0;
  PARAMETER_INIT_UNIFORM = 1;
}

message ParameterUpdaterHookConfig {
  // hook type such as  'pruning'
  required string type = 1;
  // this represents the ratio of zero element to be set by the Parameter
  optional double sparsity_ratio = 2 [ default = 0.6 ];
}

message ParameterConfig {
  required string name = 1;
  required uint64 size = 2;
  optional double learning_rate = 3 [ default = 1.0 ];
  optional double momentum = 4 [ default = 0.0 ];
  optional double initial_mean = 5 [ default = 0.0 ];
  optional double initial_std = 6 [ default = 0.01 ];
  // use L2-regularization if decay_rate set and decay_rate_l1 not set
  optional double decay_rate = 7 [ default = 0.0 ];
  // use L1-regularization if decay_rate_l1 set
  optional double decay_rate_l1 = 8 [ default = 0.0 ];
  // dims of Parameter, e.g. dims[0] as height, dims[1] as width..
  repeated uint64 dims = 9;
  // the gpu device which the parameter in.
  // Only used by ParallelNeuralNetork. Ignored otherwise.
  optional int32 device = 10 [ default = -1 ];
  // how to init the parameter: 0 -> normal, 1 -> uniform
  // 0: treat initial_mean as mean, intial_std as standard deviation
  // 1: range is (initial_mean - initial_std) to (initial_mean + initial_std)
  optional int32 initial_strategy = 11 [ default = 0 ];
  // define the variance when init the parameter, by height of the Matrix
  optional bool initial_smart = 12 [ default = false ];
  // apply regularization every # batches
  optional int32 num_batches_regularization = 13 [ default = 1 ];
  // if is_sparse is true, para is sparse, else para is dense
  optional bool is_sparse = 14 [ default = false ];
  // if para is sparse, format should be "csc" or "csr", empty means is not
  // sparse
  optional string format = 15 [ default = "" ];
  // sparse remote update or not
  optional bool sparse_remote_update = 16 [ default = false ];
  // gradient clipping threshold, no clipping by default
  optional double gradient_clipping_threshold = 17 [ default = 0.0 ];
  // static parameters are fixed when training
  optional bool is_static = 18 [ default = false ];
  // para_id should NOT be set by config_parser. It is for
  // internal use.
  optional uint64 para_id = 19;

  repeated ParameterUpdaterHookConfig update_hooks = 20;
  // setup load mat -> csr
  optional bool need_compact = 21 [ default = false ];
  // whether to do sparse update for this parameter
  optional bool sparse_update = 22 [ default = false ];

  // whether this parameter is shared or not.
  optional bool is_shared = 23 [ default = false ];
  // parameter block size
  optional uint64 parameter_block_size = 24 [ default = 0 ];
}

// Paddle/proto/ParameterServerConfig.proto

/**
 * Configuration structure for ParameterClient2.
 */
message ParameterClientConfig { required int32 trainer_id = 1; }

/**
 * Configuration structure for ParameterServer2.
 */
message ParameterServerConfig {
  // Number of ports for sending dense parameter,
  // following ports on parameter server will be visited
  // for sending dense parameter: [port, port+ports_num-1]
  required int32 ports_num = 1 [ default = 1 ];
  // Number of ports for sending sparse parameter,
  // following ports on parameter server will be visited
  // for sending sparse parameter:
  // [port+ports_num, port+ports_num+ports_num_for_sparse-1]
  required int32 ports_num_for_sparse = 2 [ default = 0 ];
  // network device name for pservers
  required string nics = 3 [ default = "xgbe0,xgbe1" ];
  required string rdma_tcp = 4 [ default = "tcp" ];
  // Listening port for pserver
  required int32 port = 5 [ default = 20134 ];
  // number of gradient servers
  required int32 num_gradient_servers = 6 [ default = 1 ];
  // number of threads for sync op exec
  required int32 pserver_num_threads = 7 [ default = 1 ];
  // control config_.async_lagged_grad_discard_ratio() min value
  required double async_lagged_ratio_min = 8 [ default = 1.0 ];
  // if async_lagged_grad_discard_ratio is not set in trainer_config.conf
  // use it as defalut value
  required double async_lagged_ratio_default = 9 [ default = 1.5 ];
}

// Paddle/proto/TrainerConfig.proto

message OptimizationConfig {
  optional int32 batch_size = 3 [ default = 1 ];
  required string algorithm = 4 [ default = "async_sgd" ];
  optional int32 num_batches_per_send_parameter = 5 [ default = 1 ];
  optional int32 num_batches_per_get_parameter = 6 [ default = 1 ];

  required double learning_rate = 7;
  optional double learning_rate_decay_a = 8 [ default = 0 ];
  optional double learning_rate_decay_b = 9 [ default = 0 ];
  optional string learning_rate_schedule = 27 [ default = "constant" ];
  // learning rate will be scaled according to learning_rate_schedule
  // 1), constant:
  // lr = learning_rate
  // 2), poly:
  // lr = learning_rate *
  //      pow(1 + learning_rate_decay_a * num_samples_processed,
  //          -learning_rate_decay_b)
  // 3), exp:
  // lr = learning_rate *
  //      pow(learning_rate_decay_a,
  //          num_samples_processed / learning_rate_decay_b)
  // 4), discexp:
  // lr = learning_rate *
  //      pow(learning_rate_decay_a,
  //          floor(num_samples_processed / learning_rate_decay_b))
  // 5), linear:
  // lr = max(learning_rate - learning_rate_decay_a * num_samples_processed,
  //          learning_rate_decay_b)

  // owlqn related
  // L1-regularization
  optional double l1weight = 10 [ default = 0.1 ];
  // L2-regularization
  optional double l2weight = 11 [ default = 0 ];
  // "c1" in wolfe condition: if (newobj <= oldobj + c1 * origDirDeriv * step)
  // then accept the step
  optional double c1 = 12 [ default = 0.0001 ];
  // multiply the step with "backoff", when wolfe condition doesn't satisfy
  optional double backoff = 13 [ default = 0.5 ];
  // how many "s"s and "y"s are kept in owlqn
  optional int32 owlqn_steps = 14 [ default = 10 ];
  // accept the step if encountered "max_backoff" times of "reduce the step"
  optional int32 max_backoff = 15 [ default = 5 ];
  // L2-regularization coefficient is reduced linearly from iteration 0 to
  // "l2weight_zero_iter", and set to 0 after "l2weight_zero_iter"
  // iterations. set "l2weight_zero_iter" to 0 to disable this strategy.
  optional int32 l2weight_zero_iter = 17 [ default = 0 ];

  // averaged sgd
  // About average_window * numBatchProcessed parameter are used
  // for average. To be accurate, between average_window * numBatchProcessed
  // and 2 * average_window * numBatchProcessed parameters are used for
  // average.
  optional double average_window = 18 [ default = 0 ];
  optional int64 max_average_window = 19 [ default = 0x7fffffffffffffff ];

  //////////////////////////
  // Options Adaptive SGD //
  //////////////////////////

  // learning method for sgd/asgd, such as "momentum", "adagrad", "adadelta",
  // "rmsprop"
  // default learning method("momentum") use global decayed learning rate with
  // momentum.
  // "adagrad", "adadelta" and "rmsprop" can set momentum too.
  optional string learning_method = 23 [ default = "momentum" ];
  optional double ada_epsilon = 24 [ default = 1e-6 ];
  optional double ada_rou = 26 [ default = 0.95 ];

  // Force to do average in cpu in order to save gpu memory usage
  optional bool do_average_in_cpu = 25 [ default = false ];

  // delta add rate in pserver, used while num_batches_per_send_parameter>1
  // will be divided by #machines automatically.
  optional double delta_add_rate = 28 [ default = 1.0 ];

  // We split a large size into smaller mini-batches, whose sizes are
  // determined by mini_batch_size. It only takes effect when there is
  // an ExternalMachine.
  optional int32 mini_batch_size = 29 [ default = 128 ];

  // automatically set if any one of parameters set sparse remote update flag
  optional bool use_sparse_remote_updater = 30 [ default = false ];

  // how to update center parameter and feedback to local parameter,
  // when use local sgd update in cluster training.
  // A option is elastic_average, proposed by the paper: Deep learning with
  // elastic averaging SGD.
  // If use elastic_average method, every trainer node should sample from whole
  // data sets.
  optional string center_parameter_update_method = 31 [ default = "average" ];

  // shrink sparse parameter value
  // only works if parameter is remote sparse update and has L1 decay rate
  optional double shrink_parameter_value = 32 [ default = 0 ];

  ////////////////////////////
  // Options Adam Optimizer //
  ////////////////////////////
  optional double adam_beta1 = 33 [ default = 0.9 ];
  optional double adam_beta2 = 34 [ default = 0.999 ];
  optional double adam_epsilon = 35 [ default = 1e-8 ];

  // arguments for learning rate scheduler
  // Format: num1:rate1,num2:rate2,...,numK:rateK
  // For learning_rate_schedule="manual", num is the number of samples,
  // For learning_rate_schedule="pass_manual",
  //  num is the number of passes (starting from 0)
  optional string learning_rate_args = 36 [ default = "" ];

  // for async sgd gradient commit control.
  // when async_lagged_grad_discard_ratio * num_gradient_servers commit passed,
  // current async gradient will be discard silently.
  optional double async_lagged_grad_discard_ratio = 37 [ default = 1.5 ];

  // global threshold for gradient clipping
  optional double gradient_clipping_threshold = 38 [ default = 0.0 ];
};

message TrainerConfig {
  optional ModelConfig model_config = 1;
  optional DataConfig data_config = 2;
  required OptimizationConfig opt_config = 3;
  optional DataConfig test_data_config = 4;
  repeated string config_files = 5;

  // the directory to save/load model files for each training path
  optional string save_dir = 6 [ default = "./output/model" ];

  // Path of the initial model parameters.
  // If it was set, start_pass will be ignored.
  optional string init_model_path = 7;

  // Start training from this pass.
  // Will load parameter from the previous pass.
  optional int32 start_pass = 8 [ default = 0 ];

  // file path to the trainer config file
  optional string config_file = 9;
}

// Paddle/proto/ParameterService.proto

/**
 * Various structs for communicating with parameter server
 */
enum ParameterUpdateMode {
  // Set parameter
  PSERVER_UPDATE_MODE_SET_PARAM = 0;      // use local param
  PSERVER_UPDATE_MODE_SET_PARAM_ZERO = 1; // set zero param

  // Update parameter once a gradient is received
  PSERVER_UPDATE_MODE_ASYNC_SGD = 2;

  // Accumulate gradient
  PSERVER_UPDATE_MODE_ADD_GRADIENT = 3;

  // Average parameters
  PSERVER_UPDATE_MODE_AVERAGE_PARAMETER = 4;

  // No update. Only get parameters back.
  PSERVER_UPDATE_MODE_GET_PARAM = 5;
  PSERVER_UPDATE_MODE_GET_PARAM_SPARSE = 6; // only get sparse rows
};

message ParameterBlock {
  // it accurately means parameter id.
  required uint64 para_id = 1;
  // global sparse row or dense block for each block in parameter
  required uint64 block_id = 2;
  // offset in (local) storage
  required uint64 begin_pos = 3;
  // actual size of block, size for last block is [endDim -beginDim],
  // others is parameter_block_size in ParameterConfig
  required uint64 block_size = 4;
}

enum PServerStatus {
  PSERVER_STATUS_NOT_SET = 0;
  PSERVER_STATUS_PARAMETER_READY = 1;
};

enum BatchStatus {
  BATCH_START = 0;
  BATCH_ON = 1;
  BATCH_FINISH = 2;
  BATCH_START_AND_FINISH = 3;
};

message SendParameterRequest {
  required ParameterUpdateMode update_mode = 1;
  repeated ParameterBlock blocks = 2;
  required bool send_back_parameter = 3;

  // number of samples used for calculating this update
  optional int64 num_samples = 4;

  // cost will be used to calculate global objective value
  optional double cost = 5;

  required BatchStatus batch_status = 6;

  optional int32 trainer_id = 7;

  // send back parameter type on pserver, PARAMETER_VALUE by default
  optional int32 send_back_parameter_type = 8 [ default = 0 ];

  // forwardbackward time in usec
  optional uint64 forwardbackward_time = 9;
}

message WaitPassStartRequest {}

message WaitPassStartResponse {}

message WaitPassFinishRequest {}

message WaitPassFinishResponse {}

enum SyncObject {
  SYNC_DEFAULT = 0; // wait for the synchronizeBarrier_
  SYNC_DATA = 1;    // wait for the synchronizeDataBarrier_
}

message SynchronizeRequest {
  required SyncObject sync_object_id = 1 [ default = SYNC_DEFAULT ];

  optional int32 trainer_id = 2;
}

message SynchronizeResponse {}

message SendParameterResponse { repeated ParameterBlock blocks = 1; }

message SetConfigRequest {
  repeated ParameterConfig param_configs = 1;
  required OptimizationConfig opt_config = 2;
  required string save_dir = 4;
  required int32 server_id = 5;
  required bool is_sparse_server = 6;
}

message SetConfigResponse {}

message GetStatusRequest {}

message GetStatusResponse { required PServerStatus status = 1; }

message SetStatusRequest { required PServerStatus status = 1; }

message SetStatusResponse {}

// create a column vector. The size is the dimension of parameter
message CreateVectorRequest {}

message CreateVectorResponse {
  // error message. Empty if success
  optional string return_message = 1;

  required int64 handle = 2;
}

message ReleaseVectorRequest { required int64 handle = 1; }

message ReleaseVectorResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

// Create a column major matrix. The number of rows is the dimension
// of parameter. The number of columns is specifed by num_cols
message CreateMatrixRequest { required int32 num_cols = 1; }

message CreateMatrixResponse {
  // error message. Empty if success
  optional string return_message = 1;

  required int64 handle = 2;
}

message ReleaseMatrixRequest { required int64 handle = 1; }

message ReleaseMatrixResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

/**
 * The operations are defined using the variables commented at Operation
 * and OperationResult
 */
enum MatrixVectorOperation {
  // r = u^T u
  PSERVER_OP_utu = 0;

  // r = u^T v
  PSERVER_OP_utv = 1;

  // u = a u
  PSERVER_OP_au = 2;

  // v = a u + b v
  PSERVER_OP_au_bv = 3;

  // u = a A x + b u
  PSERVER_OP_aAx_bu = 4;

  // Stochastic gradient update
  PSERVER_OP_SGD = 5;

  // u = a
  PSERVER_OP_RESET = 6;

  // v = u
  PSERVER_OP_COPY = 7;

  // w = a u + b v + c w
  PSERVER_OP_au_bv_cw = 8;

  // owlqn: MakeSteepestDescDir
  PSERVER_OP_MAKE_STEEPEST_DESC_DIR = 9;

  // owlqn: FixDirSigns
  PSERVER_OP_FIX_DIR_SIGNS = 10;

  // owlqn: DirDeriv
  PSERVER_OP_DIR_DERIV = 11;

  // owlqn: FixOmegaSigns
  PSERVER_OP_FIX_OMEGA_SIGNS = 12;

  // Get overall cost
  PSERVER_OP_COST = 13;

  // Pass control
  PSERVER_OP_START_PASS = 14;
  PSERVER_OP_FINISH_PASS = 15;

  // randomize value
  PSERVER_OP_RANDOMIZE = 16;

  // call optimizer apply
  PSERVER_OP_APPLY = 17;
}

message ProtoVector {
  required int64 dim = 1;
  repeated double values = 2 [ packed = true ];
}

message ProtoMatrix {
  required int64 num_rows = 1;
  required int64 num_cols = 2;
  repeated double values = 3 [ packed = true ];
}

message Operation {
  required MatrixVectorOperation operation = 1;

  // vector handles created on the pserver
  repeated int64 pvectors = 2; // u, v, w

  // matrix handles created on the pserver
  repeated int64 pmatrices = 3; // A, B, C

  repeated double scalars = 4;       // a, b, c
  repeated ProtoVector vectors = 5;  // x, y, z
  repeated ProtoMatrix matrices = 6; // X, Y, Z
}

message OperationResult {
  // error message. Empty if success
  optional string return_message = 1;
  //
  repeated double scalars = 2;       // d, e, f
  repeated ProtoVector vectors = 3;  // p, q, r
  repeated ProtoMatrix matrices = 4; // P, Q, R
}

message DoOperationRequest {
  repeated Operation operations = 1;

  // If true, wait for gradient to be ready before starting the operations
  required bool wait_for_gradient = 2;

  // If true, send back the parameter to clients after the operations are
  // finished
  required bool send_back_parameter = 3;

  // If true, and if all clients call waitPassFinish,
  // signal all clients finish the pass
  required bool release_pass = 4;
}

message DoOperationResponse {
  // error message. Empty if success
  optional string return_message = 1;

  repeated OperationResult results = 2;

  required bool pass_finish = 3;
}

message LoadValueRequest { required string dir_name = 1; }

message LoadValueResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

message SaveValueRequest { required string dir_name = 1; }

message SaveValueResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

enum DataUpdateMode {
  // Client send it's own data to pserver
  DATA_UPDATE_MODE_SET_OWN = 0;
  // Client get all user data from all pservers
  DATA_UPDATE_MODE_GET_ALL = 1;
  // Client send it's own ref feature to pserver
  DATA_UPDATE_MODE_SET_REF = 2;
  // Client get all ref featuers from all pservers
  DATA_UPDATE_MODE_GET_REF = 3;
  // Client send it's own ref label to pserver
  DATA_UPDATE_MODE_SET_REF_LABEL = 4;
  // Client get all ref labels from all pservers
  DATA_UPDATE_MODE_GET_REF_LABEL = 5;
  // Client send it's own ref grad to pserver
  DATA_UPDATE_MODE_SET_REF_GRAD = 6;
  // Client get all ref grad from all pservers
  DATA_UPDATE_MODE_GET_REF_GRAD = 7;
}

enum SendDataType {
  DATA_REF = 0;
  DATA_REFLABEL = 1;
  DATA_REFGRAD = 2;
  DATA_REDUCE_SUM = 3;
}

enum TransDataType {
  TRANS_INT32 = 0;
  TRANS_UINT32_T = 1;
  TRANS_INT64_T = 2;
  TRANS_UINT64_T = 3;
  TRANS_FLOAT = 5;
  TRANS_DOUBLE = 6;
}

message DataBlock {
  // total byte size of this data blcok
  required uint64 total_size = 1;
  // byte size of one data type
  required int32 data_size = 2;
  // data_type
  optional TransDataType data_type = 3 [ default = TRANS_DOUBLE ];
}

message SendDataRequest {
  required SendDataType type = 1;
  required DataUpdateMode update_mode = 2;
  repeated DataBlock blocks = 3;
  required uint64 client_id = 4;
  required uint64 server_id = 5;
}

message SendDataResponse {
  required SendDataType type = 1;
  repeated DataBlock blocks = 2;
  required uint64 server_id = 3;
}