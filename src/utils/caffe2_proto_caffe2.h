
// caffe2/caffe2/proto/caffe2.proto

#ifndef BUBBLEFS_UTILS_CAFFE2_PROTO_CAFFE2_H_
#define BUBBLEFS_UTILS_CAFFE2_PROTO_CAFFE2_H_

#include <map>
#include <string>
#include <vector>
#include "platform/types.h"

//syntax = "proto2";
//package caffe2; 

namespace bubblefs {
namespace mycaffe2 {

typedef std::string bytes;   

// Data type for caffe2 Index/Size. We use size_t to be safe here as well as for
// large matrices that are common in sparse math.
typedef int64_t TIndex; 

// Note(Yangqing): NVCC does not play well with unordered_map on some platforms,
// forcing us to use std::map instead of unordered_map. This may affect speed
// in some cases, but in most of the computation code we do not access map very
// often, so it should be fine for us. I am putting a CaffeMap alias so we can
// change it more easily if things work out for unordered_map down the road.
template <typename Key, typename Value>
using CaffeMap = std::map<Key, Value>;
// using CaffeMap = std::unordered_map;
  
// A few notes about the Caffe2's protobuffer convention:
// (1) Most objects are registered by their types, such as operators and nets.
//     For these, we have a string-type field "type" for registration purposes.
// (2) We do not use extension because that used to create quite some conflicts
//     in Caffe's protobuf design.
// (3) We have not used any proto3 specific features, such as Any or Map. This
//     is mainly for backward compability purposes but we may consider using
//     those in the future.

// TensorProto stores serialized Tensor objects.

struct DeviceOption;
struct NetDef;
  
struct TensorProto {
  // The dimensions in the tensor.
  std::vector<int64> dims;
  enum DataType {
    UNDEFINED,
    FLOAT = 1,  // float
    INT32 = 2, // int
    BYTE = 3, // BYTE, when deserialized, is going to be restored as uint8.
    STRING = 4,  // string
    // Less-commonly used data types.
    BOOL = 5,  // bool
    UINT8 = 6,  // uint8_t
    INT8 = 7, // int8_t
    UINT16 = 8,  // uint16_t
    INT16 = 9,  // int16_t
    INT64 = 10,  // int64_t
    FLOAT16 = 1,  // caffe2::__f16, caffe2::float16
    DOUBLE = 13,  // double
  };
  DataType data_type; //[default = FLOAT];
  // For float
  std::vector<float> float_data;
  // For int32, uint8, int8, uint16, int16, bool, and float16
  // Note about float16: in storage we will basically convert float16 byte-wise
  // to unsigned short and then store them in the int32_data field.
  std::vector<int32> int32_data;
  // For bytes
  bytes byte_data;
  // For strings
  bytes string_data;
  // For double
  std::vector<double> double_data;
  // For int64
  std::vector<int64> int64_data;
  // Optionally, a name for the tensor.
  string name;

  // Optionally, a TensorProto can contain the details about the device that
  // it was serialized from. This is useful in cases like snapshotting a whole
  // workspace in a multi-GPU environment.
  DeviceOption device_detail;
  // When loading from chunks this is going to indicate where to put data in the
  // full array. When not used full data have to be present
  struct Segment {
    std::vector<int64> begin;
    std::vector<int64> end;
  };
  Segment segment;
};

struct QTensorProto {
  std::vector<int64> dims;
  int32 precision;
  double scale;
  double bias;
  bool is_signed;
  std::vector<int32> data;
  string name;
};

// TensorProtos stores multiple TensorProto objects in one single proto. This
// is useful for small tensors; For anything big, consider using a DB for
// storage.
struct TensorProtos {
  std::vector<TensorProto> protos;
};

struct TensorShape {
  std::vector<int64> dims;
  TensorProto::DataType data_type; // [default = FLOAT];
  std::vector<int32> unknown_dims;
  bool unknown_shape;
  string name;
};

struct TensorShapes {
  std::vector<TensorShape> shapess;
};

// A named argument containing either singular float, integer and string
// values, or repeated float, int and string arrays.
struct Argument {
  string name;
  float f;
  int64 i;
  bytes s;
  NetDef n;
  std::vector<float> floats;
  std::vector<int64> ints;
  std::vector<bytes> strings;
  std::vector<NetDef> nets;
};

// DeviceType that Caffe2 currently supports.
// Note: if you add a device type, make sure you add the corresponding device
// line in core/blob_serialization.cc.
enum DeviceType {
  CPU = 0,                    // In default, we will use CPU.
  CUDA = 1,                   // CUDA.
  MKLDNN = 2,                 // Reserved for explicit MKLDNN
  OPENGL = 3,                 // OpenGL
  // Change the following number if you add more devices in the code.
  COMPILE_TIME_MAX_DEVICE_TYPES = 4,
  ONLY_FOR_TEST = 20901701   // This device type is only for test.
};

// Device-specific options. We do not distinguish DeviceOption protos for
// different DeviceTypes, so currently all devices share the same DeviceOption
// proto. Fields that are specific to a device type is ignored if the type does
// not match.
// Note: if you add fields to the DeviceOption, make sure you add the
// corresponding changes to IsSameDevice() function in utils/proto_utils.{h,cc}.
struct DeviceOption {
  // [general] Options that need to be carried out before running the execution.
  // optional DeviceType device_type = 1 [ default = CPU ];
  int32 device_type; // 0 is CPU.
  // [CUDA specific] the cuda gpu id.
  int32 cuda_gpu_id;
  // [general] The random seed to start the device random number generator with.
  uint32 random_seed;
  // [general] What node this op should execute on.
  // Used for net transformation purposes. Must be empty at execution time.
  string node_name;
};

// Operator Definition.
struct OperatorDef {
  std::vector<string> input; // the name of the input blobs
  std::vector<string> output; // the name of output top blobs
  string name; // the operator name. This is optional.
  // the operator type. This is needed to create the object from the operator
  // registry.
  string type;
  std::vector<Argument> arg;

  // The device option that the operator should run under.
  DeviceOption device_option;

  // Optionally, one can specify an engine when there are multiple
  // implementations available simultaneously for one device type.
  // If one specifies an engine but that engine does not exist in the compiled
  // Caffe2 binary, Caffe2 will fall back to the default engine of that device
  // type.
  string engine;


  // Additional 'fake' inputs used for expressing control dependencies
  // in the operator graph. This can be used to ensure that an
  // operator does not run until another operator is ready, for e.g.
  // scheduling control. These are not passed as actual inputs to the
  // Operator implementation, and are only used by the Net class for
  // scheduling purposes.
  std::vector<string> control_input;

  // is_gradient_op argument is only used as a hint in shape inference
  // and has no runtime significance
  bool is_gradient_op;
};

// Network definition.
struct NetDef {
  string name; // the network's name
  // Operators that the network contains.
  // Note: this is not named "operator" because that is a reserved word in C++.
  std::vector<OperatorDef> op;

  // The type of network that the net should be run with. This routes the
  // network instantiation to different execution modes. The default mode,
  // "simple", runs the operators in a sequential way as the original Caffe
  // implementation does.
  string type;

  // the number of workers, if the operators in the network is to be carried out
  // in parallel.
  // Note: This is to be deprecated. Using the arg field with "num_workers" as
  // key.
  int32 num_workers;

  // The device option for the network. If a network has a specific device
  // option and one of its operators does not have it set, we will copy over the
  // device option to the operator. This allows us to basically avoid putting
  // device options at every operator.
  DeviceOption device_option;

  std::vector<Argument> arg;

  // Two optional fields to declare external input and output of a net.
  // If these two are set, when a net is created, we will sanity check for
  // every op whether its input is declared (either as an external input,
  // or as an intermediate blob created by one of the ops), and sanity check
  // if all blobs in external_output are produced.
  //
  // In cases of memory optimization, declaring external_input and
  // external_output also ensures that storage of these blobs are persistent:
  // for any blob in external_input and external_output, after a network run
  // finishes, their content are actually the right content. Any intermediate
  // blobs' contents may be overwritten.
  std::vector<string> external_input;
  std::vector<string> external_output;
};

// ExecutionStep is actually a sort-of-hacky way we simulate iteration right
// now.
struct ExecutionStep {
  // ExecutionStep should either contain a set of substeps, or a set of
  // network names to run in this execution step. They should NOT both be set
  // at the same time.
  string name;
  // An execution step could be recursive, in which it involves a set of
  // substeps.
  ExecutionStep *substep;
  // Alternatively, an execution step could involve one or more networks.
  // Note that you cannot have both substeps and networks. Choose one.
  // Note that an execution step refers networks by their name. The actual
  // network definition of the same name should be included in the network field
  // of the plan. The reason is that a network object might hold internal states
  // (think of a data layer), so we want to have the same network object that
  // multiple steps could ask to run.
  std::vector<string> network;
  // Number of iterations to run this step. The substeps or the networks
  // specified will be run sequentially, and one sequential run is considered
  // one iteration. If this is not set, the number of iterations is assumed to
  // be 1.
  int64 num_iter;

  // Criteria network specifies a single output (TensorCPU<bool>) of
  // size (1), is run on every iteration by the executor, and
  // execution terminates when the output[0] is `false`.
  string criteria_network = 5;

  // DEPRECATED. Use `run_every_ms`.
  string report_net;
  int32 report_interval;

  // If provided, execute this step at every time interval (in millisecs)
  // while its sibiling execution steps execute in parallel. This step is
  // guaranteed to run at least once after all non-interval siblings finished.
  int64 run_every_ms;

  // If false or not set, execute sub-steps serially.
  // If true, execute all substeps concurrently, each one in a separte thread.
  bool concurrent_substeps;

  // Name of a scalar boolean tensor.
  // ES checks this blob AFTER every substeps/subnets.
  // If specified, and the value is true, then ES will skip the rest and return
  // immediately.
  // This means that the report_net and the first step will always be called.
  // Use cases:
  // 1) the first substep stops the rest if data condition not met
  // 2) the first substep decide which of the rest of the steps should be run.
  // 3) external control
  //
  // ** It is the user's responsibility to not to put this blob in race conditions.
  // ** For example when setting this blob in concurrent substeps
  string should_stop_blob;

  // if only_once is true, this step will only be executed once. this ONLY takes
  // effect when using should_stop_blob
  bool only_once;

  // Whether to create a child workspace for this step.
  // If yes, the workflow and nets are re-created every time this step is run.
  bool create_workspace;

  // How many copies of the children execution steps to run concurrently.
  int32 num_concurrent_instances;
};

struct PlanDef {
  // All the networks that are used in this execution. Note that networks should
  // be ordered in the way they are executed, i.e. for a layer in a network, all
  // its input blobs should already have been initialized by the layers or
  // networks defined before it.
  string name;
  // The networks that are going to be used in this plan.
  std::vector<NetDef> network;
  std::vector<ExecutionStep> execution_step;
};

// Protobuf format for blobs that are not Tensors. We use a key to store the
// type of the blob. For example for a serialized DBProto, the type should
// be "DBReader" and the content should be a serialized DBProto object.
struct BlobProto {
  string name;
  string type;
  TensorProto tensor;
  bytes content;
  QTensorProto qtensor;
  // If blob is not Tensor and is divided into chunks, content_num_chunks
  // contains number of chunks, into which blob was divided.
  int32 content_num_chunks;
  int32 content_chunk_id;
};

// Protobuf format to serialize DBReader.
struct DBReaderProto {
  // The name for the DB object in the workspace.
  string name;
  // The source of the DB
  string source;
  // The type of the DB
  string db_type;
  // The current key of the DB if the DB supports seeking.
  string key;
};

}  // namespace mycaffe2
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_PROTO_CAFFE2_H_