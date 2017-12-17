
#ifndef BUBBLEFS_UTILS_TENSORFLOW_FRAMEWORK_PROTO_H_
#define BUBBLEFS_UTILS_TENSORFLOW_FRAMEWORK_PROTO_H_

// syntax = "proto3";
// package tensorflow;

#include <map>
#include <vector>
#include "platform/types.h"

namespace bubblefs {
namespace mytensorflow {
  
typedef std::string bytes; 
  
/// tensorflow/tensorflow/core/framework/types.proto  
  
//option cc_enable_arenas = true;
//option java_outer_classname = "VersionsProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Version information for a piece of serialized data
//
// There are different types of versions for each type of data
// (GraphDef, etc.), but they all have the same common shape
// described here.
//
// Each consumer has "consumer" and "min_producer" versions (specified
// elsewhere).  A consumer is allowed to consume this data if
//
//   producer >= min_producer
//   consumer >= min_consumer
//   consumer not in bad_consumers
//
struct VersionDef {
  // The version of the code that produced this data.
  int32 producer;

  // Any consumer below this version is not allowed to consume this data.
  int32 min_consumer;

  // Specific consumer versions which are disallowed (e.g. due to bugs).
  std::vector<int32> bad_consumers;
};

/// tensorflow/tensorflow/core/framework/allocation_description.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "AllocationDescriptionProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct AllocationDescription {
  // Total number of bytes requested
  int64 requested_bytes;

  // Total number of bytes allocated if known
  int64 allocated_bytes;

  // Name of the allocator used
  string allocator_name;

  // Identifier of the allocated buffer if known
  int64 allocation_id;

  // Set if this tensor only has one remaining reference
  bool has_single_reference;

  // Address of the allocation.
  uint64 ptr;
};

/// tensorflow/tensorflow/core/framework/device_attributes.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "DeviceAttributesProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct DeviceLocality {
  // Optional bus locality of device.  Default value of 0 means
  // no specific locality.  Specific localities are indexed from 1.
  int32 bus_id;
};

struct DeviceAttributes {
  // Fully specified name of the device within a cluster.
  string name;

  // String representation of device_type.
  string device_type;

  // Memory capacity of device in bytes.
  int64 memory_limit;

  // Platform-specific data about device that may be useful
  // for supporting efficient data transfers.
  DeviceLocality locality;

  // A device is assigned a global unique number each time it is
  // initialized. "incarnation" should never be 0.
  int64 incarnation;

  // String representation of the physical device that this device maps to.
  string physical_device_desc;
};

/// tensorflow/tensorflow/core/framework/types.proto
  
//option cc_enable_arenas = true;
//option java_outer_classname = "TypesProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// LINT.IfChange
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,  // Arbitrary C++ data types
  DT_UINT32 = 22,
  DT_UINT64 = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101,
  DT_DOUBLE_REF = 102,
  DT_INT32_REF = 103,
  DT_UINT8_REF = 104,
  DT_INT16_REF = 105,
  DT_INT8_REF = 106,
  DT_STRING_REF = 107,
  DT_COMPLEX64_REF = 108,
  DT_INT64_REF = 109,
  DT_BOOL_REF = 110,
  DT_QINT8_REF = 111,
  DT_QUINT8_REF = 112,
  DT_QINT32_REF = 113,
  DT_BFLOAT16_REF = 114,
  DT_QINT16_REF = 115,
  DT_QUINT16_REF = 116,
  DT_UINT16_REF = 117,
  DT_COMPLEX128_REF = 118,
  DT_HALF_REF = 119,
  DT_RESOURCE_REF = 120,
  DT_VARIANT_REF = 121,
  DT_UINT32_REF = 122,
  DT_UINT64_REF = 123
};  

/// tensorflow/tensorflow/core/framework/resource_handle.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ResourceHandle";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
struct ResourceHandleProto {
  // Unique name for the device containing the resource.
  string device;

  // Container in which this resource is placed.
  string container;

  // Unique name of this resource.
  string name;

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code;

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  string maybe_type_name;
};

/// tensorflow/tensorflow/core/framework/variable.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "VariableProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Protocol buffer representing a Variable.
struct SaveSliceInfoDef {
  // Name of the full variable of which this is a slice.
  string full_name;
  // Shape of the full variable.
  std::vector<int64> full_shape;
  // Offset of this variable into the full variable.
  std::vector<int64> var_offset;
  // Shape of this variable.
  std::vector<int64> var_shape;
};

struct VariableDef {
  // Name of the variable tensor.
  string variable_name;

  // Name of the tensor holding the variable's initial value.
  string initial_value_name;

  // Name of the initializer op.
  string initializer_name;

  // Name of the snapshot tensor.
  string snapshot_name;

  // Support for saving variables as slices of a larger variable.
  SaveSliceInfoDef save_slice_info_def;

  // Whether to represent this as a ResourceVariable.
  bool is_resource = 5;
};

/// tensorflow/tensorflow/core/framework/iterator.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "IteratorProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.util";

// Protocol buffer representing the metadata for an iterator's state stored
// as a Variant tensor.
struct IteratorStateMetadata {
  // A user-specified version string.
  string version;

  // Keys for tensors in the VariantTensorDataProto.
  std::vector<string> keys;
};

/// tensorflow/tensorflow/core/framework/tensor_shape.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "TensorShapeProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Dimensions of a tensor.
struct TensorShapeProto {
  // One dimension of the tensor.
  struct Dim {
    // Size of the tensor in that dimension.
    // This value must be >= -1, but values of -1 are reserved for "unknown"
    // shapes (values of -1 mean "unknown" dimension).  Certain wrappers
    // that work with TensorShapeProto may fail at runtime when deserializing
    // a TensorShapeProto containing a dim value of -1.
    int64 size;

    // Optional name of the tensor dimension.
    string name;
  };

  // Dimensions of the tensor, such as {"input", 30}, {"output", 40}
  // for a 30 x 40 2D tensor.  If an entry has size -1, this
  // corresponds to a dimension of unknown size. The names are
  // optional.
  //
  // The order of entries in "dim" matters: It indicates the layout of the
  // values in the tensor in-memory representation.
  //
  // The first entry in "dim" is the outermost dimension used to layout the
  // values, the last entry is the innermost dimension.  This matches the
  // in-memory layout of RowMajor Eigen tensors.
  //
  // If "dim.size()" > 0, "unknown_rank" must be false.
  std::vector<Dim> dim;

  // If true, the number of dimensions in the shape is unknown.
  //
  // If true, "dim.size()" must be 0.
  bool unknown_rank;
};

/// tensorflow/tensorflow/core/framework/tensor.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "TensorProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Protocol buffer representing a tensor.

struct TensorProto;

// Protocol buffer representing the serialization format of DT_VARIANT tensors.
struct VariantTensorDataProto {
  // Name of the type of objects being serialized.
  string type_name;
  // Portions of the object that are not Tensors.
  bytes metadata;
  // Tensors contained within objects being serialized.
  std::vector<TensorProto*> tensors;
};

struct TensorProto {
  DataType dtype;

  // Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
  TensorShapeProto tensor_shape;

  // Only one of the representations below is set, one of "tensor_contents" and
  // the "xxx_val" attributes.  We are not using oneof because as oneofs cannot
  // contain repeated fields it would require another extra set of messages.

  // Version number.
  //
  // In version 0, if the "repeated xxx" representations contain only one
  // element, that element is repeated to fill the shape.  This makes it easy
  // to represent a constant Tensor with a single value.
  int32 version_number;

  // Serialized raw tensor content from either Tensor::AsProtoTensorContent or
  // memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
  // can be used for all tensor types. The purpose of this representation is to
  // reduce serialization overhead during RPC call by avoiding serialization of
  // many repeated small items.
  bytes tensor_content;

  // Type specific representations that make it easy to create tensor protos in
  // all languages.  Only the representation corresponding to "dtype" can
  // be set.  The values hold the flattened representation of the tensor in
  // row major order.

  // DT_HALF, DT_BFLOAT16. Note that since protobuf has no int16 type, we'll
  // have some pointless zero padding for each value here.
  std::vector<int32> half_val;

  // DT_FLOAT.
  std::vector<float> float_val;

  // DT_DOUBLE.
  std::vector<double> double_val;

  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  std::vector<int32> int_val;

  // DT_STRING
  std::vector<bytes> string_val;

  // DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
  // and imaginary parts of i-th single precision complex.
  std::vector<float> scomplex_val;

  // DT_INT64
  std::vector<int64> int64_val;

  // DT_BOOL
  std::vector<bool> bool_val;

  // DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
  // and imaginary parts of i-th double precision complex.
  std::vector<double> dcomplex_val;

  // DT_RESOURCE
  std::vector<ResourceHandleProto> resource_handle_val;

  // DT_VARIANT
  std::vector<VariantTensorDataProto> variant_val;

  // DT_UINT32
  std::vector<uint32> uint32_val;

  // DT_UINT64
  std::vector<uint64> uint64_val;
};

/// tensorflow/tensorflow/core/framework/attr_value.proto

//option cc_enable_arenas = true;
///option java_outer_classname = "AttrValueProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Protocol buffer representing the value for an attr used to configure an Op.
// Comment indicates the corresponding attr type.  Only the field matching the
// attr type may be filled.

struct AttrValue;

// A list of attr names and their values. The whole list is attached
// with a string name.  E.g., MatMul[T=float].
struct NameAttrList {
  string name;
  std::map<string, AttrValue*> attr;
};

struct AttrValue {
  // LINT.IfChange
  struct ListValue {
    std::vector<bytes> s;        // "list(string)"
    std::vector<int64> i;        // "list(int)"
    std::vector<float> f;        // "list(float)"
    std::vector<bool> b;         // "list(bool)"
    std::vector<DataType> type;  // "list(type)"
    std::vector<TensorShapeProto> shape;         // "list(shape)"
    std::vector<TensorProto> tensor;             // "list(tensor)"
    std::vector<NameAttrList> func;              // "list(attr)"
  };
  // LINT.ThenChange(https://www.tensorflow.org/code/tensorflow/c/c_api.cc)

  union value {
    bytes s;                 // "string"
    int64 i;                 // "int"
    float f;                 // "float"
    bool b;                  // "bool"
    DataType type;           // "type"
    TensorShapeProto shape;  // "shape"
    TensorProto tensor;      // "tensor"
    ListValue list;          // any "list(...)"

    // "func" represents a function. func.name is a function's name or
    // a primitive op's name. func.attr.first is the name of an attr
    // defined for that function. func.attr.second is the value for
    // that attr in the instantiation.
    NameAttrList func;

    // This is a placeholder only used in nodes defined inside a
    // function.  It indicates the attr value will be supplied when
    // the function is instantiated.  For example, let us suppose a
    // node "N" in function "FN". "N" has an attr "A" with value
    // placeholder = "foo". When FN is instantiated with attr "foo"
    // set to "bar", the instantiated node N's attr A will have been
    // given the value "bar".
    string placeholder;
  };
};

/// tensorflow/tensorflow/core/framework/tensor_slice.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "TensorSliceProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Can only be interpreted if you know the corresponding TensorShape.
struct TensorSliceProto {
  // Extent of the slice in one dimension.
  struct Extent {
    // Either both or no attributes must be set.  When no attribute is set
    // means: All data in that dimension.

    // Start index of the slice, starting at 0.
    int64 start;

    // Length of the slice: if the length is missing or -1 we will
    // interpret this as "everything in this dimension".  We use
    // "oneof" to preserve information about whether the length is
    // present without changing the serialization format from the
    // prior proto2 version of this proto.
    union has_length {
      int64 length;
    };
  };

  // Extent of the slice in all tensor dimensions.
  //
  // Must have one entry for each of the dimension of the tensor that this
  // slice belongs to.  The order of sizes is the same as the order of
  // dimensions in the TensorShape.
  std::vector<Extent> extent;
};

/// tensorflow/tensorflow/core/framework/tensor_description.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "TensorDescriptionProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct TensorDescription {
  // Data type of tensor elements
  DataType dtype;

  // Shape of the tensor.
  TensorShapeProto shape;

  // Information about the size and allocator used for the data
  AllocationDescription allocation_description;
};

/// tensorflow/tensorflow/core/framework/node_def.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "NodeProto";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct NodeDef {
  // The name given to this operator. Used for naming inputs,
  // logging, visualization, etc.  Unique within a single GraphDef.
  // Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_./]*".
  string name;

  // The operation name.  There may be custom parameters in attrs.
  // Op names starting with an underscore are reserved for internal use.
  string op;

  // Each input is "node:src_output" with "node" being a string name and
  // "src_output" indicating which output tensor to use from "node". If
  // "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
  // may optionally be followed by control inputs that have the format
  // "^node".
  std::vector<string> input;

  // A (possibly partial) specification for the device on which this
  // node should be placed.
  // The expected syntax for this string is as follows:
  //
  // DEVICE_SPEC ::= PARTIAL_SPEC
  //
  // PARTIAL_SPEC ::= ("/" CONSTRAINT) *
  // CONSTRAINT ::= ("job:" JOB_NAME)
  //              | ("replica:" [1-9][0-9]*)
  //              | ("task:" [1-9][0-9]*)
  //              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
  //
  // Valid values for this string include:
  // * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
  // * "/job:worker/device:GPU:3"                   (partial specification)
  // * ""                                    (no specification)
  //
  // If the constraints do not resolve to a single device (or if this
  // field is empty or not present), the runtime will attempt to
  // choose a device automatically.
  string device;

  // Operation-specific graph-construction-time configuration.
  // Note that this should include all attrs defined in the
  // corresponding OpDef, including those with a value matching
  // the default -- this allows the default to change and makes
  // NodeDefs easier to interpret on their own.  However, if
  // an attr with a default is not specified in this list, the
  // default will be used.
  // The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
  // one of the names from the corresponding OpDef's attr field).
  // The values must have a type matching the corresponding OpDef
  // attr's type field.
  // TODO(josh11b): Add some examples here showing best practices.
  std::map<string, AttrValue> attr;
};

/// tensorflow/tensorflow/core/framework/op_def.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "OpDefProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// LINT.ThenChange(
//     https://www.tensorflow.org/code/tensorflow/core/framework/op_def_util.cc)

// Information about version-dependent deprecation of an op
struct OpDeprecation {
  // First GraphDef version at which the op is disallowed.
  int32 version;

  // Explanation of why it was deprecated and what to use instead.
  string explanation;
};

// Defines an operation. A NodeDef in a GraphDef specifies an Op by
// using the "op" field which should match the name of a OpDef.
// LINT.IfChange

struct OpDef {
  // Op names starting with an underscore are reserved for internal use.
  // Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9_]*".
  string name;

  // For describing inputs and outputs.
  struct ArgDef {
    // Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
    string name;

    // Human readable description.
    string description;

    // Describes the type of one or more tensors that are accepted/produced
    // by this input/output arg.  The only legal combinations are:
    // * For a single tensor: either the "type" field is set or the
    //   "type_attr" field is set to the name of an attr with type "type".
    // * For a sequence of tensors with the same type: the "number_attr"
    //   field will be set to the name of an attr with type "int", and
    //   either the "type" or "type_attr" field will be set as for
    //   single tensors.
    // * For a sequence of tensors, the "type_list_attr" field will be set
    //   to the name of an attr with type "list(type)".
    DataType type;
    string type_attr;    // if specified, attr must have type "type"
    string number_attr;  // if specified, attr must have type "int"
    // If specified, attr must have type "list(type)", and none of
    // type, type_attr, and number_attr may be specified.
    string type_list_attr;

    // For inputs: if true, the inputs are required to be refs.
    //   By default, inputs can be either refs or non-refs.
    // For outputs: if true, outputs are refs, otherwise they are not.
    bool is_ref = 16;
  };

  // Description of the input(s).
  std::vector<ArgDef> input_arg;

  // Description of the output(s).
  std::vector<ArgDef> output_arg = 3;

  // Description of the graph-construction-time configuration of this
  // Op.  That is to say, this describes the attr fields that will
  // be specified in the NodeDef.
  struct AttrDef {
    // A descriptive name for the argument.  May be used, e.g. by the
    // Python client, as a keyword argument name, and so should match
    // the regexp "[a-z][a-z0-9_]+".
    string name = 1;

    // One of the type names from attr_value.proto ("string", "list(string)",
    // "int", etc.).
    string type = 2;

    // A reasonable default for this attribute if the user does not supply
    // a value.  If not specified, the user must supply a value.
    AttrValue default_value = 3;

    // Human-readable description.
    string description = 4;

    // TODO(josh11b): bool is_optional?

    // --- Constraints ---
    // These constraints are only in effect if specified.  Default is no
    // constraints.

    // For type == "int", this is a minimum value.  For "list(___)"
    // types, this is the minimum length.
    bool has_minimum = 5;
    int64 minimum = 6;

    // The set of allowed values.  Has type that is the "list" version
    // of the "type" field above (uses the "list" field of AttrValue).
    // If type == "type" or "list(type)" above, then the "type" field
    // of "allowed_values.list" has the set of allowed DataTypes.
    // If type == "string" or "list(string)", then the "s" field of
    // "allowed_values.list" has the set of allowed strings.
    AttrValue allowed_values;
  };
  std::vector<AttrDef> attr;

  // Optional deprecation based on GraphDef versions.
  OpDeprecation deprecation;

  // One-line human-readable description of what the Op does.
  string summary;

  // Additional, longer human-readable description of what the Op does.
  string description;

  // -------------------------------------------------------------------------
  // Which optimizations this operation can participate in.

  // True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
  bool is_commutative;

  // If is_aggregate is true, then this operation accepts N >= 2
  // inputs and produces 1 output all of the same type.  Should be
  // associative and commutative, and produce output with the same
  // shape as the input.  The optimizer may replace an aggregate op
  // taking input from multiple devices with a tree of aggregate ops
  // that aggregate locally within each device (and possibly within
  // groups of nearby devices) before communicating.
  // TODO(josh11b): Implement that optimization.
  bool is_aggregate;  // for things like add

  // Other optimizations go here, like
  //   can_alias_input, rewrite_when_output_unused, partitioning_strategy, etc.

  // -------------------------------------------------------------------------
  // Optimization constraints.

  // By default Ops may be moved between devices.  Stateful ops should
  // either not be moved, or should only be moved if that state can also
  // be moved (e.g. via some sort of save / restore).
  // Stateful ops are guaranteed to never be optimized away by Common
  // Subexpression Elimination (CSE).
  bool is_stateful;  // for things like variables, queue

  // -------------------------------------------------------------------------
  // Non-standard options.

  // By default, all inputs to an Op must be initialized Tensors.  Ops
  // that may initialize tensors for the first time should set this
  // field to true, to allow the Op to take an uninitialized Tensor as
  // input.
  bool allows_uninitialized_input;  // for Assign, etc.
};

// A collection of OpDefs
struct OpList {
  std::vector<OpDef> op;
};

/// tensorflow/tensorflow/core/framework/op_gen_overrides.proto

// Used to override the default API & behavior in the generated code
// for client languages, from what you would get from the OpDef alone.
// This is so we can evolve the API while remaining backwards
// compatible when interpretting old graphs.  Overrides go in an
// "op_gen_overrides.pbtxt" file with a text-format OpGenOverrides
// message.  Right now these only apply to the C++ API.
// TODO(josh11b): In the future there will be a common set of overrides
// and per-client-language overrides.
//
// WARNING: Be *very* careful using these features -- these overrides
// can change the semantics of existing code.  These changes may need
// to wait until a major release of TensorFlow to avoid breaking our
// compatibility promises.
struct OpGenOverride {
  // Name of the op to apply overrides to.
  string name;

  // Do not include this op in the generated API.
  // If `skip` is true, all other overrides are ignored for this op.
  bool skip;

  // Hide this op by putting it into an internal namespace (or whatever
  // is appropriate in the target language).
  bool hide;

  // Use a different name in the API than the op's name. Note that
  // the op's name in `backticks` will also be replaced in the docs.
  string rename_to;

  // Create *additional* API endpoints with different names (contrast
  // with rename_to, which affects the original name).
  std::vector<string> alias;

  // Map the name of an attr to a new default value to use.  This
  // default will be used when creating new graphs, as opposed to the
  // default in the OpDef, which will be used when interpreting old
  // GraphDefs.  If this attr is also renamed (using attr_rename
  // below), use the original name of the attr.
  struct AttrDefault {
    string name;
    AttrValue value;
  };
  std::vector<AttrDefault> attr_default;

  // Change the name used to access attrs/inputs/outputs in the API
  // from what is used in the GraphDef.  Note that these names in
  // `backticks` will also be replaced in the docs.
  struct Rename {
    string from;
    string to;
  };
  std::vector<Rename> attr_rename;
  std::vector<Rename> input_rename;
  std::vector<Rename> output_rename;
};

struct OpGenOverrides {
  std::vector<OpGenOverride> op;
};

/// tensorflow/tensorflow/core/framework/function.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "FunctionProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// A function can be instantiated when the runtime can bind every attr
// with a value. When a GraphDef has a call to a function, it must
// have binding for every attr defined in the signature.
//
// TODO(zhifengc):
//   * device spec, etc.
struct FunctionDef {
  // The definition of the function's name, arguments, return values,
  // attrs etc.
  OpDef signature;

  // Attributes specific to this function definition.
  std::map<string, AttrValue> attr;

  // NOTE: field id 2 deleted on Jan 11, 2016, GraphDef version 21.

  // In both of the following fields, there is the need to specify an
  // output that is used as either the input to another node (in
  // `node_def`) or as a return value of the function (in `ret`).
  // Unlike the NodeDefs in GraphDef, we need to be able to specify a
  // list in some cases (instead of just single outputs).  Also, we
  // need to be able to deal with lists of unknown length (so the
  // output index may not be known at function definition time).  So
  // we use the following format instead:
  // * "fun_in" where "fun_in" is the name of a function input arg in
  //   the `signature` field above.  This represents that input, whether
  //   it is a single tensor or a list.
  // * "fun_in:0" gives the first element of a function input arg (a
  //   non-list input is considered a list of length 1 for these
  //   purposes).
  // * "node:out" where "node" is the name of a node in `node_def` and
  //   "out" is the name one of its op's output arguments (the name
  //   comes from the OpDef of the node's op). This represents that
  //   node's output, whether it is a single tensor or a list.
  //   Note: We enforce that an op's output arguments are never
  //   renamed in the backwards-compatibility test.
  // * "node:out:0" gives the first element of a node output arg (a
  //   non-list output is considered a list of length 1 for these
  //   purposes).
  //
  // NOT CURRENTLY SUPPORTED (but may be in the future):
  // * "node:out:-1" gives last element in a node output list
  // * "node:out:1:" gives a list with all but the first element in a
  //   node output list
  // * "node:out::-1" gives a list with all but the last element in a
  //   node output list

  // The body of the function.  Unlike the NodeDefs in a GraphDef, attrs
  // may have values of type `placeholder` and the `input` field uses
  // the "output" format above.

  // By convention, "op" in node_def is resolved by consulting with a
  // user-defined library first. If not resolved, "func" is assumed to
  // be a builtin op.
  std::vector<NodeDef> node_def;

  // A mapping from the output arg names from `signature` to the
  // outputs from `node_def` that should be returned by the function.
  std::map<string, string> ret;
};

// GradientDef defines the gradient function of a function defined in
// a function library.
//
// A gradient function g (specified by gradient_func) for a function f
// (specified by function_name) must follow the following:
//
// The function 'f' must be a numerical function which takes N inputs
// and produces M outputs. Its gradient function 'g', which is a
// function taking N + M inputs and produces N outputs.
//
// I.e. if we have
//    (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
// then, g is
//    (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
//                                      dL/dy1, dL/dy2, ..., dL/dy_M),
// where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
// loss function). dL/dx_i is the partial derivative of L with respect
// to x_i.
struct GradientDef {
  string function_name;  // The function name.
  string gradient_func;  // The gradient function's name.
};

// A library is a set of named functions.
struct FunctionDefLibrary {
  std::vector<FunctionDef> function;
  std::vector<GradientDef> gradient;
};

/// tensorflow/tensorflow/core/framework/graph.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "GraphProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Represents the graph of operations
struct GraphDef {
  std::vector<NodeDef> node;

  // Compatibility versions of the graph.  See core/public/version.h for version
  // history.  The GraphDef version is distinct from the TensorFlow version, and
  // each release of TensorFlow will support a range of GraphDef versions.
  VersionDef versions;

  // Deprecated single version field; use versions above instead.  Since all
  // GraphDef changes before "versions" was introduced were forward
  // compatible, this field is entirely ignored.
  int32 version; //[deprecated = true];

  // EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
  //
  // "library" provides user-defined functions.
  //
  // Naming:
  //   * library.function.name are in a flat namespace.
  //     NOTE: We may need to change it to be hierarchical to support
  //     different orgs. E.g.,
  //     { "/google/nn", { ... }},
  //     { "/google/vision", { ... }}
  //     { "/org_foo/module_bar", { ... }}
  //     map<string, FunctionDefLib> named_lib;
  //   * If node[i].op is the name of one function in "library",
  //     node[i] is deemed as a function call. Otherwise, node[i].op
  //     must be a primitive operation supported by the runtime.
  //
  //
  // Function call semantics:
  //
  //   * The callee may start execution as soon as some of its inputs
  //     are ready. The caller may want to use Tuple() mechanism to
  //     ensure all inputs are ready in the same time.
  //
  //   * The consumer of return values may start executing as soon as
  //     the return values the consumer depends on are ready.  The
  //     consumer may want to use Tuple() mechanism to ensure the
  //     consumer does not start until all return values of the callee
  //     function are ready.
  FunctionDefLibrary library;
};

/// tensorflow/tensorflow/core/framework/graph_transfer_info.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "GraphTransferInfoProto";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
struct GraphTransferInfo {
  enum Destination {
    NOP = 0,
    HEXAGON = 1
  };
  struct NodeInput {
    int32 node_id;
    int32 output_port;
  };
  struct NodeInfo {
    string name1;
    int32 node_id;
    string type_name;
    int32 soc_op_id;
    int32 padding_id;
    int32 input_count;
    int32 output_count;
  };
  struct ConstNodeInfo {
    string name;
    int32 node_id;
    std::vector<int64> shape;
    bytes data;
    DataType dtype;
  };
  struct NodeInputInfo {
    int32 node_id;
    std::vector<NodeInput> node_input;
  };
  struct NodeOutputInfo {
    int32 node_id;
    std::vector<int32> max_byte_size;
  };
  struct GraphInputNodeInfo {
    string name;
    std::vector<int64> shape;
    DataType dtype;
  };

  struct GraphOutputNodeInfo {
    string name;
    std::vector<int64> shape;
    DataType dtype;
  };

  std::vector<NodeInfo> node_info;
  std::vector<ConstNodeInfo> const_node_info;
  std::vector<NodeInputInfo> node_input_info;
  std::vector<NodeOutputInfo> node_output_info;
  // Input Node parameters of transferred graph
  std::vector<GraphInputNodeInfo> graph_input_node_info;
  std::vector<GraphOutputNodeInfo> graph_output_node_info;
  // Destination of graph transfer
  Destination destination;
};

/// tensorflow/tensorflow/core/framework/cost_graph.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "CostGraphProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct CostGraphDef {
  struct Node {
    // The name of the node. Names are globally unique.
    string name;

    // The device of the node. Can be empty if the node is mapped to the
    // default partition or partitioning hasn't been run yet.
    string device;

    // The id of the node. Node ids are only unique inside a partition.
    int32 id;

    // Inputs of this node. They must be executed before this node can be
    // executed. An input is a particular output of another node, specified
    // by the node id and the output index.
    struct InputInfo {
      int32 preceding_node;
      int32 preceding_port;
    };
    std::vector<InputInfo> input_info;

    // Outputs of this node.
    struct OutputInfo {
      int64 size;
      // If >= 0, the output is an alias of an input. Note that an alias input
      // may itself be an alias. The algorithm will therefore need to follow
      // those pointers.
      int64 alias_input_port;
      TensorShapeProto shape;
      DataType dtype;
    };
    std::vector<OutputInfo> output_info;

    // Temporary memory used by this node.
    int64 temporary_memory_size;

    int64 host_temp_memory_size;
    int64 device_temp_memory_size;
    int64 host_persistent_memory_size;
    int64 device_persistent_memory_size;

    // Estimate of the computational cost of this node, in microseconds.
    int64 compute_cost;

    // Analytical estimate of the computational cost of this node, in
    // microseconds.
    int64 compute_time;

    // Analytical estimate of the memory access cost of this node, in
    // microseconds.
    int64 memory_time;

    // If true, the output is permanent: it can't be discarded, because this
    // node is part of the "final output". Nodes may depend on final nodes.
    bool is_final;

    // Ids of the control inputs for this node.
    std::vector<int32> control_input;
  };
  std::vector<Node> node;
};

/// tensorflow/tensorflow/core/framework/kernel_def.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "KernelDefProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct KernelDef {
  // Must match the name of an Op.
  string op;

  // Type of device this kernel runs on.
  string device_type;

  struct AttrConstraint {
    // Name of an attr from the Op.
    string name;

    // A list of values that this kernel supports for this attr.
    // Like OpDef.AttrDef.allowed_values, except for kernels instead of Ops.
    AttrValue allowed_values;
  };
  std::vector<AttrConstraint> constraint;

  // Names of the Op's input_/output_args that reside in host memory
  // instead of device memory.
  std::vector<string> host_memory_arg;

  // This allows experimental kernels to be registered for an op that
  // won't be used unless the user specifies a "_kernel" attr with
  // value matching this.
  string label;
};

/// tensorflow/tensorflow/core/framework/reader_base.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ReaderBaseProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// For serializing and restoring the state of ReaderBase, see
// reader_base.h for details.
struct ReaderBaseState {
  int64 work_started;
  int64 work_finished;
  int64 num_records_produced;
  bytes current_work;
};

/// tensorflow/tensorflow/core/framework/step_stats.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "StepStatsProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// An allocation/de-allocation operation performed by the allocator.
struct AllocationRecord {
  // The timestamp of the operation.
  int64 alloc_micros;
  // Number of bytes allocated, or de-allocated if negative.
  int64 alloc_bytes;
};

struct AllocatorMemoryUsed {
  string allocator_name;
  // These are per-node allocator memory stats.
  int64 total_bytes;
  int64 peak_bytes;
  // The bytes that are not deallocated.
  int64 live_bytes;
  // The allocation and deallocation timeline.
  std::vector<AllocationRecord> allocation_records;

  // These are snapshots of the overall allocator memory stats.
  // The number of live bytes currently allocated by the allocator.
  int64 allocator_bytes_in_use;
};

// Output sizes recorded for a single execution of a graph node.
struct NodeOutput {
  int32 slot;
  TensorDescription tensor_description;
};

// For memory tracking.
struct MemoryStats {
  int64 host_temp_memory_size;
  int64 device_temp_memory_size;
  int64 host_persistent_memory_size;
  int64 device_persistent_memory_size;
  std::vector<int64> host_persistent_tensor_alloc_ids;
  std::vector<int64> device_persistent_tensor_alloc_ids;
};

// Time/size stats recorded for a single execution of a graph node.
struct NodeExecStats {
  // TODO(tucker): Use some more compact form of node identity than
  // the full string name.  Either all processes should agree on a
  // global id (cost_id?) for each node, or we should use a hash of
  // the name.
  string node_name;
  int64 all_start_micros;
  int64 op_start_rel_micros;
  int64 op_end_rel_micros;
  int64 all_end_rel_micros;
  std::vector<AllocatorMemoryUsed> memory;
  std::vector<NodeOutput> output;
  string timeline_label;
  int64 scheduled_micros;
  uint32 thread_id;
  std::vector<AllocationDescription> referenced_tensor;
  MemoryStats memory_stats;
};

struct DeviceStepStats {
  string device;
  std::vector<NodeExecStats> node_stats;
};

struct StepStats {
  std::vector<DeviceStepStats> dev_stats;
};

/// tensorflow/tensorflow/core/framework/summary.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "SummaryProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Metadata associated with a series of Summary data
struct SummaryDescription {
  // Hint on how plugins should process the data in this series.
  // Supported values include "scalar", "histogram", "image", "audio"
  string type_hint;
};

// Serialization format for histogram module in
// core/lib/histogram/histogram.h
struct HistogramProto {
  double min;
  double max;
  double num;
  double sum;
  double sum_squares;

  // Parallel arrays encoding the bucket boundaries and the bucket values.
  // bucket(i) is the count for the bucket i.  The range for
  // a bucket is:
  //   i == 0:  -DBL_MAX .. bucket_limit(0)
  //   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
  std::vector<double> bucket_limit;
  std::vector<double> bucket;
};

// A SummaryMetadata encapsulates information on which plugins are able to make
// use of a certain summary value.
struct SummaryMetadata {
  struct PluginData {
    // The name of the plugin this data pertains to.
    string plugin_name;

    // The content to store for the plugin. The best practice is for this to be
    // a binary serialized protocol buffer.
    bytes content;
  };

  // Data that associates a summary with a certain plugin.
  PluginData plugin_data;

  // Display name for viewing in TensorBoard.
  string display_name;

  // Longform readable description of the summary sequence. Markdown supported.
  string summary_description;
};

// A Summary is a set of named values to be displayed by the
// visualizer.
//
// Summaries are produced regularly during training, as controlled by
// the "summary_interval_secs" attribute of the training operation.
// Summaries are also produced at the end of an evaluation.
struct Summary {
  struct Image {
    // Dimensions of the image.
    int32 height;
    int32 width;
    // Valid colorspace values are
    //   1 - grayscale
    //   2 - grayscale + alpha
    //   3 - RGB
    //   4 - RGBA
    //   5 - DIGITAL_YUV
    //   6 - BGRA
    int32 colorspace;
    // Image data in encoded format.  All image formats supported by
    // image_codec::CoderUtil can be stored here.
    bytes encoded_image_string;
  };

  struct Audio {
    // Sample rate of the audio in Hz.
    float sample_rate;
    // Number of channels of audio.
    int64 num_channels;
    // Length of the audio in frames (samples per channel).
    int64 length_frames;
    // Encoded audio data and its associated RFC 2045 content type (e.g.
    // "audio/wav").
    bytes encoded_audio_string;
    string content_type;
  };

  struct Value {
    // This field is deprecated and will not be set.
    string node_name;

    // Tag name for the data. Used by TensorBoard plugins to organize data. Tags
    // are often organized by scope (which contains slashes to convey
    // hierarchy). For example: foo/bar/0
    string tag;

    // Contains metadata on the summary value such as which plugins may use it.
    // Take note that many summary values may lack a metadata field. This is
    // because the FileWriter only keeps a metadata object on the first summary
    // value with a certain tag for each tag. TensorBoard then remembers which
    // tags are associated with which plugins. This saves space.
    SummaryMetadata metadata;

    // Value associated with the tag.
    union value {
      float simple_value;
      bytes obsolete_old_style_histogram;
      Image image;
      HistogramProto histo;
      Audio audio;
      TensorProto tensor;
    };
  };

  // Set of values for the summary.
  std::vector<Value> value;
};

/// tensorflow/tensorflow/core/framework/log_memory.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "LogMemoryProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct MemoryLogStep {
  // Process-unique step id.
  int64 step_id;

  // Handle describing the feeds and fetches of the step.
  string handle;
};

struct MemoryLogTensorAllocation {
  // Process-unique step id.
  int64 step_id;

  // Name of the kernel making the allocation as set in GraphDef,
  // e.g., "affine2/weights/Assign".
  string kernel_name;

  // Allocated tensor details.
  TensorDescription tensor;
};

struct MemoryLogTensorDeallocation {
  // Id of the tensor buffer being deallocated, used to match to a
  // corresponding allocation.
  int64 allocation_id;

  // Name of the allocator used.
  string allocator_name;
};

struct MemoryLogTensorOutput {
  // Process-unique step id.
  int64 step_id;

  // Name of the kernel producing an output as set in GraphDef, e.g.,
  // "affine2/weights/Assign".
  string kernel_name;

  // Index of the output being set.
  int32 index;

  // Output tensor details.
  TensorDescription tensor;
};

struct MemoryLogRawAllocation {
  // Process-unique step id.
  int64 step_id;

  // Name of the operation making the allocation.
  string operation;

  // Number of bytes in the allocation.
  int64 num_bytes;

  // Address of the allocation.
  uint64 ptr;

  // Id of the tensor buffer being allocated, used to match to a
  // corresponding deallocation.
  int64 allocation_id;

  // Name of the allocator used.
  string allocator_name;
};

struct MemoryLogRawDeallocation {
  // Process-unique step id.
  int64 step_id = 1;

  // Name of the operation making the deallocation.
  string operation = 2;

  // Id of the tensor buffer being deallocated, used to match to a
  // corresponding allocation.
  int64 allocation_id = 3;

  // Name of the allocator used.
  string allocator_name = 4;

  // True if the deallocation is queued and will be performed later,
  // e.g. for GPU lazy freeing of buffers.
  bool deferred = 5;
};

/// tensorflow/tensorflow/core/framework/api_def.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ApiDefProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Used to specify and override the default API & behavior in the
// generated code for client languages, from what you would get from
// the OpDef alone. There will be a set of ApiDefs that are common
// to all client languages, and another set per client language.
// The per-client-language ApiDefs will inherit values from the
// common ApiDefs which it can either replace or modify.
//
// We separate the API definition from the OpDef so we can evolve the
// API while remaining backwards compatible when interpretting old
// graphs.  Overrides go in an "api_def.pbtxt" file with a text-format
// ApiDefs message.
//
// WARNING: Be *very* careful changing the API for any existing op --
// you can change the semantics of existing code.  These changes may
// need to wait until a major release of TensorFlow to avoid breaking
// our compatibility promises.
struct ApiDef {
  // Name of the op (in the OpDef) to specify the API for.
  string graph_op_name;

  enum Visibility {
    // Normally this is "VISIBLE" unless you are inheriting a
    // different value from another ApiDef.
    DEFAULT_VISIBILITY = 0,
    // Publicly visible in the API.
    VISIBLE = 1,
    // Do not include this op in the generated API. If visibility is
    // set to 'SKIP', other fields are ignored for this op.
    SKIP = 2,
    // Hide this op by putting it into an internal namespace (or whatever
    // is appropriate in the target language).
    HIDDEN = 3
  };
  Visibility visibility;

  // If you specify any endpoint, this will replace all of the
  // inherited endpoints.  The first endpoint should be the
  // "canonical" endpoint, and should not be deprecated (unless all
  // endpoints are deprecated).
  struct Endpoint {
    // Name should be either like "CamelCaseName" or
    // "Package.CamelCaseName". Client-language-specific ApiDefs may
    // use a snake_case convention instead of CamelCase.
    string name;

    // First GraphDef version at which the op is disallowed.
    int32 deprecation_version;
  };
  std::vector<Endpoint> endpoint;

  struct Arg {
    string name;

    // Change the name used to access this arg in the API from what
    // is used in the GraphDef.  Note that these names in `backticks`
    // will also be replaced in the summary & description fields.
    string rename_to;

    // Note: this will replace any inherited arg doc. There is no
    // current way of modifying arg descriptions (other than replacing
    // them entirely) as can be done with op descriptions.
    string description;
  };
  std::vector<Arg> in_arg;
  std::vector<Arg> out_arg;
  // List of original in_arg names to specify new argument order.
  // Length of arg_order should be either empty to keep current order
  // or match size of in_arg.
  std::vector<string> arg_order;

  // Description of the graph-construction-time configuration of this
  // Op.  That is to say, this describes the attr fields that will
  // be specified in the NodeDef.
  struct Attr {
    string name;

    // Change the name used to access this attr in the API from what
    // is used in the GraphDef.  Note that these names in `backticks`
    // will also be replaced in the summary & description fields.
    string rename_to;

    // Specify a new default value to use for this attr.  This default
    // will be used when creating new graphs, as opposed to the
    // default in the OpDef, which will be used when interpreting old
    // GraphDefs.
    AttrValue default_value;

    // Note: this will replace any inherited attr doc, there is no current
    // way of modifying attr descriptions as can be done with op descriptions.
    string description;
  };
  std::vector<Attr> attr;

  // One-line human-readable description of what the Op does.
  string summary;

  // Additional, longer human-readable description of what the Op does.
  string description;

  // Modify an existing/inherited description by adding text to the beginning
  // or end.
  string description_prefix;
  string description_suffix;
};

struct ApiDefs {
  std::vector<ApiDef> op;
};

} // namespace mytensorflow
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TENSORFLOW_FRAMEWORK_PROTO_H_