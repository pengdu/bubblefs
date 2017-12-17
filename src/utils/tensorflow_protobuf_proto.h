/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef BUBBLEFS_UTILS_TENSORFLOW_PROTOBUF_PROTO_H_
#define BUBBLEFS_UTILS_TENSORFLOW_PROTOBUF_PROTO_H_

#include <map>
#include <string>
#include <vector>
#include "platform/types.h"
#include "utils/tensorflow_framework_proto.h"

namespace bubblefs {
namespace mytensorflow {
  
//syntax = "proto3";
//package tensorflow;

/// tensorflow/tensorflow/core/protobuf/cluster.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ClusterProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

// This file contains protos to be used when defining a TensorFlow
// cluster.
//
// EXAMPLES
// --------
//
// 1. A single-process cluster, containing "/job:local/task:0".
//
//    Cluster:
//      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
//
//    Server:
//      cluster { $CLUSTER } job_name: 'local' task_index: 0
//
// 2. A two-process cluster, containing "/job:local/task:{0,1}".
//
//    Cluster:
//      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
//                          tasks { key: 1 value: 'localhost:2223' } }
//
//    Servers:
//      cluster { $CLUSTER } job_name: 'local' task_index: 0
//      cluster { $CLUSTER } job_name: 'local' task_index: 1
//
// 3. A two-job cluster, containing "/job:worker/task:{0,1,2}" and
//    "/job:ps/task:{0,1}".
//
//    Cluster:
//      job { name: 'worker' tasks { key: 0 value: 'worker1:2222' }
//                           tasks { key: 1 value: 'worker2:2222' }
//                           tasks { key: 2 value: 'worker3:2222' } }
//      job { name: 'ps'     tasks { key: 0 value: 'ps0:2222' }
//                           tasks { key: 1 value: 'ps1:2222' } }
//
//    Servers:
//      cluster { $CLUSTER } job_name: 'worker' task_index: 0
//      cluster { $CLUSTER } job_name: 'worker' task_index: 1
//      cluster { $CLUSTER } job_name: 'worker' task_index: 2
//      cluster { $CLUSTER } job_name: 'ps'     task_index: 0
//      cluster { $CLUSTER } job_name: 'ps'     task_index: 1

// Defines a single job in a TensorFlow cluster.
struct JobDef {
  // The name of this job.
  string name;

  // Mapping from task ID to "hostname:port" string.
  //
  // If the `name` field contains "worker", and the `tasks` map contains a
  // mapping from 7 to "example.org:2222", then the device prefix
  // "/job:worker/task:7" will be assigned to "example.org:2222".
  std::map<int32, string> tasks;
};

// Defines a TensorFlow cluster as a set of jobs.
struct ClusterDef {
  // The jobs that comprise the cluster.
  std::vector<JobDef> job;
};

/// tensorflow/tensorflow/core/protobuf/control_flow.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ControlFlowProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// Control flow context related protocol buffers.

// Protocol buffer representing the values in ControlFlowContext.
struct ValuesDef {
  // Value names that have been seen in this context.
  std::vector<string> values;

  // Value names referenced by but external to this context.
  std::map<string, string> external_values;
};

// Protocol buffer representing a CondContext object.
struct CondContextDef {
  // Name of the context.
  string context_name;

  // Name of the pred tensor.
  string pred_name;

  // Name of the pivot tensor.
  string pivot_name;

  // Branch prediction. 0 or 1.
  int32 branch;

  // Values and external values in control flow context.
  ValuesDef values_def;
};

// Protocol buffer representing a WhileContext object.
struct WhileContextDef {
  // Name of the context.
  string context_name;

  // The number of iterations allowed to run in parallel.
  int32 parallel_iterations;

  // Whether backprop is enabled for this while loop.
  bool back_prop;

  // Whether GPU-CPU memory swap is enabled for this loop.
  bool swap_memory;

  // Name of the pivot tensor.
  string pivot_name;

  // Name of the pivot_for_pred tensor.
  string pivot_for_pred_name;

  // Name of the pivot_for_body tensor.
  string pivot_for_body_name;

  // List of names for exit tensors.
  std::vector<string> loop_exit_names;

  // List of names for enter tensors.
  std::vector<string> loop_enter_names;

  // Values and external values in control flow context.
  ValuesDef values_def;

  // Optional name of the maximum_iterations tensor.
  string maximum_iterations_name;

  // Next available id: 12.
};

/// tensorflow/tensorflow/core/protobuf/saver.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "SaverProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.util";

// Protocol buffer representing the configuration of a Saver.
struct SaverDef {
  // The name of the tensor in which to specify the filename when saving or
  // restoring a model checkpoint.
  string filename_tensor_name;

  // The operation to run when saving a model checkpoint.
  string save_tensor_name;

  // The operation to run when restoring a model checkpoint.
  string restore_op_name;

  // Maximum number of checkpoints to keep.  If 0, no checkpoints are deleted.
  int32 max_to_keep;

  // Shard the save files, one per device that has Variable nodes.
  bool sharded;

  // How often to keep an additional checkpoint. If not specified, only the last
  // "max_to_keep" checkpoints are kept; if specified, in addition to keeping
  // the last "max_to_keep" checkpoints, an additional checkpoint will be kept
  // for every n hours of training.
  float keep_checkpoint_every_n_hours;

  // A version number that identifies a different on-disk checkpoint format.
  // Usually, each subclass of BaseSaverBuilder works with a particular
  // version/format.  However, it is possible that the same builder may be
  // upgraded to support a newer checkpoint format in the future.
  enum CheckpointFormatVersion {
    // Internal legacy format.
    LEGACY = 0,
    // Deprecated format: tf.Saver() which works with tensorflow::table::Table.
    V1 = 1,
    // Current format: more efficient.
    V2 = 2
  };
  CheckpointFormatVersion version;
};

/// tensorflow/tensorflow/core/protobuf/device_properties.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "DevicePropertiesProtos";

struct DeviceProperties {
  // Device type (CPU, GPU, ...)
  string type;
  // Vendor (Intel, nvidia, ...)
  string vendor;
  // Model (Haswell, K40, ...)
  string model;
  // Core Frequency in Mhz
  int64 frequency;
  // Number of cores
  int64 num_cores;
  // Version of the tools and libraries used with this device (e.g. gcc 4.9,
  // cudnn 5.1)
  std::map<string, string> environment;
  // Number of registers per core.
  int64 num_registers;
  // L1 cache size in bytes
  int64 l1_cache_size;
  // L2 cache size in bytes
  int64 l2_cache_size;
  // L3 cache size in bytes
  int64 l3_cache_size;
  // Shared memory size per multiprocessor in bytes. This field is
  // applicable to GPUs only.
  int64 shared_memory_size_per_multiprocessor;
  // Memory size in bytes
  int64 memory_size;
  // Memory bandwidth in KB/s
  int64 bandwidth;
};

struct NamedDevice {
  string name;
  DeviceProperties properties;
};

/// tensorflow/tensorflow/core/protobuf/named_tensor.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "NamedTensorProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// A pair of tensor name and tensor values.
struct NamedTensorProto {
  // Name of the tensor.
  string name;

  // The client can populate a TensorProto using a tensorflow::Tensor`, or
  // directly using the protobuf field accessors.
  //
  // The client specifies whether the returned tensor values should be
  // filled tensor fields (float_val, int_val, etc.) or encoded in a
  // compact form in tensor.tensor_content.
  TensorProto tensor;
};

/// tensorflow/tensorflow/core/protobuf/debug.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "DebugProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

// EXPERIMENTAL. Option for watching a node.
struct DebugTensorWatch {
  // Name of the node to watch.
  string node_name;

  // Output slot to watch.
  // The semantics of output_slot == -1 is that the node is only watched for
  // completion, but not for any output tensors. See NodeCompletionCallback
  // in debug_gateway.h.
  // TODO(cais): Implement this semantics.
  int32 output_slot;

  // Name(s) of the debugging op(s).
  // One or more than one probes on a tensor.
  // e.g., {"DebugIdentity", "DebugNanCount"}
  std::vector<string> debug_ops;

  // URL(s) for debug targets(s).
  //
  // Supported URL formats are:
  //   - file:///foo/tfdbg_dump: Writes out Event content to file
  //     /foo/tfdbg_dump.  Assumes all directories can be created if they don't
  //     already exist.
  //   - grpc://localhost:11011: Sends an RPC request to an EventListener
  //     service running at localhost:11011 with the event.
  //   - memcbk:///event_key: Routes tensors to clients using the
  //     callback registered with the DebugCallbackRegistry for event_key.
  //
  // Each debug op listed in debug_ops will publish its output tensor (debug
  // signal) to all URLs in debug_urls.
  //
  // N.B. Session::Run() supports concurrent invocations of the same inputs
  // (feed keys), outputs and target nodes. If such concurrent invocations
  // are to be debugged, the callers of Session::Run() must use distinct
  // debug_urls to make sure that the streamed or dumped events do not overlap
  // among the invocations.
  // TODO(cais): More visible documentation of this in g3docs.
  std::vector<string> debug_urls;

  // Do not error out if debug op creation fails (e.g., due to dtype
  // incompatibility). Instead, just log the failure.
  bool tolerate_debug_op_creation_failures;
};

// EXPERIMENTAL. Options for initializing DebuggerState.
struct DebugOptions {
  // Debugging options
  std::vector<DebugTensorWatch> debug_tensor_watch_opts;

  // Caller-specified global step count.
  // Note that this is distinct from the session run count and the executor
  // step count.
  int64 global_step;
};

struct DebuggedSourceFile {
  // The host name on which a source code file is located.
  string host;

  // Path to the source code file.
  string file_path;

  // The timestamp at which the source code file is last modified.
  int64 last_modified;

  // Byte size of the file.
  int64 bytes;

  // Line-by-line content of the source code file.
  std::vector<string> lines;
};

struct DebuggedSourceFiles {
  // A collection of source code files.
  std::vector<DebuggedSourceFile> source_files;
};

/// tensorflow/tensorflow/core/protobuf/rewriter_config.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "RewriterConfigProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct AutoParallelOptions {
  bool enable;
  int32 num_replicas;
};

struct RewriterConfig {
  // Graph rewriting is experimental and subject to change, not covered by any
  // API stability guarantees.

  // Configuration options for the meta-optimizer. Unless otherwise noted, these
  // configuration options do not apply to explicitly triggered optimization
  // passes in the optimizers field.

  enum Toggle {
    DEFAULT = 0,
    ON = 1,
    OFF = 2,
    // Enable some aggressive optimizations that use assumptions that TF graphs
    // may break. For example, assume the shape of a placeholder matches its
    // actual feed.
    AGGRESSIVE = 3
  };

  // Optimize tensor layouts
  Toggle layout_optimizer;
  // Fold constants (default is ON)
  Toggle constant_folding;
  // Arithmetic optimizations (default is ON)
  Toggle arithmetic_optimization;
  // Control dependency optimizations (default is ON).
  Toggle dependency_optimization;
  // If true, don't remove unnecessary ops from the graph
  bool disable_model_pruning;

  enum MemOptType {
    // The default setting (currently disabled)
    DEFAULT_MEM_OPT = 0,
    // Disabled in the meta-optimizer.
    NO_MEM_OPT = 1,
    // Driven by manual op-level annotations.
    MANUAL = 2,
    // Driven by heuristics. The behavior of these heuristics is subject to
    // change. Currently includes an experimental recomputation and swapping
    // heuristics. Manual annotations are respected, but additional nodes are
    // selected automatically.
    SWAPPING_HEURISTICS = 4,
    RECOMPUTATION_HEURISTICS = 5,
    // Use any combination of swapping and recomputation heuristics.
    HEURISTICS = 3
  };
  // Configures memory optimization passes through the meta-optimizer. Has no
  // effect on manually requested memory optimization passes in the optimizers
  // field.
  MemOptType memory_optimization;
  // The prefix for nodes which are valid outputs of recomputations. Inputs to
  // nodes with this name prefix may be recomputed (subject either to manual
  // annotation of those input nodes or to manual annotation and heuristics
  // depending on memory_optimization), but the prefixed nodes themselves will
  // not be recomputed. Typically this will be "gradients/", indicating that
  // activations from the forward pass of a graph may be recomputed as inputs to
  // gradients, but may be adjusted if gradients are inside a name scope or if
  // inputs to non-gradients should be recomputed. Defaults to "gradients/" if
  // empty or not set.
  string memory_optimizer_target_node_name_prefix;

  // Configures AutoParallel optimization passes either through the
  // meta-optimizer or when manually specified through the optimizers field.
  AutoParallelOptions auto_parallel;

  // If non-empty, will use this as an alternative way to specify a list of
  // optimizations to turn on and the order of the optimizations (replacing the
  // meta-optimizer).
  //
  // Of the RewriterConfig options, only the AutoParallel configuration options
  // (the auto_parallel field) apply to manually requested optimization passes
  // ("autoparallel"). Memory optimization passes ("memory") invoked here are
  // not configurable (in contrast to memory optimization passes through the
  // meta-optimizer) and act only on manual op annotations.
  std::vector<string> optimizers;
};

/// tensorflow/tensorflow/core/protobuf/config.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ConfigProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.framework";

struct GPUOptions {
  // A value between 0 and 1 that indicates what fraction of the
  // available GPU memory to pre-allocate for each process.  1 means
  // to pre-allocate all of the GPU memory, 0.5 means the process
  // allocates ~50% of the available GPU memory.
  double per_process_gpu_memory_fraction;

  // The type of GPU allocation strategy to use.
  //
  // Allowed values:
  // "": The empty string (default) uses a system-chosen default
  //     which may change over time.
  //
  // "BFC": A "Best-fit with coalescing" algorithm, simplified from a
  //        version of dlmalloc.
  string allocator_type;

  // Delay deletion of up to this many bytes to reduce the number of
  // interactions with gpu driver code.  If 0, the system chooses
  // a reasonable default (several MBs).
  int64 deferred_deletion_bytes;

  // If true, the allocator does not pre-allocate the entire specified
  // GPU memory region, instead starting small and growing as needed.
  bool allow_growth;

  // A comma-separated list of GPU ids that determines the 'visible'
  // to 'virtual' mapping of GPU devices.  For example, if TensorFlow
  // can see 8 GPU devices in the process, and one wanted to map
  // visible GPU devices 5 and 3 as "/device:GPU:0", and "/device:GPU:1", then one
  // would specify this field as "5,3".  This field is similar in
  // spirit to the CUDA_VISIBLE_DEVICES environment variable, except
  // it applies to the visible GPU devices in the process.
  //
  // NOTE: The GPU driver provides the process with the visible GPUs
  // in an order which is not guaranteed to have any correlation to
  // the *physical* GPU id in the machine.  This field is used for
  // remapping "visible" to "virtual", which means this operates only
  // after the process starts.  Users are required to use vendor
  // specific mechanisms (e.g., CUDA_VISIBLE_DEVICES) to control the
  // physical to visible device mapping prior to invoking TensorFlow.
  string visible_device_list;

  // In the event polling loop sleep this many microseconds between
  // PollEvents calls, when the queue is not empty.  If value is not
  // set or set to 0, gets set to a non-zero default.
  int32 polling_active_delay_usecs;

  // In the event polling loop sleep this many millisconds between
  // PollEvents calls, when the queue is empty.  If value is not
  // set or set to 0, gets set to a non-zero default.
  int32 polling_inactive_delay_msecs;

  // Force all tensors to be gpu_compatible. On a GPU-enabled TensorFlow,
  // enabling this option forces all CPU tensors to be allocated with Cuda
  // pinned memory. Normally, TensorFlow will infer which tensors should be
  // allocated as the pinned memory. But in case where the inference is
  // incomplete, this option can significantly speed up the cross-device memory
  // copy performance as long as it fits the memory.
  // Note that this option is not something that should be
  // enabled by default for unknown or very large models, since all Cuda pinned
  // memory is unpageable, having too much pinned memory might negatively impact
  // the overall host system performance.
  bool force_gpu_compatible;
};

// Options passed to the graph optimizer
struct OptimizerOptions {
  // If true, optimize the graph using common subexpression elimination.
  bool do_common_subexpression_elimination;

  // If true, perform constant folding optimization on the graph.
  bool do_constant_folding;

  // Constant folding optimization replaces tensors whose values can be
  // predetermined, with constant nodes. To avoid inserting too large constants,
  // the size of each constant created can be limited. If this value is zero, a
  // default limit of 10 MiB will be applied. If constant folding optimization
  // is disabled, this value is ignored.
  int64 max_folded_constant_in_bytes;

  // If true, perform function inlining on the graph.
  bool do_function_inlining;

  // Optimization level
  enum Level {
    // L1 is the default level.
    // Optimization performed at L1 :
    // 1. Common subexpression elimination
    // 2. Constant folding
    L1 = 0,

    // No optimizations
    L0 = -1
  };

  // Overall optimization level. The actual optimizations applied will be the
  // logical OR of the flags that this level implies and any flags already set.
  Level opt_level;

  // Control the use of the compiler/jit.  Experimental.
  enum GlobalJitLevel {
    DEFAULT = 0, // Default setting ("off" now, but later expected to be "on")
    OFF = -1,
    // The following settings turn on compilation, with higher values being
    // more aggressive.  Higher values may reduce opportunities for parallelism
    // and may use more memory.  (At present, there is no distinction, but this
    // is expected to change.)
    ON_1 = 1,
    ON_2 = 2
  };
  GlobalJitLevel global_jit_level;
};

struct GraphOptions {
  // Removed, use optimizer_options below.
  //reserved "skip_common_subexpression_elimination";
  //reserved 1;

  // If true, use control flow to schedule the activation of Recv nodes.
  // (Currently ignored.)
  bool enable_recv_scheduling;

  // Options controlling how graph is optimized.
  OptimizerOptions optimizer_options;

  // The number of steps to run before returning a cost model detailing
  // the memory usage and performance of each node of the graph. 0 means
  // no cost model.
  int64 build_cost_model;

  // The number of steps to skip before collecting statistics for the
  // cost model.
  int64 build_cost_model_after;

  // Annotate each Node with Op output shape data, to the extent it can
  // be statically inferred.
  bool infer_shapes;

  // Only place the subgraphs that are run, rather than the entire graph.
  //
  // This is useful for interactive graph building, where one might
  // produce graphs that cannot be placed during the debugging
  // process.  In particular, it allows the client to continue work in
  // a session after adding a node to a graph whose placement
  // constraints are unsatisfiable.
  bool place_pruned_graph;

  // If true, transfer float values between processes as bfloat16.
  bool enable_bfloat16_sendrecv;

  // If > 0, record a timeline every this many steps.
  // EXPERIMENTAL: This currently has no effect in MasterSession.
  int32 timeline_step;

  // Options that control the type and amount of graph rewriting.
  // Not currently configurable via the public Python API (i.e. there is no API
  // stability guarantee if you import RewriterConfig explicitly).
  RewriterConfig rewrite_options = 10;
};

struct ThreadPoolOptionProto {
  // The number of threads in the pool.
  //
  // 0 means the system picks a value based on where this option proto is used
  // (see the declaration of the specific field for more info).
  int32 num_threads;

  // The global name of the threadpool.
  //
  // If empty, then the threadpool is made and used according to the scope it's
  // in - e.g., for a session threadpool, it is used by that session only.
  //
  // If non-empty, then:
  // - a global threadpool associated with this name is looked
  //   up or created. This allows, for example, sharing one threadpool across
  //   many sessions (e.g., like the default behavior, if
  //   inter_op_parallelism_threads is not configured), but still partitioning
  //   into a large and small pool.
  // - if the threadpool for this global_name already exists, then it is an
  //   error if the existing pool was created using a different num_threads
  //   value as is specified on this call.
  // - threadpools created this way are never garbage collected.
  string global_name;
};

struct RPCOptions {
  // If true, always use RPC to contact the session target.
  //
  // If false (the default option), TensorFlow may use an optimized
  // transport for client-master communication that avoids the RPC
  // stack. This option is primarily for used testing the RPC stack.
  bool use_rpc_for_inprocess_master;
};

// Session configuration parameters.
// The system picks appropriate values for fields that are not set.
struct ConfigProto {
  // Map from device type name (e.g., "CPU" or "GPU" ) to maximum
  // number of devices of that type to use.  If a particular device
  // type is not found in the map, the system picks an appropriate
  // number.
  std::map<string, int32> device_count;

  // The execution of an individual op (for some op types) can be
  // parallelized on a pool of intra_op_parallelism_threads.
  // 0 means the system picks an appropriate number.
  int32 intra_op_parallelism_threads;

  // Nodes that perform blocking operations are enqueued on a pool of
  // inter_op_parallelism_threads available in each process.
  //
  // 0 means the system picks an appropriate number.
  //
  // Note that the first Session created in the process sets the
  // number of threads for all future sessions unless use_per_session_threads is
  // true or session_inter_op_thread_pool is configured.
  int32 inter_op_parallelism_threads;

  // If true, use a new set of threads for this session rather than the global
  // pool of threads. Only supported by direct sessions.
  //
  // If false, use the global threads created by the first session, or the
  // per-session thread pools configured by session_inter_op_thread_pool.
  //
  // This option is deprecated. The same effect can be achieved by setting
  // session_inter_op_thread_pool to have one element, whose num_threads equals
  // inter_op_parallelism_threads.
  bool use_per_session_threads;

  // This option is experimental - it may be replaced with a different mechanism
  // in the future.
  //
  // Configures session thread pools. If this is configured, then RunOptions for
  // a Run call can select the thread pool to use.
  //
  // The intended use is for when some session invocations need to run in a
  // background pool limited to a small number of threads:
  // - For example, a session may be configured to have one large pool (for
  // regular compute) and one small pool (for periodic, low priority work);
  // using the small pool is currently the mechanism for limiting the inter-op
  // parallelism of the low priority work.  Note that it does not limit the
  // parallelism of work spawned by a single op kernel implementation.
  // - Using this setting is normally not needed in training, but may help some
  // serving use cases.
  // - It is also generally recommended to set the global_name field of this
  // proto, to avoid creating multiple large pools. It is typically better to
  // run the non-low-priority work, even across sessions, in a single large
  // pool.
  std::vector<ThreadPoolOptionProto> session_inter_op_thread_pool;

  // Assignment of Nodes to Devices is recomputed every placement_period
  // steps until the system warms up (at which point the recomputation
  // typically slows down automatically).
  int32 placement_period;

  // When any filters are present sessions will ignore all devices which do not
  // match the filters. Each filter can be partially specified, e.g. "/job:ps"
  // "/job:worker/replica:3", etc.
  std::vector<string> device_filters;

  // Options that apply to all GPUs.
  GPUOptions gpu_options;

  // Whether soft placement is allowed. If allow_soft_placement is true,
  // an op will be placed on CPU if
  //   1. there's no GPU implementation for the OP
  // or
  //   2. no GPU devices are known or registered
  // or
  //   3. need to co-locate with reftype input(s) which are from CPU.
  bool allow_soft_placement;

  // Whether device placements should be logged.
  bool log_device_placement;

  // Options that apply to all graphs.
  GraphOptions graph_options;

  // Global timeout for all blocking operations in this session.  If non-zero,
  // and not overridden on a per-operation basis, this value will be used as the
  // deadline for all blocking operations.
  int64 operation_timeout_in_ms;

  // Options that apply when this session uses the distributed runtime.
  RPCOptions rpc_options;

  // Optional list of all workers to use in this session.
  ClusterDef cluster_def;

  // If true, any resources such as Variables used in the session will not be
  // shared with other sessions.
  bool isolate_session_state;

  // Next: 16
};

// Options for a single Run() call.
struct RunOptions {
  // TODO(pbar) Turn this into a TraceOptions proto which allows
  // tracing to be controlled in a more orthogonal manner?
  enum TraceLevel {
    NO_TRACE = 0,
    SOFTWARE_TRACE = 1,
    HARDWARE_TRACE = 2,
    FULL_TRACE = 3,
  };
  TraceLevel trace_level;

  // Time to wait for operation to complete in milliseconds.
  int64 timeout_in_ms;

  // The thread pool to use, if session_inter_op_thread_pool is configured.
  int32 inter_op_thread_pool;

  // Whether the partition graph(s) executed by the executor(s) should be
  // outputted via RunMetadata.
  bool output_partition_graphs;

  // EXPERIMENTAL.  Options used to initialize DebuggerState, if enabled.
  DebugOptions debug_options;

  // When enabled, causes tensor alllocation information to be included in
  // the error message when the Run() call fails because the allocator ran
  // out of memory (OOM).
  //
  // Enabling this option can slow down the Run() call.
  bool report_tensor_allocations_upon_oom;

  //reserved 4;
};

// Metadata output (i.e., non-Tensor) for a single Run() call.
struct RunMetadata {
  // Statistics traced for this step. Populated if tracing is turned on via the
  // "RunOptions" proto.
  // EXPERIMENTAL: The format and set of events may change in future versions.
  StepStats step_stats;

  // The cost graph for the computation defined by the run call.
  CostGraphDef cost_graph;

  // Graphs of the partitions executed by executors.
  std::vector<GraphDef> partition_graphs;
};

/// tensorflow/tensorflow/core/protobuf/tensorflow_server.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "ServerProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

// Defines the configuration of a single TensorFlow server.
struct ServerDef {
  // The cluster of which this server is a member.
  ClusterDef cluster;

  // The name of the job of which this server is a member.
  //
  // NOTE(mrry): The `cluster` field must contain a `JobDef` with a `name` field
  // that matches this name.
  string job_name;

  // The task index of this server in its job.
  //
  // NOTE: The `cluster` field must contain a `JobDef` with a matching `name`
  // and a mapping in its `tasks` field for this index.
  int32 task_index;

  // The default configuration for sessions that run on this server.
  ConfigProto default_session_config;

  // The protocol to be used by this server.
  //
  // Acceptable values include: "grpc".
  string protocol;
};

/// tensorflow/tensorflow/core/protobuf/master.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "DistributedRuntimeProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

////////////////////////////////////////////////////////////////////////////////
//
// CreateSession method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

struct CreateSessionRequest {
  // The initial graph definition.
  GraphDef graph_def;

  // Configuration options.
  ConfigProto config;

  // The target string used from the client's perspective.
  string target;
};

struct CreateSessionResponse {
  // The session handle to be used in subsequent calls for the created session.
  //
  // The client must arrange to call CloseSession with this returned
  // session handle to close the session.
  string session_handle;

  // The initial version number for the graph, to be used in the next call
  // to ExtendSession.
  int64 graph_version;
};

////////////////////////////////////////////////////////////////////////////////
//
// ExtendSession method request/response protos.
//
// The "graph_def" specifies a set of nodes to be added to the session's graph.
//
// A typical "graph_def" will contain:
//
// * Zero or more new nodes with names that do not exist in the server-side
//   graph. These will be added to the graph.
//
// PRECONDITION: The server-side current version is req.current_version.
//   None of the names in req.graph_def appeared in previous successful calls to
//   CreateSession or ExtendSession with the same session_handle.
// POSTCONDITION: The server-side current version is resp.new_version.
//
////////////////////////////////////////////////////////////////////////////////

struct ExtendSessionRequest {
  // REQUIRED: session_handle must be returned by a CreateSession call
  // to the same master service.
  string session_handle;

  // REQUIRED: The nodes to be added to the session's graph. If any node has
  // the same name as an existing node, the operation will fail with
  // ILLEGAL_ARGUMENT.
  GraphDef graph_def;

  // REQUIRED: The version number of the graph to be extended. This will be
  // tested against the current server-side version number, and the operation
  // will fail with FAILED_PRECONDITION if they do not match.
  int64 current_graph_version;
};

struct ExtendSessionResponse {
  // TODO(mrry): Return something about the operation?

  // The new version number for the extended graph, to be used in the next call
  // to ExtendSession.
  int64 new_graph_version;
};

////////////////////////////////////////////////////////////////////////////////
//
// RunStep method request/response protos.
//
// The caller should provide the feeds needed by the graph and specify
// what nodes should be fetched.
//
////////////////////////////////////////////////////////////////////////////////

struct RunStepRequest {
  // REQUIRED: session_handle must be returned by a CreateSession call
  // to the same master service.
  string session_handle;

  // Tensors to be fed in the step. Each feed is a named tensor.
  std::vector<NamedTensorProto> feed;

  // Fetches. A list of tensor names. The caller expects a tensor to
  // be returned for each fetch[i] (see RunStepResponse.tensor). The
  // order of specified fetches does not change the execution order.
  std::vector<string> fetch;

  // Target Nodes. A list of node names. The named nodes will be run
  // to but their outputs will not be fetched.
  std::vector<string> target;

  // Options for the run call.
  RunOptions options;

  // Partial run handle (optional). If specified, this will be a partial run
  // execution, run up to the specified fetches.
  string partial_run_handle;
};

struct RunStepResponse {
  // NOTE: The order of the returned tensors may or may not match
  // the fetch order specified in RunStepRequest.
  std::vector<NamedTensorProto> tensor;

  // Returned metadata if requested in the options.
  RunMetadata metadata;
};

////////////////////////////////////////////////////////////////////////////////
//
// PartialRunSetup method request/response protos.
//
// The caller should provide the future partial run feeds, fetches, and targets.
// Then the caller can use RunStepRequest with is_partial set to make partial
// run calls.
//
////////////////////////////////////////////////////////////////////////////////

struct PartialRunSetupRequest {
  // REQUIRED: session_handle must be returned by a CreateSession call
  // to the same master service.
  string session_handle;

  // Tensors to be fed in future steps.
  std::vector<string> feed;

  // Fetches. A list of tensor names. The caller expects a tensor to be returned
  // for each fetch[i] (see RunStepResponse.tensor), for corresponding partial
  // RunStepRequests. The order of specified fetches does not change the
  // execution order.
  std::vector<string> fetch;

  // Target Nodes. A list of node names. The named nodes will be run in future
  // steps, but their outputs will not be fetched.
  std::vector<string> target;
};

struct PartialRunSetupResponse {
  // The unique handle corresponding to the ongoing partial run call setup by
  // the invocation to PartialRunSetup. This handle may be passed to
  // RunStepRequest to send and receive tensors for this partial run.
  string partial_run_handle;
};

////////////////////////////////////////////////////////////////////////////////
//
// CloseSession method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

struct CloseSessionRequest {
  // REQUIRED: session_handle must be returned by a CreateSession call
  // to the same master service.
  string session_handle;
};

struct CloseSessionResponse {
};

// Reset() allows misbehaving or slow sessions to be aborted and closed, and
// causes their resources eventually to be released.  Reset() does not wait
// for the computations in old sessions to cease; it merely starts the
// process of tearing them down.  However, if a new session is started after
// a Reset(), the new session is isolated from changes that old sessions
// (started prior to the Reset()) may continue to make to resources, provided
// all those resources are in containers listed in "containers".
//
// Old sessions may continue to have side-effects on resources not in
// containers listed in "containers", and thus may affect future
// sessions' results in ways that are hard to predict.  Thus, if well-defined
// behavior is desired, is it recommended that all containers be listed in
// "containers".  Similarly, if a device_filter is specified, results may be
// hard to predict.
struct ResetRequest {
  // A list of container names, which may be empty.
  //
  // If 'container' is not empty, releases resoures in the given
  // containers in all devices.
  //
  // If 'container' is empty, releases resources in the default
  // container in all devices.
  std::vector<string> container;

  // When any filters are present, only devices that match the filters
  // will be reset. Each filter can be partially specified,
  // e.g. "/job:ps" "/job:worker/replica:3", etc.
  std::vector<string> device_filters;
};

struct ResetResponse {
};

////////////////////////////////////////////////////////////////////////////////
//
// ListDevices method request/response protos.
//
// Returns information about the TensorFlow devices that are available
// to this master.
//
////////////////////////////////////////////////////////////////////////////////

struct ListDevicesRequest {
  // Optional: session_handle must be returned by a CreateSession call to the
  // same master service.
  //
  // When session_handle is empty, the ClusterSpec provided when the master was
  // started is used to compute the available devices. If the session_handle is
  // provided but not recognized, an error is returned. Finally, if a valid
  // session_handle is provided, the cluster configuration for that session is
  // used when computing the response.
  string session_handle;
};

struct ListDevicesResponse {
  std::vector<DeviceAttributes> local_device;
  std::vector<DeviceAttributes> remote_device;
};

/// tensorflow/tensorflow/core/protobuf/worker.proto

//option cc_enable_arenas = true;
//option java_outer_classname = "WorkerProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

////////////////////////////////////////////////////////////////////////////////
//
// GetStatus method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

struct GetStatusRequest {
};

struct GetStatusResponse {
  std::vector<DeviceAttributes> device_attributes;
};

////////////////////////////////////////////////////////////////////////////////
//
// CreateSession method request/response messages
//
// For each session,
//
////////////////////////////////////////////////////////////////////////////////

struct CreateWorkerSessionRequest {
  // Sessions are identified by a given handle.
  string session_handle;

  // Defines the configuration of a TensorFlow worker.
  ServerDef server_def;

  // If true, any resources such as Variables used in the session will not be
  // shared with other sessions.
  bool isolate_session_state;
};

struct CreateWorkerSessionResponse {
};

////////////////////////////////////////////////////////////////////////////////
//
// DeleteSession method request/response messages
//
// Deletes all worker-side state associated with the given session handle.
//
////////////////////////////////////////////////////////////////////////////////

struct DeleteWorkerSessionRequest {
  // Sessions are identified by a given handle.
  string session_handle;
};

struct DeleteWorkerSessionResponse {
};

////////////////////////////////////////////////////////////////////////////////
//
// RegisterGraph method request/response messages
//
// For each session, after the master placed every node on a device,
// it partitions the whole graph into many subgraphs. All the nodes in
// a subgraph were in the same worker, but potentially on many devices
// owned by that worker (e.g. cpu0, plus gpu0, gpu1, ..., gpu7). The
// master registers subgraphs for a worker before running any steps. A
// successful registration returns a graph handle to be used in latter
// RunGraph requests.
//
////////////////////////////////////////////////////////////////////////////////

struct RegisterGraphRequest {
  // Subgraphs are scoped within one session.
  string session_handle;

  // "graph_def" has the subgraph of nodes for this worker, with each node
  // having its device_name filled in.
  GraphDef graph_def;

  // True iff the graph (before partitioning) contains control flow nodes.
  //
  // As of 01/11/2015, this is no longer set by clients.
  bool has_control_flow; // [deprecated = true];

  // Configuration options for the session in which this graph was created.
  GraphOptions graph_options;

  // Field(s) used by TensorFlow Debugger (tfdbg).
  DebugOptions debug_options;
};

struct RegisterGraphResponse {
  // If the registration succeeds, returns an opaque graph_handle to
  // the master. The master calls RunGraph with graph_handle to
  // compute different steps.
  string graph_handle;
};

////////////////////////////////////////////////////////////////////////////////
//
// DeregisterGraph method request/response messages
//
// The master deregisters the given graph_handle when the graph is no
// longer needed (e.g., the overall graph is re-scheduled and nodes
// are re-placed).
//
// The worker deregisters a graph_handle automatically according to on
// a TTL-base policy in case of master restarts.
//
////////////////////////////////////////////////////////////////////////////////

struct DeregisterGraphRequest {
  // The session_handle used when registering the graph. If session_handle is
  // empty, a single global namespace is used.
  string session_handle;

  // REQUIRED: graph_handle must be returned by a RegisterGraph call
  // to the same WorkerService.
  string graph_handle;
};

struct DeregisterGraphResponse {
  // TODO(mrry): Optionally add summary stats for the graph.
};

////////////////////////////////////////////////////////////////////////////////
//
// CleanupAll method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

struct CleanupAllRequest {
  // A list of container names.
  //
  // If 'container' is not empty, releases resources in the given
  // containers in all devices.
  //
  // If 'container' is empty, releases resources in the default
  // container in all devices.
  std::vector<string> container;
};

struct CleanupAllResponse {
};

////////////////////////////////////////////////////////////////////////////////
//
// RunGraph request / response messages
//
// The worker executes all subgraphs registered under graph_handle.
// RunGraph returns after the execution finishes or an error is
// encountered.
// A sequence of RunGraphRequests with is_partial may be sent to RunGraph for
// partial graph execution.
//
////////////////////////////////////////////////////////////////////////////////

// Options specific to the execution of a single step.
struct ExecutorOpts {
  bool record_cost;
  bool record_timeline;
  bool record_partition_graphs;
  bool report_tensor_allocations_upon_oom;
};

struct RunGraphRequest {
  // session_handle is the master-generated unique id for this session.
  // If session_handle is non-empty, it must be the same as used when
  // registering the graph. If it is empty, a single global namespace is used to
  // search for the graph_handle.
  string session_handle;

  // REQUIRED: graph_handle must be returned by a RegisterGraph call
  // to the same WorkerService.
  string graph_handle;

  // A unique ID to distinguish different runs of the same graph.
  //
  // The master generates a global unique `step_id` to distinguish
  // different runs of the graph computation. Subgraphs communicate
  // (e.g., send/recv ops) with each other using `step_id` to
  // distinguish tensors generated by different runs.
  int64 step_id;

  // Options for this step.
  ExecutorOpts exec_opts;

  // Runs the graph.
  //
  // Sends the tensors in "send" into the graph before the run and
  // fetches the keys into `RunGraphResponse.recv` after the run.
  std::vector<NamedTensorProto> send;
  std::vector<string> recv_key;

  // True if the RunGraphRequest is a partial run request.
  bool is_partial;
  // True if this is the last partial run request in a sequence of requests.
  bool is_last_partial_run;

  // Next: 9
};

struct RunGraphResponse {
  // A list of tensors corresponding to those requested by
  // `RunGraphRequest.recv_key`.
  std::vector<NamedTensorProto> recv;

  // If the request asked for execution stats, the cost graph, or the partition
  // graphs, these are returned here.
  // TODO(suharshs): Package these in a RunMetadata instead.
  StepStats step_stats;
  CostGraphDef cost_graph;
  std::vector<GraphDef> partition_graph;
};

////////////////////////////////////////////////////////////////////////////////
//
// CleanupGraph method request/response messages
//
// After the master receives RunGraph responses from all workers, the
// master instructs every worker to cleanup any remaining state of a
// step (e.g. tensors buffered by a `Send` op but not picked up by
// other workers). The master does not necessarily need to wait for
// completion of CleanupGraph calls.
//
// Workers should cleanup step states automatically according to a
// TTL-based policy in case of master restarts.
//
////////////////////////////////////////////////////////////////////////////////

struct CleanupGraphRequest {
  int64 step_id;
};

struct CleanupGraphResponse {
};

////////////////////////////////////////////////////////////////////////////////
//
// RecvTensor method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

struct RecvTensorRequest {
  // The step in which the tensor will be produced.
  //
  // REQUIRED: This must eventually correspond to the `step_id` passed
  // into a RunGraph call on the same WorkerService.
  int64 step_id;

  // A key that identifies the tensor to be received.
  string rendezvous_key;

  // If true, use an out-of-band DMA mechanism to transfer the
  // received tensor.
  bool dma_ok;

  // Optional information on client-side device locality.
  DeviceLocality client_locality;

  // Optional information on server-side device locality.
  DeviceLocality server_locality;

  // Optional information needed by the RPC subsystem.
  //google.protobuf.Any transport_options = 6;
};

struct RecvTensorResponse {
  // The tensor as a proto.
  TensorProto tensor;

  // If true, this tensor was the output of a dead node, and the
  // content is invalid.
  bool is_dead;

  // The time at which tensor was available and started to be returned.
  int64 send_start_micros;

  // Optional additional information about how to receive the tensor,
  // e.g. in the event that `RecvTensorRequest.dma_ok` was true.
  //google.protobuf.Any transport_options = 4;
};

////////////////////////////////////////////////////////////////////////////////
//
// Logging method request/response messages
//
// NOTE(mrry): This feature is not supported in the open-source
// version, and these messages are expected to change.
//
////////////////////////////////////////////////////////////////////////////////

// Out-of-band request to begin or end logging, or
// to retrieve logs for particular steps.
struct LoggingRequest {
  // If true, RPC logging will be activated.
  bool rpc_logging;

  // If true, discard any saved logging data (for all steps).
  bool clear;

  // When set, requests all saved log data pertaining to the step.
  // Any log data retrieved is eliminated from the store and cannot be
  // retrieved again.
  std::vector<int64> fetch_step_id;
};

struct LabeledStepStats {
  int64 step_id;
  StepStats step_stats;
};

struct LoggingResponse {
  std::vector<LabeledStepStats> step;
};

////////////////////////////////////////////////////////////////////////////////
//
// Tracing method request/response messages
//
// NOTE(mrry): This feature is not supported in the open-source
// version, and these messages are expected to change.
//
////////////////////////////////////////////////////////////////////////////////

struct TraceOpts {
  // Length of the trace to be taken, in seconds.
  double duration;
  // If true, capture step profile locally in each worker. Currently
  // unimplemented.
  bool use_step_profiler;
  // If true, capture kernel events from each worker.
  bool use_kernel_profiler;
  // If true, capture extended profiling events from TensorFlow process.
  bool use_extended_profiler;
  // If true, capture GPU profiling events locally on each
  // machine. Currently unimplemented.
  bool use_gpu_profiler;
  // If true, collect sampled profile events. Currently unimplemented.
  bool use_sample_profiler;
};

// Out-of-band request to configure distributed tracing.
struct TracingRequest {
  TraceOpts options;
};

struct TracingResponse {
};

/// tensorflow/tensorflow/core/protobuf/master_service.proto

//option java_outer_classname = "MasterServiceProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

////////////////////////////////////////////////////////////////////////////////
//
// MasterService defines a TensorFlow service with which a client can
// interact to execute a distributed TensorFlow computation.
//
// A master service keeps track of multiple "master sessions". Each
// session encapsulates a computation graph and its associated state,
// and typically corresponds to a single "client session" (e.g. a
// `tensorflow::Session` instance).
//
// A session is responsible for the following:
// * assigning each node to a device (locally or remotely) using a
//   placement algorithm. This may make decisions based on collected
//   statistics from the workers in the system (e.g., memory usage,
//   bandwidth consumption, etc.)
//
// * inserting intermediate nodes and edges to support cross-device
//   and cross-process data flows and resource management.
//
// * issuing commands to workers to execute the subgraphs associated
//   with those workers.
//
// Typically, a client carries out an iterative computation
// (e.g. training) by invoking RPCs against the master in a
// client-side loop. The client first creates a client session that
// connects to a particular master (using gRPC for example). The
// master creates a corresponding master session that is hosted on
// the master and caches state between the client's invocations.
//
// After the session is established, the master returns an opaque
// handle to the client that can be used to associate the client and
// master sessions.
//
// The client may send an initial graph to the master in the
// CreateSession call, and add nodes to the graph using ExtendSession.
//
// The most frequent operation a master is "RunStep", which implements
// the `Session::Run()` API. It supports feeding in arguments,
// executing a dataflow computation, and fetching arguments.
//
// Finally, when the client no longer needs the session, it should
// close the session by invoking CloseSession, which allows the master
// to reclaim resources associated with the session. The master may
// implement a garbage collection scheme that closes sessions that
// have been inactive for some time.
//
// For example, the following pseudo-code illustrates how a client
// interacts with a master:
//
// stub = NewStub("/job:mnist/replica:0/task:0")
// {handle} = stub->CreateSession({graph_def})
// do {
//   stub->RunStep({handle, {feeds}, {fetches}})
//   // The client can evaluate a predicate locally, based on the
//   // result of `fetches`, to determine whether to terminate. For
//   // example, it might fetch the loss and evaluate whether it is less
//   // than some threshold.
// } while (!should_stop({fetches}));
// stub->CloseSession({handle})
//
////////////////////////////////////////////////////////////////////////////////

struct MasterService { // service
  // Creates a session.
  //rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);

  // Extends a session.
  //rpc ExtendSession(ExtendSessionRequest) returns (ExtendSessionResponse);

  // Prepares future partial run calls.
  //rpc PartialRunSetup(PartialRunSetupRequest) returns (PartialRunSetupResponse);

  // Drives the graph computation.
  //rpc RunStep(RunStepRequest) returns (RunStepResponse);

  // Closes a session.
  //rpc CloseSession(CloseSessionRequest) returns (CloseSessionResponse);

  // List the devices usable by the master.
  //rpc ListDevices(ListDevicesRequest) returns (ListDevicesResponse);

  // Close and abandon all existing sessions.  Ongoing computations
  // will no longer affect fresh ones via the resources in containers listed in
  // the ResetRequest.  See ResetRequest for more details.
  //rpc Reset(ResetRequest) returns (ResetResponse);
};

/// tensorflow/tensorflow/core/protobuf/worker_service.proto

//option java_outer_classname = "WorkerServiceProtos";
//option java_multiple_files = true;
//option java_package = "org.tensorflow.distruntime";

////////////////////////////////////////////////////////////////////////////////
//
// WorkerService defines a TensorFlow service that executes dataflow
// graphs on a set of local devices, on behalf of a MasterService.
//
// A worker service keeps track of multiple "registered graphs". Each
// registered graph is a subgraph of a client's graph, corresponding to
// only the nodes that should execute on this worker (and any
// additional nodes necessary for inter-process communication using
// the `RecvTensor` method).
//
////////////////////////////////////////////////////////////////////////////////

struct WorkerService { // service
  // See worker.proto for details.
  //rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);

  // See worker.proto for details.
  //rpc CreateWorkerSession(CreateWorkerSessionRequest)
      //returns (CreateWorkerSessionResponse);

  // See worker.proto for details.
  // rpc DeleteWorkerSession(DeleteWorkerSessionRequest)
      //returns (DeleteWorkerSessionResponse);

  // See worker.proto for details.
  //rpc RegisterGraph(RegisterGraphRequest) returns (RegisterGraphResponse);

  // See worker.proto for details.
  //rpc DeregisterGraph(DeregisterGraphRequest) returns (DeregisterGraphResponse);

  // See worker.proto for details.
  //rpc RunGraph(RunGraphRequest) returns (RunGraphResponse);

  // See worker.proto for details.
  //rpc CleanupGraph(CleanupGraphRequest) returns (CleanupGraphResponse);

  // See worker.proto for details.
  //rpc CleanupAll(CleanupAllRequest) returns (CleanupAllResponse);

  // See worker.proto for details.
  //rpc RecvTensor(RecvTensorRequest) returns (RecvTensorResponse) {
    // RecvTensor Method
  //}

  // See worker.proto for details.
  //rpc Logging(LoggingRequest) returns (LoggingResponse);

  // See worker.proto for details.
  //rpc Tracing(TracingRequest) returns (TracingResponse);
};

} // namespace mytensorflow
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TENSORFLOW_PROTOBUF_PROTO_H_