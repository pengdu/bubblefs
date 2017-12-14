
// apollo/modules/routing/proto/topo_graph.proto

#ifndef BUBBLEFS_UTILS_APOLLO_PROTO_TOPO_GRAPH_H_
#define BUBBLEFS_UTILS_APOLLO_PROTO_TOPO_GRAPH_H_

#include <string>
#include <vector>
#include "utils/apollo_proto_geometry.h"

namespace bubblefs {
namespace myapollo {
namespace routing { 

struct CurvePoint {
  double s;
};

struct CurveRange {
  CurvePoint start;
  CurvePoint end;
};

struct Node {
  std::string lane_id;
  double length;
  CurveRange left_out;
  CurveRange right_out;
  double cost;
  hdmap::Curve central_curve;
  bool is_virtual;
  std::string road_id;
};

struct Edge {
  enum DirectionType {
    FORWARD = 0,
    LEFT = 1,
    RIGHT = 2,
  };

  std::string from_lane_id;
  std::string to_lane_id;
  double cost;
  DirectionType direction_type;
};

struct Graph {
  std::string hdmap_version;
  std::string hdmap_district;
  std::vector<Node> node;
  std::vector<Edge> edge;
};
  
} // namespace routing
} // namespace myapollo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_APOLLO_PROTO_TOPO_GRAPH_H_