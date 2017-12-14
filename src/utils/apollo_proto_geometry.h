
// apollo/modules/common/proto/geometry.proto
// apollo/modules/map/proto/map_geometry.proto

#ifndef BUBBLEFS_UTILS_APOLLO_PROTO_GEOMETRY_H_
#define BUBBLEFS_UTILS_APOLLO_PROTO_GEOMETRY_H_

#include <vector>

namespace bubblefs {
namespace myapollo {
  
namespace common {  
  
// A point in the map reference frame. The map defines an origin, whose
// coordinate is (0, 0, 0).
// Most modules, including localization, perception, and prediction, generate
// results based on the map reference frame.
// Currently, the map uses Universal Transverse Mercator (UTM) projection. See
// the link below for the definition of map origin.
//   https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
// The z field of PointENU can be omitted. If so, it is a 2D location and we do
// not care its height.
struct PointENU {
  double x;  // East from the origin, in meters.
  double y;  // North from the origin, in meters.
  double z;  // Up from the WGS-84 ellipsoid, in meters.
};

// A point in the global reference frame. Similar to PointENU, PointLLH allows
// omitting the height field for representing a 2D location.
struct PointLLH {
  // Longitude in degrees, ranging from -180 to 180.
  double lon;
  // Latitude in degrees, ranging from -90 to 90.
  double lat;
  // WGS-84 ellipsoid height in meters.
  double height;
};

// A general 2D point. Its meaning and units depend on context, and must be
// explained in comments.
struct Point2D {
  double x;
  double y;
};

// A general 3D point. Its meaning and units depend on context, and must be
// explained in comments.
struct Point3D {
  double x;
  double y;
  double z;
};

// A unit quaternion that represents a spatial rotation. See the link below for
// details.
//   https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
// The scalar part qw can be omitted. In this case, qw should be calculated by
//   qw = sqrt(1 - qx * qx - qy * qy - qz * qz).
struct Quaternion {
  double qx;
  double qy;
  double qz;
  double qw;
};

} // namespace common

namespace hdmap {

// Polygon, not necessary convex.
struct Polygon {
  common::PointENU point;
};

// Straight line segment.
struct LineSegment {
  common::PointENU point;
};

// Generalization of a line.
struct CurveSegment {
  LineSegment line_segment;
  double s;  // start position (s-coordinate)
  common::PointENU start_position;
  double heading; // start orientation
  double length;
};

// An object similar to a line but that need not be straight.
struct Curve {
  std::vector<CurveSegment> segment;
};

} // namespace hdmap

} // namespace myapollo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_APOLLO_PROTO_GEOMETRY_H_