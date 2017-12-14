/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// apollo/modules/planning/math/curve_math.cc

/**
 * @file curvature.cc
 **/
#include "utils/apollo_curve_math.h"

#include <cmath>

namespace bubblefs {
namespace myapollo {
namespace planning {

// kappa = (dx * d2y - dy * d2x) / [(dx * dx + dy * dy)^(3/2)]
double CurveMath::ComputeCurvature(const double dx, const double d2x,
                                   const double dy, const double d2y) {
  const double a = dx * d2y - dy * d2x;
  constexpr double kOrder = 1.5;
  const double b = std::pow(dx * dx + dy * dy, kOrder);
  return a / b;
}

double CurveMath::ComputeCurvatureDerivative(const double dx, const double d2x,
                                             const double d3x, const double dy,
                                             const double d2y,
                                             const double d3y) {
  const double a = dx * d2y - dy * d2x;
  const double b = dx * d3y - dy * d3x;
  const double c = dx * d2x + dy * d2y;
  const double d = dx * dx + dy * dy;

  return (b * d - 3.0 * a * c) / std::pow(d, 3.0);
}

}  // namespace planning
}  // namespace myapollo
}  // namespace bubblefs