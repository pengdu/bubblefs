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

// apollo/modules/common/math/sin_table.h

/**
 * @file
 * @brief Exports the SIN_TABLE, used by the Angle class.
 */

#ifndef BUBBLEFS_UTILS_APOLLO_SIN_TABLE_H_
#define BUBBLEFS_UTILS_APOLLO_SIN_TABLE_H_

/**
 * @namespace apollo::common::math
 * @brief apollo::common::math
 */
namespace bubblefs {
namespace myapollo {
namespace common {
namespace math {

//! Used by Angle class to speed-up computation of trigonometric functions.
#define SIN_TABLE_SIZE 16385
extern const float SIN_TABLE[SIN_TABLE_SIZE];

}  // namespace math
}  // namespace common
}  // namespace myapollo
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_APOLLO_SIN_TABLE_H_