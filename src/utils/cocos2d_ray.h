/****************************************************************************
 Copyright (c) 2014-2017 Chukong Technologies Inc.
 
 http://www.cocos2d-x.org
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ****************************************************************************/

// cocos2d-x/cocos/3d/CCRay.h

#ifndef BUBBLEFS_UTILS_COCOS2D_RAY_H_
#define BUBBLEFS_UTILS_COCOS2D_RAY_H_

#include "utils/cocos2d_math.h"
#include "utils/cocos2d_aabb.h"
#include "utils/cocos2d_obb.h"
#include "utils/cocos2d_plane.h"

namespace bubblefs {
namespace mycocos2d {

/**
 * @addtogroup _3d
 * @{
 */

/**
 * @brief Ray is a line with one end. usually use it to check intersects with some object,such as Plane, OBB, AABB
 * @js NA
 **/
class Ray
{
public:
    /**
     * Constructor.
     *
     * @lua new
     */
    Ray();

    /**
     * Constructor.
     * @lua NA
     */
    Ray(const Ray& ray);
    
    /**
     * Constructs a new ray initialized to the specified values.
     *
     * @param origin The ray's origin.
     * @param direction The ray's direction.
     * @lua new
     */
    Ray(const Vec3& origin, const Vec3& direction);

    /**
     * Destructor.
     * @lua NA
     */
    ~Ray();

    /**
     * Check whether this ray intersects with the specified AABB.
     */
    bool intersects(const AABB& aabb, float* distance = nullptr) const;
    
    /**
     * Check whether this ray intersects with the specified OBB.
     */
    bool intersects(const OBB& obb, float* distance = nullptr) const;

    float dist(const Plane& plane) const;
    Vec3 intersects(const Plane& plane) const;
    
    /**
     * Sets this ray to the specified values.
     *
     * @param origin The ray's origin.
     * @param direction The ray's direction.
     */
    void set(const Vec3& origin, const Vec3& direction);

    /**
     * Transforms this ray by the given transformation matrix.
     *
     * @param matrix The transformation matrix to transform by.
     */
    void transform(const Mat4& matrix);

    Vec3 _origin;        // The ray origin position.
    Vec3 _direction;     // The ray direction vector.
};

} // namespace mycocos2d
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_COCOS2D_RAY_H_