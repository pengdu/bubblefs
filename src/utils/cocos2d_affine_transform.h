/****************************************************************************
Copyright (c) 2010-2012 cocos2d-x.org
Copyright (c) 2013-2017 Chukong Technologies Inc.
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

// cocos2d-x/cocos/math/CCAffineTransform.h

#ifndef BUBBLEFS_UTILS_COCOS2D_AFFINE_TRANSFORM_H_
#define BUBBLEFS_UTILS_COCOS2D_AFFINE_TRANSFORM_H_

#include "platform/cocos2d_macros.h"
#include "utils/cocos2d_geometry.h"
#include "utils/cocos2d_math.h"

namespace bubblefs {
namespace mycocos2d {

/**@{
 Affine transform
 a   b    0
 c   d    0
 tx  ty   1
 
 Identity
 1   0    0
 0   1    0
 0   0    1
 */
struct AffineTransform {
    float a, b, c, d;
    float tx, ty;

    static const AffineTransform IDENTITY;
};

/**@}*/

/**Make affine transform.*/
AffineTransform __CCAffineTransformMake(float a, float b, float c, float d, float tx, float ty);
#define AffineTransformMake __CCAffineTransformMake

/**Multiply point (x,y,1) by a  affine transform.*/
Vec2 __CCPointApplyAffineTransform(const Vec2& point, const AffineTransform& t);
#define PointApplyAffineTransform __CCPointApplyAffineTransform

/**Multiply size (width,height,0) by a  affine transform.*/
Size __CCSizeApplyAffineTransform(const Size& size, const AffineTransform& t);
#define SizeApplyAffineTransform __CCSizeApplyAffineTransform
/**Make identity affine transform.*/
AffineTransform AffineTransformMakeIdentity();
/**Transform Rect, which will transform the four vertices of the point.*/
Rect RectApplyAffineTransform(const Rect& rect, const AffineTransform& anAffineTransform);
/**@{
 Transform vec2 and Rect by Mat4.
 */
Rect RectApplyTransform(const Rect& rect, const Mat4& transform);
Vec2 PointApplyTransform(const Vec2& point, const Mat4& transform);
/**@}*/
/**
 Translation, equals
 1  0  1
 0  1  0   * affine transform
 tx ty 1
 */
AffineTransform AffineTransformTranslate(const AffineTransform& t, float tx, float ty);
/**
 Rotation, equals
 cos(angle)   sin(angle)   0
 -sin(angle)  cos(angle)   0  * AffineTransform
 0            0            1
 */
AffineTransform AffineTransformRotate(const AffineTransform& aTransform, float anAngle);
/**
 Scale, equals
 sx   0   0
 0    sy  0  * affineTransform
 0    0   1
 */
AffineTransform AffineTransformScale(const AffineTransform& t, float sx, float sy);
/**Concat two affine transform, t1 * t2*/
AffineTransform AffineTransformConcat(const AffineTransform& t1, const AffineTransform& t2);
/**Compare affine transform.*/
bool AffineTransformEqualToTransform(const AffineTransform& t1, const AffineTransform& t2);
/**Get the inverse of affine transform.*/
AffineTransform AffineTransformInvert(const AffineTransform& t);
/**Concat Mat4, return t1 * t2.*/
Mat4 TransformConcat(const Mat4& t1, const Mat4& t2);

extern const AffineTransform AffineTransformIdentity;

} // namespace mycocos2d
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_COCOS2D_AFFINE_TRANSFORM_H_