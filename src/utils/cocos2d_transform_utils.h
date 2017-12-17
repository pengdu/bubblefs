/****************************************************************************
Copyright (c) 2009      Valentin Milea
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

// cocos2d-x/cocos/math/TransformUtils.h

#ifndef BUBBLEFS_UTILS_COCOS2D_TRANSFORM_UTILS_H_
#define BUBBLEFS_UTILS_COCOS2D_TRANSFORM_UTILS_H_

// TODO: when in MAC or windows, it includes <OpenGL/gl.h>
#include "platform/cocos2d_gl.h"
#include "platform/cocos2d_macros.h"

namespace bubblefs {
namespace mycocos2d {

struct AffineTransform;
/**@{
 Conversion between mat4*4 and AffineTransform.
 @param m The Mat4*4 pointer.
 @param t Affine transform.
 */
void CGAffineToGL(const AffineTransform &t, GLfloat *m);
void GLToCGAffine(const GLfloat *m, AffineTransform *t);

} // namespace mycocos2d
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_COCOS2D_TRANSFORM_UTILS_H_