/**
Copyright 2013 BlackBerry Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 
Original file from GamePlay3D: http://gameplay3d.org
This file was modified to fit the cocos2d-x project
*/

// cocos2d-x/cocos/math/MathUtilSSE.inl
// cocos2d-x/cocos/math/MathUtil.inl
// cocos2d-x/cocos/math/MathUtil.cpp

#include "platform/cocos2d_macros.h"
#include "utils/cocos2d_math_util.h"

namespace bubblefs {
namespace mycocos2d {

#ifdef __SSE__

void MathUtil::addMatrix(const __m128 m[4], float scalar, __m128 dst[4])
{
    __m128 s = _mm_set1_ps(scalar);
    dst[0] = _mm_add_ps(m[0], s);
    dst[1] = _mm_add_ps(m[1], s);
    dst[2] = _mm_add_ps(m[2], s);
    dst[3] = _mm_add_ps(m[3], s);
}

void MathUtil::addMatrix(const __m128 m1[4], const __m128 m2[4], __m128 dst[4])
{
    dst[0] = _mm_add_ps(m1[0], m2[0]);
    dst[1] = _mm_add_ps(m1[1], m2[1]);
    dst[2] = _mm_add_ps(m1[2], m2[2]);
    dst[3] = _mm_add_ps(m1[3], m2[3]);
}

void MathUtil::subtractMatrix(const __m128 m1[4], const __m128 m2[4], __m128 dst[4])
{
    dst[0] = _mm_sub_ps(m1[0], m2[0]);
    dst[1] = _mm_sub_ps(m1[1], m2[1]);
    dst[2] = _mm_sub_ps(m1[2], m2[2]);
    dst[3] = _mm_sub_ps(m1[3], m2[3]);
}

void MathUtil::multiplyMatrix(const __m128 m[4], float scalar, __m128 dst[4])
{
    __m128 s = _mm_set1_ps(scalar);
    dst[0] = _mm_mul_ps(m[0], s);
    dst[1] = _mm_mul_ps(m[1], s);
    dst[2] = _mm_mul_ps(m[2], s);
    dst[3] = _mm_mul_ps(m[3], s);
}

void MathUtil::multiplyMatrix(const __m128 m1[4], const __m128 m2[4], __m128 dst[4])
{
    __m128 dst0, dst1, dst2, dst3;
        {
                __m128 e0 = _mm_shuffle_ps(m2[0], m2[0], _MM_SHUFFLE(0, 0, 0, 0));
                __m128 e1 = _mm_shuffle_ps(m2[0], m2[0], _MM_SHUFFLE(1, 1, 1, 1));
                __m128 e2 = _mm_shuffle_ps(m2[0], m2[0], _MM_SHUFFLE(2, 2, 2, 2));
                __m128 e3 = _mm_shuffle_ps(m2[0], m2[0], _MM_SHUFFLE(3, 3, 3, 3));
        
                __m128 v0 = _mm_mul_ps(m1[0], e0);
                __m128 v1 = _mm_mul_ps(m1[1], e1);
                __m128 v2 = _mm_mul_ps(m1[2], e2);
                __m128 v3 = _mm_mul_ps(m1[3], e3);
        
                __m128 a0 = _mm_add_ps(v0, v1);
                __m128 a1 = _mm_add_ps(v2, v3);
                __m128 a2 = _mm_add_ps(a0, a1);
        
                dst0 = a2;
        }
    
        {
                __m128 e0 = _mm_shuffle_ps(m2[1], m2[1], _MM_SHUFFLE(0, 0, 0, 0));
                __m128 e1 = _mm_shuffle_ps(m2[1], m2[1], _MM_SHUFFLE(1, 1, 1, 1));
                __m128 e2 = _mm_shuffle_ps(m2[1], m2[1], _MM_SHUFFLE(2, 2, 2, 2));
                __m128 e3 = _mm_shuffle_ps(m2[1], m2[1], _MM_SHUFFLE(3, 3, 3, 3));
        
                __m128 v0 = _mm_mul_ps(m1[0], e0);
                __m128 v1 = _mm_mul_ps(m1[1], e1);
                __m128 v2 = _mm_mul_ps(m1[2], e2);
                __m128 v3 = _mm_mul_ps(m1[3], e3);
        
                __m128 a0 = _mm_add_ps(v0, v1);
                __m128 a1 = _mm_add_ps(v2, v3);
                __m128 a2 = _mm_add_ps(a0, a1);
        
                dst1 = a2;
        }
    
        {
                __m128 e0 = _mm_shuffle_ps(m2[2], m2[2], _MM_SHUFFLE(0, 0, 0, 0));
                __m128 e1 = _mm_shuffle_ps(m2[2], m2[2], _MM_SHUFFLE(1, 1, 1, 1));
                __m128 e2 = _mm_shuffle_ps(m2[2], m2[2], _MM_SHUFFLE(2, 2, 2, 2));
                __m128 e3 = _mm_shuffle_ps(m2[2], m2[2], _MM_SHUFFLE(3, 3, 3, 3));
        
                __m128 v0 = _mm_mul_ps(m1[0], e0);
                __m128 v1 = _mm_mul_ps(m1[1], e1);
                __m128 v2 = _mm_mul_ps(m1[2], e2);
                __m128 v3 = _mm_mul_ps(m1[3], e3);
        
                __m128 a0 = _mm_add_ps(v0, v1);
                __m128 a1 = _mm_add_ps(v2, v3);
                __m128 a2 = _mm_add_ps(a0, a1);
        
                dst2 = a2;
        }
    
        {
                __m128 e0 = _mm_shuffle_ps(m2[3], m2[3], _MM_SHUFFLE(0, 0, 0, 0));
                __m128 e1 = _mm_shuffle_ps(m2[3], m2[3], _MM_SHUFFLE(1, 1, 1, 1));
                __m128 e2 = _mm_shuffle_ps(m2[3], m2[3], _MM_SHUFFLE(2, 2, 2, 2));
                __m128 e3 = _mm_shuffle_ps(m2[3], m2[3], _MM_SHUFFLE(3, 3, 3, 3));
        
                __m128 v0 = _mm_mul_ps(m1[0], e0);
                __m128 v1 = _mm_mul_ps(m1[1], e1);
                __m128 v2 = _mm_mul_ps(m1[2], e2);
                __m128 v3 = _mm_mul_ps(m1[3], e3);
        
                __m128 a0 = _mm_add_ps(v0, v1);
                __m128 a1 = _mm_add_ps(v2, v3);
                __m128 a2 = _mm_add_ps(a0, a1);
        
                dst3 = a2;
        }
    dst[0] = dst0;
    dst[1] = dst1;
    dst[2] = dst2;
    dst[3] = dst3;
}

void MathUtil::negateMatrix(const __m128 m[4], __m128 dst[4])
{
    __m128 z = _mm_setzero_ps();
    dst[0] = _mm_sub_ps(z, m[0]);
    dst[1] = _mm_sub_ps(z, m[1]);
    dst[2] = _mm_sub_ps(z, m[2]);
    dst[3] = _mm_sub_ps(z, m[3]);
}

void MathUtil::transposeMatrix(const __m128 m[4], __m128 dst[4])
{
    __m128 tmp0 = _mm_shuffle_ps(m[0], m[1], 0x44);
    __m128 tmp2 = _mm_shuffle_ps(m[0], m[1], 0xEE);
    __m128 tmp1 = _mm_shuffle_ps(m[2], m[3], 0x44);
    __m128 tmp3 = _mm_shuffle_ps(m[2], m[3], 0xEE);
    
    dst[0] = _mm_shuffle_ps(tmp0, tmp1, 0x88);
    dst[1] = _mm_shuffle_ps(tmp0, tmp1, 0xDD);
    dst[2] = _mm_shuffle_ps(tmp2, tmp3, 0x88);
    dst[3] = _mm_shuffle_ps(tmp2, tmp3, 0xDD);
}

void MathUtil::transformVec4(const __m128 m[4], const __m128& v, __m128& dst)
{
    __m128 col1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 col2 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 col3 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 col4 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
    
    dst = _mm_add_ps(
                     _mm_add_ps(_mm_mul_ps(m[0], col1), _mm_mul_ps(m[1], col2)),
                     _mm_add_ps(_mm_mul_ps(m[2], col3), _mm_mul_ps(m[3], col4))
                     );
} 
  
#endif // __SSE__
  
class MathUtilC
{
public:
    inline static void addMatrix(const float* m, float scalar, float* dst);
    
    inline static void addMatrix(const float* m1, const float* m2, float* dst);
    
    inline static void subtractMatrix(const float* m1, const float* m2, float* dst);
    
    inline static void multiplyMatrix(const float* m, float scalar, float* dst);
    
    inline static void multiplyMatrix(const float* m1, const float* m2, float* dst);
    
    inline static void negateMatrix(const float* m, float* dst);
    
    inline static void transposeMatrix(const float* m, float* dst);
    
    inline static void transformVec4(const float* m, float x, float y, float z, float w, float* dst);
    
    inline static void transformVec4(const float* m, const float* v, float* dst);
    
    inline static void crossVec3(const float* v1, const float* v2, float* dst);
};

inline void MathUtilC::addMatrix(const float* m, float scalar, float* dst)
{
    dst[0]  = m[0]  + scalar;
    dst[1]  = m[1]  + scalar;
    dst[2]  = m[2]  + scalar;
    dst[3]  = m[3]  + scalar;
    dst[4]  = m[4]  + scalar;
    dst[5]  = m[5]  + scalar;
    dst[6]  = m[6]  + scalar;
    dst[7]  = m[7]  + scalar;
    dst[8]  = m[8]  + scalar;
    dst[9]  = m[9]  + scalar;
    dst[10] = m[10] + scalar;
    dst[11] = m[11] + scalar;
    dst[12] = m[12] + scalar;
    dst[13] = m[13] + scalar;
    dst[14] = m[14] + scalar;
    dst[15] = m[15] + scalar;
}

inline void MathUtilC::addMatrix(const float* m1, const float* m2, float* dst)
{
    dst[0]  = m1[0]  + m2[0];
    dst[1]  = m1[1]  + m2[1];
    dst[2]  = m1[2]  + m2[2];
    dst[3]  = m1[3]  + m2[3];
    dst[4]  = m1[4]  + m2[4];
    dst[5]  = m1[5]  + m2[5];
    dst[6]  = m1[6]  + m2[6];
    dst[7]  = m1[7]  + m2[7];
    dst[8]  = m1[8]  + m2[8];
    dst[9]  = m1[9]  + m2[9];
    dst[10] = m1[10] + m2[10];
    dst[11] = m1[11] + m2[11];
    dst[12] = m1[12] + m2[12];
    dst[13] = m1[13] + m2[13];
    dst[14] = m1[14] + m2[14];
    dst[15] = m1[15] + m2[15];
}

inline void MathUtilC::subtractMatrix(const float* m1, const float* m2, float* dst)
{
    dst[0]  = m1[0]  - m2[0];
    dst[1]  = m1[1]  - m2[1];
    dst[2]  = m1[2]  - m2[2];
    dst[3]  = m1[3]  - m2[3];
    dst[4]  = m1[4]  - m2[4];
    dst[5]  = m1[5]  - m2[5];
    dst[6]  = m1[6]  - m2[6];
    dst[7]  = m1[7]  - m2[7];
    dst[8]  = m1[8]  - m2[8];
    dst[9]  = m1[9]  - m2[9];
    dst[10] = m1[10] - m2[10];
    dst[11] = m1[11] - m2[11];
    dst[12] = m1[12] - m2[12];
    dst[13] = m1[13] - m2[13];
    dst[14] = m1[14] - m2[14];
    dst[15] = m1[15] - m2[15];
}

inline void MathUtilC::multiplyMatrix(const float* m, float scalar, float* dst)
{
    dst[0]  = m[0]  * scalar;
    dst[1]  = m[1]  * scalar;
    dst[2]  = m[2]  * scalar;
    dst[3]  = m[3]  * scalar;
    dst[4]  = m[4]  * scalar;
    dst[5]  = m[5]  * scalar;
    dst[6]  = m[6]  * scalar;
    dst[7]  = m[7]  * scalar;
    dst[8]  = m[8]  * scalar;
    dst[9]  = m[9]  * scalar;
    dst[10] = m[10] * scalar;
    dst[11] = m[11] * scalar;
    dst[12] = m[12] * scalar;
    dst[13] = m[13] * scalar;
    dst[14] = m[14] * scalar;
    dst[15] = m[15] * scalar;
}

inline void MathUtilC::multiplyMatrix(const float* m1, const float* m2, float* dst)
{
    // Support the case where m1 or m2 is the same array as dst.
    float product[16];
    
    product[0]  = m1[0] * m2[0]  + m1[4] * m2[1] + m1[8]   * m2[2]  + m1[12] * m2[3];
    product[1]  = m1[1] * m2[0]  + m1[5] * m2[1] + m1[9]   * m2[2]  + m1[13] * m2[3];
    product[2]  = m1[2] * m2[0]  + m1[6] * m2[1] + m1[10]  * m2[2]  + m1[14] * m2[3];
    product[3]  = m1[3] * m2[0]  + m1[7] * m2[1] + m1[11]  * m2[2]  + m1[15] * m2[3];
    
    product[4]  = m1[0] * m2[4]  + m1[4] * m2[5] + m1[8]   * m2[6]  + m1[12] * m2[7];
    product[5]  = m1[1] * m2[4]  + m1[5] * m2[5] + m1[9]   * m2[6]  + m1[13] * m2[7];
    product[6]  = m1[2] * m2[4]  + m1[6] * m2[5] + m1[10]  * m2[6]  + m1[14] * m2[7];
    product[7]  = m1[3] * m2[4]  + m1[7] * m2[5] + m1[11]  * m2[6]  + m1[15] * m2[7];
    
    product[8]  = m1[0] * m2[8]  + m1[4] * m2[9] + m1[8]   * m2[10] + m1[12] * m2[11];
    product[9]  = m1[1] * m2[8]  + m1[5] * m2[9] + m1[9]   * m2[10] + m1[13] * m2[11];
    product[10] = m1[2] * m2[8]  + m1[6] * m2[9] + m1[10]  * m2[10] + m1[14] * m2[11];
    product[11] = m1[3] * m2[8]  + m1[7] * m2[9] + m1[11]  * m2[10] + m1[15] * m2[11];
    
    product[12] = m1[0] * m2[12] + m1[4] * m2[13] + m1[8]  * m2[14] + m1[12] * m2[15];
    product[13] = m1[1] * m2[12] + m1[5] * m2[13] + m1[9]  * m2[14] + m1[13] * m2[15];
    product[14] = m1[2] * m2[12] + m1[6] * m2[13] + m1[10] * m2[14] + m1[14] * m2[15];
    product[15] = m1[3] * m2[12] + m1[7] * m2[13] + m1[11] * m2[14] + m1[15] * m2[15];
    
    memcpy(dst, product, MATRIX_SIZE);
}

inline void MathUtilC::negateMatrix(const float* m, float* dst)
{
    dst[0]  = -m[0];
    dst[1]  = -m[1];
    dst[2]  = -m[2];
    dst[3]  = -m[3];
    dst[4]  = -m[4];
    dst[5]  = -m[5];
    dst[6]  = -m[6];
    dst[7]  = -m[7];
    dst[8]  = -m[8];
    dst[9]  = -m[9];
    dst[10] = -m[10];
    dst[11] = -m[11];
    dst[12] = -m[12];
    dst[13] = -m[13];
    dst[14] = -m[14];
    dst[15] = -m[15];
}

inline void MathUtilC::transposeMatrix(const float* m, float* dst)
{
    float t[16] = {
        m[0], m[4], m[8], m[12],
        m[1], m[5], m[9], m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15]
    };
    memcpy(dst, t, MATRIX_SIZE);
}

inline void MathUtilC::transformVec4(const float* m, float x, float y, float z, float w, float* dst)
{
    dst[0] = x * m[0] + y * m[4] + z * m[8] + w * m[12];
    dst[1] = x * m[1] + y * m[5] + z * m[9] + w * m[13];
    dst[2] = x * m[2] + y * m[6] + z * m[10] + w * m[14];
}

inline void MathUtilC::transformVec4(const float* m, const float* v, float* dst)
{
    // Handle case where v == dst.
    float x = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
    float y = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
    float z = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
    float w = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
    
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
    dst[3] = w;
}

inline void MathUtilC::crossVec3(const float* v1, const float* v2, float* dst)
{
    float x = (v1[1] * v2[2]) - (v1[2] * v2[1]);
    float y = (v1[2] * v2[0]) - (v1[0] * v2[2]);
    float z = (v1[0] * v2[1]) - (v1[1] * v2[0]);
    
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
}  
  
void MathUtil::smooth(float* x, float target, float elapsedTime, float responseTime)
{
    ASSERT(x);
    
    if (elapsedTime > 0)
    {
        *x += (target - *x) * elapsedTime / (elapsedTime + responseTime);
    }
}

void MathUtil::smooth(float* x, float target, float elapsedTime, float riseTime, float fallTime)
{
    ASSERT(x);
    
    if (elapsedTime > 0)
    {
        float delta = target - *x;
        *x += delta * elapsedTime / (elapsedTime + (delta > 0 ? riseTime : fallTime));
    }
}

float MathUtil::lerp(float from, float to, float alpha)
{
    return from * (1.0f - alpha) + to * alpha;
}

bool MathUtil::isNeon32Enabled()
{
    return false;
}

bool MathUtil::isNeon64Enabled()
{
    return false;
}

void MathUtil::addMatrix(const float* m, float scalar, float* dst)
{
    MathUtilC::addMatrix(m, scalar, dst);
}

void MathUtil::addMatrix(const float* m1, const float* m2, float* dst)
{
    MathUtilC::addMatrix(m1, m2, dst);
}

void MathUtil::subtractMatrix(const float* m1, const float* m2, float* dst)
{
    MathUtilC::subtractMatrix(m1, m2, dst);
}

void MathUtil::multiplyMatrix(const float* m, float scalar, float* dst)
{
    MathUtilC::multiplyMatrix(m, scalar, dst);
}

void MathUtil::multiplyMatrix(const float* m1, const float* m2, float* dst)
{
    MathUtilC::multiplyMatrix(m1, m2, dst);
}

void MathUtil::negateMatrix(const float* m, float* dst)
{
    MathUtilC::negateMatrix(m, dst);
}

void MathUtil::transposeMatrix(const float* m, float* dst)
{
    MathUtilC::transposeMatrix(m, dst);
}

void MathUtil::transformVec4(const float* m, float x, float y, float z, float w, float* dst)
{
    MathUtilC::transformVec4(m, x, y, z, w, dst);
}

void MathUtil::transformVec4(const float* m, const float* v, float* dst)
{
    MathUtilC::transformVec4(m, v, dst);
}

void MathUtil::crossVec3(const float* v1, const float* v2, float* dst)
{
    MathUtilC::crossVec3(v1, v2, dst);
}

} // namespace mycocos2d
} // namespace bubblefs