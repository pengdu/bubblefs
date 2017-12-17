/****************************************************************************
Copyright (c) 2010-2012 cocos2d-x.org
Copyright (c) 2013-2017 Chukong Technologies
 
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

// cocos2d-x/cocos/platform/CCPlatformMacros.h
// cocos2d-x/cocos/base/ccMacros.h

#ifndef BUBBEFS_PLATFORM_COCOS2D_MACROS_H_
#define BUBBEFS_PLATFORM_COCOS2D_MACROS_H_

#include <assert.h>
#include <float.h>
#include "platform/macros.h"

/**
 * Define some platform specific macros.
 */

/** @def CREATE_FUNC(__TYPE__)
 * Define a create function for a specific type, such as Layer.
 *
 * @param __TYPE__  class type to add create(), such as Layer.
 */
#define CC_CREATE_FUNC(__TYPE__) \
static __TYPE__* create() \
{ \
    __TYPE__ *pRet = new(std::nothrow) __TYPE__(); \
    if (pRet && pRet->init()) \
    { \
        pRet->autorelease(); \
        return pRet; \
    } \
    else \
    { \
        delete pRet; \
        pRet = nullptr; \
        return nullptr; \
    } \
}

/** @def CC_PROPERTY_READONLY 
 * It is used to declare a protected variable. We can use getter to read the variable.
 * 
 * @param varType     the type of variable.
 * @param varName     variable name.
 * @param funName     "get + funName" will be the name of the getter.
 * @warning   The getter is a public virtual function, you should rewrite it first.
 *            The variables and methods declared after CC_PROPERTY_READONLY are all public.
 *            If you need protected or private, please declare.
 */
#define CC_PROPERTY_READONLY(varType, varName, funName)\
protected: varType varName; public: virtual varType get##funName(void) const;

#define CC_PROPERTY_READONLY_PASS_BY_REF(varType, varName, funName)\
protected: varType varName; public: virtual const varType& get##funName(void) const;

/** @def CC_PROPERTY 
 * It is used to declare a protected variable.
 * We can use getter to read the variable, and use the setter to change the variable.
 *
 * @param varType     The type of variable.
 * @param varName     Variable name.
 * @param funName     "get + funName" will be the name of the getter.
 *                    "set + funName" will be the name of the setter.
 * @warning   The getter and setter are public virtual functions, you should rewrite them first.
 *            The variables and methods declared after CC_PROPERTY are all public.
 *            If you need protected or private, please declare.
 */
#define CC_PROPERTY(varType, varName, funName)\
protected: varType varName; public: virtual varType get##funName(void) const; virtual void set##funName(varType var);

#define CC_PROPERTY_PASS_BY_REF(varType, varName, funName)\
protected: varType varName; public: virtual const varType& get##funName(void) const; virtual void set##funName(const varType& var);

/** @def CC_SYNTHESIZE_READONLY 
 * It is used to declare a protected variable. We can use getter to read the variable.
 *
 * @param varType     The type of variable.
 * @param varName     Variable name.
 * @param funName     "get + funName" will be the name of the getter.
 * @warning   The getter is a public inline function.
 *            The variables and methods declared after CC_SYNTHESIZE_READONLY are all public.
 *            If you need protected or private, please declare.
 */
#define CC_SYNTHESIZE_READONLY(varType, varName, funName)\
protected: varType varName; public: virtual inline varType get##funName(void) const { return varName; }

#define CC_SYNTHESIZE_READONLY_PASS_BY_REF(varType, varName, funName)\
protected: varType varName; public: virtual inline const varType& get##funName(void) const { return varName; }

/** @def CC_SYNTHESIZE 
 * It is used to declare a protected variable.
 * We can use getter to read the variable, and use the setter to change the variable.
 *
 * @param varType     The type of variable.
 * @param varName     Variable name.
 * @param funName     "get + funName" will be the name of the getter.
 *                    "set + funName" will be the name of the setter.
 * @warning   The getter and setter are public inline functions.
 *            The variables and methods declared after CC_SYNTHESIZE are all public.
 *            If you need protected or private, please declare.
 */
#define CC_SYNTHESIZE(varType, varName, funName)\
protected: varType varName; public: virtual inline varType get##funName(void) const { return varName; } virtual inline void set##funName(varType var){ varName = var; }

#define CC_SYNTHESIZE_PASS_BY_REF(varType, varName, funName)\
protected: varType varName; public: virtual inline const varType& get##funName(void) const { return varName; } virtual inline void set##funName(const varType& var){ varName = var; }

/** @def CC_DISALLOW_COPY_AND_ASSIGN(TypeName)
 * A macro to disallow the copy constructor and operator= functions.
 * This should be used in the private: declarations for a class
 */
#if defined(__GNUC__) && ((__GNUC__ >= 5) || ((__GNUG__ == 4) && (__GNUC_MINOR__ >= 4))) \
    || (defined(__clang__) && (__clang_major__ >= 3)) || (_MSC_VER >= 1800)
#define CC_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete; \
    TypeName &operator =(const TypeName &) = delete;
#else
#define CC_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &); \
    TypeName &operator =(const TypeName &);
#endif

/** @def CC_DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName)
 * A macro to disallow all the implicit constructors, namely the
 * default constructor, copy constructor and operator= functions.
 *
 * This should be used in the private: declarations for a class
 * that wants to prevent anyone from instantiating it. This is
 * especially useful for classes containing only static methods. 
 */
#define CC_DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName)    \
    TypeName();                                        \
    CC_DISALLOW_COPY_AND_ASSIGN(TypeName)

#define CC_SAFE_DELETE(p)           do { delete (p); (p) = nullptr; } while(0)
#define CC_SAFE_DELETE_ARRAY(p)     do { if(p) { delete[] (p); (p) = nullptr; } } while(0)
#define CC_SAFE_FREE(p)             do { if(p) { free(p); (p) = nullptr; } } while(0)
#define CC_SAFE_RELEASE(p)          do { if(p) { (p)->release(); } } while(0)
#define CC_SAFE_RELEASE_NULL(p)     do { if(p) { (p)->release(); (p) = nullptr; } } while(0)
#define CC_SAFE_RETAIN(p)           do { if(p) { (p)->retain(); } } while(0)
#define CC_BREAK_IF(cond)           if(cond) break

#define CCLOG(...)       do {} while (0)
#define CCLOGINFO(...)   do {} while (0)
#define CCLOGERROR(...)  do {} while (0)
#define CCLOGWARN(...)   do {} while (0)
#define CCASSERT(cond, msg)
#define CC_ASSERT(cond)  assert(cond)

/** @def CC_SWAP
simple macro that swaps 2 variables
 @deprecated use std::swap() instead
*/
#define CC_SWAP(x, y, type)    \
{    type temp = (x);        \
    x = y; y = temp;        \
}

#ifndef FLT_EPSILON
#define FLT_EPSILON     1.192092896e-07F
#endif // FLT_EPSILON

#ifndef DBL_EPSILON
#define DBL_EPSILON  2.2204460492503131e-016
#endif

/** @def CC_DEGREES_TO_RADIANS
 converts degrees to radians
 */
#define CC_DEGREES_TO_RADIANS(__ANGLE__) ((__ANGLE__) * 0.01745329252f) // PI / 180

/** @def CC_RADIANS_TO_DEGREES
 converts radians to degrees
 */
#define CC_RADIANS_TO_DEGREES(__ANGLE__) ((__ANGLE__) * 57.29577951f) // PI * 180

#define CC_REPEAT_FOREVER (UINT_MAX -1)

/**
Helper macros which converts 4-byte little/big endian 
integral number to the machine native number representation
 
It should work same as apples CFSwapInt32LittleToHost(..)
*/

/// when define returns true it means that our architecture uses big endian
#define CC_HOST_IS_BIG_ENDIAN (bool)(*(unsigned short *)"\0\xff" < 0x100) 
#define CC_SWAP32(i)  ((i & 0x000000ff) << 24 | (i & 0x0000ff00) << 8 | (i & 0x00ff0000) >> 8 | (i & 0xff000000) >> 24)
#define CC_SWAP16(i)  ((i & 0x00ff) << 8 | (i &0xff00) >> 8)   
#define CC_SWAP_INT32_LITTLE_TO_HOST(i) ((CC_HOST_IS_BIG_ENDIAN == true)? CC_SWAP32(i) : (i) )
#define CC_SWAP_INT16_LITTLE_TO_HOST(i) ((CC_HOST_IS_BIG_ENDIAN == true)? CC_SWAP16(i) : (i) )
#define CC_SWAP_INT32_BIG_TO_HOST(i)    ((CC_HOST_IS_BIG_ENDIAN == true)? (i) : CC_SWAP32(i) )
#define CC_SWAP_INT16_BIG_TO_HOST(i)    ((CC_HOST_IS_BIG_ENDIAN == true)? (i):  CC_SWAP16(i) )

/*********************************/
 /** 64bits Program Sense Macros **/
 /*********************************/
#if defined(_M_X64) || defined(_WIN64) || defined(__LP64__) || defined(_LP64) || defined(__x86_64)
#define CC_64BITS 1
#else
#define CC_64BITS 0
#endif

/******************************************************************************************/
 /** LittleEndian Sense Macro, from google protobuf see:                                  **/
 /** https://github.com/google/protobuf/blob/master/src/google/protobuf/io/coded_stream.h **/
 /******************************************************************************************/
#ifdef _MSC_VER
  #if defined(_M_IX86)
    #define CC_LITTLE_ENDIAN 1
  #else
    #define CC_LITTLE_ENDIAN 0
  #endif
  #if _MSC_VER >= 1300 && !defined(__INTEL_COMPILER)
    #pragma runtime_checks("c", off)
  #endif
#else
  #include <sys/param.h>
  #include <endian.h>
  #if ((defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)) || \
         (defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN)) 
    #define CC_LITTLE_ENDIAN 1
  #else
    #define CC_LITTLE_ENDIAN 0
  #endif
#endif

// new callbacks based on C++11
#define CC_CALLBACK_0(__selector__,__target__, ...) std::bind(&__selector__,__target__, ##__VA_ARGS__)
#define CC_CALLBACK_1(__selector__,__target__, ...) std::bind(&__selector__,__target__, std::placeholders::_1, ##__VA_ARGS__)
#define CC_CALLBACK_2(__selector__,__target__, ...) std::bind(&__selector__,__target__, std::placeholders::_1, std::placeholders::_2, ##__VA_ARGS__)
#define CC_CALLBACK_3(__selector__,__target__, ...) std::bind(&__selector__,__target__, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ##__VA_ARGS__)

#endif // BUBBEFS_PLATFORM_COCOS2D_MACROS_H_