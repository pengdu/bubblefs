/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the QtCore module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see http://www.qt.io/terms-conditions. For further
** information use the contact form at http://www.qt.io/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 or version 3 as published by the Free
** Software Foundation and appearing in the file LICENSE.LGPLv21 and
** LICENSE.LGPLv3 included in the packaging of this file. Please review the
** following information to ensure the GNU Lesser General Public License
** requirements will be met: https://www.gnu.org/licenses/lgpl.html and
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** As a special exception, The Qt Company gives you certain additional
** rights. These rights are described in The Qt Company LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

// qt/src/corelib/global/qglobal.h

#ifndef BUBBLEFS_PLATFORM_QT_QGLOBAL_H_
#define BUBBLEFS_PLATFORM_QT_QGLOBAL_H_

#include <stddef.h>
#include <utility>

namespace bubblefs {
namespace myqt {
  
/*
   Size-dependent types (architechture-dependent byte order)
   Make sure to update QMetaType when changing these typedefs
*/

typedef signed char qint8;         /* 8 bit signed */
typedef unsigned char quint8;      /* 8 bit unsigned */
typedef short qint16;              /* 16 bit signed */
typedef unsigned short quint16;    /* 16 bit unsigned */
typedef int qint32;                /* 32 bit signed */
typedef unsigned int quint32;      /* 32 bit unsigned */
#  define Q_INT64_C(c) static_cast<long long>(c ## LL)     /* signed 64 bit constant */
#  define Q_UINT64_C(c) static_cast<unsigned long long>(c ## ULL) /* unsigned 64 bit constant */
typedef long long qint64;           /* 64 bit signed */
typedef unsigned long long quint64; /* 64 bit unsigned */

typedef qint64 qlonglong;
typedef quint64 qulonglong;
typedef double qreal;

/*
  quintptr and qptrdiff is guaranteed to be the same size as a pointer, i.e.
      sizeof(void *) == sizeof(quintptr)
      && sizeof(void *) == sizeof(qptrdiff)
*/
template <int> struct QIntegerForSize;
template <>    struct QIntegerForSize<1> { typedef quint8  Unsigned; typedef qint8  Signed; };
template <>    struct QIntegerForSize<2> { typedef quint16 Unsigned; typedef qint16 Signed; };
template <>    struct QIntegerForSize<4> { typedef quint32 Unsigned; typedef qint32 Signed; };
template <>    struct QIntegerForSize<8> { typedef quint64 Unsigned; typedef qint64 Signed; };
template <class T> struct QIntegerForSizeof: QIntegerForSize<sizeof(T)> { };
typedef QIntegerForSizeof<void*>::Unsigned quintptr;
typedef QIntegerForSizeof<void*>::Signed qptrdiff;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

/*
   Utility macros and inline functions
*/

template <typename T>
constexpr inline T qAbs(const T &t) { return t >= 0 ? t : -t; }

constexpr inline int qRound(qreal d)
{ return d >= qreal(0.0) ? int(d + qreal(0.5)) : int(d - int(d-1) + qreal(0.5)) + int(d-1); }

constexpr inline qint64 qRound64(qreal d)
{ return d >= qreal(0.0) ? qint64(d + qreal(0.5)) : qint64(d - qreal(qint64(d-1)) + qreal(0.5)) + qint64(d-1); }

template <typename T>
constexpr inline const T &qMin(const T &a, const T &b) { return (a < b) ? a : b; }
template <typename T>
constexpr inline const T &qMax(const T &a, const T &b) { return (a < b) ? b : a; }
template <typename T>
constexpr inline const T &qBound(const T &min, const T &val, const T &max)
{ return qMax(min, qMin(max, val)); }

#if defined(QT_NO_THREAD)

template <typename T>
class QGlobalStatic
{
public:
    T *pointer;
    inline QGlobalStatic(T *p) : pointer(p) { }
    inline ~QGlobalStatic() { pointer = 0; }
};

#define Q_GLOBAL_STATIC(TYPE, NAME)                                  \
    static TYPE *NAME()                                              \
    {                                                                \
        static TYPE thisVariable;                                    \
        static QGlobalStatic<TYPE > thisGlobalStatic(&thisVariable); \
        return thisGlobalStatic.pointer;                             \
    }

#define Q_GLOBAL_STATIC_WITH_ARGS(TYPE, NAME, ARGS)                  \
    static TYPE *NAME()                                              \
    {                                                                \
        static TYPE thisVariable ARGS;                               \
        static QGlobalStatic<TYPE > thisGlobalStatic(&thisVariable); \
        return thisGlobalStatic.pointer;                             \
    }

#define Q_GLOBAL_STATIC_WITH_INITIALIZER(TYPE, NAME, INITIALIZER)    \
    static TYPE *NAME()                                              \
    {                                                                \
        static TYPE thisVariable;                                    \
        static QGlobalStatic<TYPE > thisGlobalStatic(0);             \
        if (!thisGlobalStatic.pointer) {                             \
            TYPE *x = thisGlobalStatic.pointer = &thisVariable;      \
            INITIALIZER;                                             \
        }                                                            \
        return thisGlobalStatic.pointer;                             \
    }

#else

// forward declaration, since qatomic.h needs qglobal.h
template <typename T> class QBasicAtomicPointer;

// POD for Q_GLOBAL_STATIC
template <typename T>
class QGlobalStatic
{
public:
    QBasicAtomicPointer<T> pointer;
    bool destroyed;
};

// Created as a function-local static to delete a QGlobalStatic<T>
template <typename T>
class QGlobalStaticDeleter
{
public:
    QGlobalStatic<T> &globalStatic;
    QGlobalStaticDeleter(QGlobalStatic<T> &_globalStatic)
        : globalStatic(_globalStatic)
    { }

    inline ~QGlobalStaticDeleter()
    {
        delete globalStatic.pointer;
        globalStatic.pointer = 0;
        globalStatic.destroyed = true;
    }
};

#define Q_GLOBAL_STATIC_INIT(TYPE, NAME)                                      \
        static QGlobalStatic<TYPE > this_ ## NAME                             \
                            = { Q_BASIC_ATOMIC_INITIALIZER(0), false }

#define Q_GLOBAL_STATIC(TYPE, NAME)                                           \
    static TYPE *NAME()                                                       \
    {                                                                         \
        Q_GLOBAL_STATIC_INIT(TYPE, _StaticVar_);                              \
        if (!this__StaticVar_.pointer && !this__StaticVar_.destroyed) {       \
            TYPE *x = new TYPE;                                               \
            if (!this__StaticVar_.pointer.testAndSetOrdered(0, x))            \
                delete x;                                                     \
            else                                                              \
                static QGlobalStaticDeleter<TYPE > cleanup(this__StaticVar_); \
        }                                                                     \
        return this__StaticVar_.pointer;                                      \
    }

#define Q_GLOBAL_STATIC_WITH_ARGS(TYPE, NAME, ARGS)                           \
    static TYPE *NAME()                                                       \
    {                                                                         \
        Q_GLOBAL_STATIC_INIT(TYPE, _StaticVar_);                              \
        if (!this__StaticVar_.pointer && !this__StaticVar_.destroyed) {       \
            TYPE *x = new TYPE ARGS;                                          \
            if (!this__StaticVar_.pointer.testAndSetOrdered(0, x))            \
                delete x;                                                     \
            else                                                              \
                static QGlobalStaticDeleter<TYPE > cleanup(this__StaticVar_); \
        }                                                                     \
        return this__StaticVar_.pointer;                                      \
    }

#define Q_GLOBAL_STATIC_WITH_INITIALIZER(TYPE, NAME, INITIALIZER)             \
    static TYPE *NAME()                                                       \
    {                                                                         \
        Q_GLOBAL_STATIC_INIT(TYPE, _StaticVar_);                              \
        if (!this__StaticVar_.pointer && !this__StaticVar_.destroyed) {       \
            QScopedPointer<TYPE > x(new TYPE);                                \
            INITIALIZER;                                                      \
            if (this__StaticVar_.pointer.testAndSetOrdered(0, x.data())) {    \
                static QGlobalStaticDeleter<TYPE > cleanup(this__StaticVar_); \
                x.take();                                                     \
            }                                                                 \
        }                                                                     \
        return this__StaticVar_.pointer;                                      \
    }

#endif // QGlobalStatic if (QT_NO_THREAD)

class QFlag
{
    int i;
public:
    inline QFlag(int i);
    inline operator int() const { return i; }
};

inline QFlag::QFlag(int ai) : i(ai) {}

template<typename Enum>
class QFlags
{
    typedef void **Zero;
    int i;
public:
    typedef Enum enum_type;
    constexpr inline QFlags(const QFlags &f) : i(f.i) {}
    constexpr inline QFlags(Enum f) : i(f) {}
    constexpr inline QFlags(Zero = 0) : i(0) {}
    inline QFlags(QFlag f) : i(f) {}

    inline QFlags &operator=(const QFlags &f) { i = f.i; return *this; }
    inline QFlags &operator&=(int mask) { i &= mask; return *this; }
    inline QFlags &operator&=(uint mask) { i &= mask; return *this; }
    inline QFlags &operator|=(QFlags f) { i |= f.i; return *this; }
    inline QFlags &operator|=(Enum f) { i |= f; return *this; }
    inline QFlags &operator^=(QFlags f) { i ^= f.i; return *this; }
    inline QFlags &operator^=(Enum f) { i ^= f; return *this; }

    constexpr inline operator int() const { return i; }

    constexpr inline QFlags operator|(QFlags f) const { return QFlags(Enum(i | f.i)); }
    constexpr inline QFlags operator|(Enum f) const { return QFlags(Enum(i | f)); }
    constexpr inline QFlags operator^(QFlags f) const { return QFlags(Enum(i ^ f.i)); }
    constexpr inline QFlags operator^(Enum f) const { return QFlags(Enum(i ^ f)); }
    constexpr inline QFlags operator&(int mask) const { return QFlags(Enum(i & mask)); }
    constexpr inline QFlags operator&(uint mask) const { return QFlags(Enum(i & mask)); }
    constexpr inline QFlags operator&(Enum f) const { return QFlags(Enum(i & f)); }
    constexpr inline QFlags operator~() const { return QFlags(Enum(~i)); }

    constexpr inline bool operator!() const { return !i; }

    inline bool testFlag(Enum f) const { return (i & f) == f && (f != 0 || i == int(f) ); }
};

#if defined(__GNUC__)
/* make use of typeof-extension */
template <typename T>
class QForeachContainer {
public:
    inline QForeachContainer(const T& t) : c(t), brk(0), i(c.begin()), e(c.end()) { }
    const T c;
    int brk;
    typename T::const_iterator i, e;
};

#define Q_FOREACH(variable, container)                                \
for (QForeachContainer<__typeof__(container)> _container_(container); \
     !_container_.brk && _container_.i != _container_.e;              \
     __extension__  ({ ++_container_.brk; ++_container_.i; }))                       \
    for (variable = *_container_.i;; __extension__ ({--_container_.brk; break;}))
#endif // Q_FOREACH
      
class QBool
{
    bool b;

public:
    inline explicit QBool(bool B) : b(B) {}
    inline operator const void *() const
    { return b ? static_cast<const void *>(this) : static_cast<const void *>(0); }
};

inline bool operator==(QBool b1, bool b2) { return !b1 == !b2; }
inline bool operator==(bool b1, QBool b2) { return !b1 == !b2; }
inline bool operator==(QBool b1, QBool b2) { return !b1 == !b2; }
inline bool operator!=(QBool b1, bool b2) { return !b1 != !b2; }
inline bool operator!=(bool b1, QBool b2) { return !b1 != !b2; }
inline bool operator!=(QBool b1, QBool b2) { return !b1 != !b2; }

constexpr static inline bool qFuzzyCompare(double p1, double p2)
{
    return (qAbs(p1 - p2) <= 0.000000000001 * qMin(qAbs(p1), qAbs(p2)));
}

constexpr static inline bool qFuzzyCompare(float p1, float p2)
{
    return (qAbs(p1 - p2) <= 0.00001f * qMin(qAbs(p1), qAbs(p2)));
}

/*!
  \internal
*/
constexpr static inline bool qFuzzyIsNull(double d)
{
    return qAbs(d) <= 0.000000000001;
}

/*!
  \internal
*/
constexpr static inline bool qFuzzyIsNull(float f)
{
    return qAbs(f) <= 0.00001f;
}

/*
   This function tests a double for a null value. It doesn't
   check whether the actual value is 0 or close to 0, but whether
   it is binary 0.
*/
static inline bool qIsNull(double d)
{
    union U {
        double d;
        quint64 u;
    };
    U val;
    val.d = d;
    return val.u == quint64(0);
}

/*
   This function tests a float for a null value. It doesn't
   check whether the actual value is 0 or close to 0, but whether
   it is binary 0.
*/
static inline bool qIsNull(float f)
{
    union U {
        float f;
        quint32 u;
    };
    U val;
    val.f = f;
    return val.u == 0u;
}

template <typename T>
inline void qSwap(T &value1, T &value2)
{
#ifdef QT_NO_STL
    const T t = value1;
    value1 = value2;
    value2 = t;
#else
    std::swap(value1, value2);
#endif
}

} // namespace myqt
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_QT_QGLOBAL_H_