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

// qt/src/corelib/tools/qelapsedtimer.cpp
// qt/src/corelib/tools/qelapsedtimer_unix.cpp

/*!
    \class QElapsedTimer
    \brief The QElapsedTimer class provides a fast way to calculate elapsed times.
    \since 4.7
    \reentrant
    \ingroup tools
    \inmodule QtCore
    The QElapsedTimer class is usually used to quickly calculate how much
    time has elapsed between two events. Its API is similar to that of QTime,
    so code that was using that can be ported quickly to the new class.
    However, unlike QTime, QElapsedTimer tries to use monotonic clocks if
    possible. This means it's not possible to convert QElapsedTimer objects
    to a human-readable time.
    The typical use-case for the class is to determine how much time was
    spent in a slow operation. The simplest example of such a case is for
    debugging purposes, as in the following example:
    \snippet doc/src/snippets/qelapsedtimer/main.cpp 0
    In this example, the timer is started by a call to start() and the
    elapsed timer is calculated by the elapsed() function.
    The time elapsed can also be used to recalculate the time available for
    another operation, after the first one is complete. This is useful when
    the execution must complete within a certain time period, but several
    steps are needed. The \tt{waitFor}-type functions in QIODevice and its
    subclasses are good examples of such need. In that case, the code could
    be as follows:
    \snippet doc/src/snippets/qelapsedtimer/main.cpp 1
    Another use-case is to execute a certain operation for a specific
    timeslice. For this, QElapsedTimer provides the hasExpired() convenience
    function, which can be used to determine if a certain number of
    milliseconds has already elapsed:
    \snippet doc/src/snippets/qelapsedtimer/main.cpp 2
    \section1 Reference clocks
    QElapsedTimer will use the platform's monotonic reference clock in all
    platforms that support it (see QElapsedTimer::isMonotonic()). This has
    the added benefit that QElapsedTimer is immune to time adjustments, such
    as the user correcting the time. Also unlike QTime, QElapsedTimer is
    immune to changes in the timezone settings, such as daylight savings
    periods.
    On the other hand, this means QElapsedTimer values can only be compared
    with other values that use the same reference. This is especially true if
    the time since the reference is extracted from the QElapsedTimer object
    (QElapsedTimer::msecsSinceReference()) and serialised. These values
    should never be exchanged across the network or saved to disk, since
    there's no telling whether the computer node receiving the data is the
    same as the one originating it or if it has rebooted since.
    It is, however, possible to exchange the value with other processes
    running on the same machine, provided that they also use the same
    reference clock. QElapsedTimer will always use the same clock, so it's
    safe to compare with the value coming from another process in the same
    machine. If comparing to values produced by other APIs, you should check
    that the clock used is the same as QElapsedTimer (see
    QElapsedTimer::clockType()).
    \section2 32-bit overflows
    Some of the clocks that QElapsedTimer have a limited range and may
    overflow after hitting the upper limit (usually 32-bit). QElapsedTimer
    deals with this overflow issue and presents a consistent timing. However,
    when extracting the time since reference from QElapsedTimer, two
    different processes in the same machine may have different understanding
    of how much time has actually elapsed.
    The information on which clocks types may overflow and how to remedy that
    issue is documented along with the clock types.
    \sa QTime, QTimer
*/

/*!
    \enum QElapsedTimer::ClockType
    This enum contains the different clock types that QElapsedTimer may use.
    QElapsedTimer will always use the same clock type in a particular
    machine, so this value will not change during the lifetime of a program.
    It is provided so that QElapsedTimer can be used with other non-Qt
    implementations, to guarantee that the same reference clock is being
    used.
    \value SystemTime         The human-readable system time. This clock is not monotonic.
    \value MonotonicClock     The system's monotonic clock, usually found in Unix systems. This clock is monotonic and does not overflow.
    \value TickCounter        The system's tick counter, used on Windows and Symbian systems. This clock may overflow.
    \value MachAbsoluteTime   The Mach kernel's absolute time (Mac OS X). This clock is monotonic and does not overflow.
    \value PerformanceCounter The high-resolution performance counter provided by Windows. This clock is monotonic and does not overflow.
    \section2 SystemTime
    The system time clock is purely the real time, expressed in milliseconds
    since Jan 1, 1970 at 0:00 UTC. It's equivalent to the value returned by
    the C and POSIX \tt{time} function, with the milliseconds added. This
    clock type is currently only used on Unix systems that do not support
    monotonic clocks (see below).
    This is the only non-monotonic clock that QElapsedTimer may use.
    \section2 MonotonicClock
    This is the system's monotonic clock, expressed in milliseconds since an
    arbitrary point in the past. This clock type is used on Unix systems
    which support POSIX monotonic clocks (\tt{_POSIX_MONOTONIC_CLOCK}).
    This clock does not overflow.
    \section2 TickCounter
    The tick counter clock type is based on the system's or the processor's
    tick counter, multiplied by the duration of a tick. This clock type is
    used on Windows and Symbian platforms. If the high-precision performance
    counter is available on Windows, the \tt{PerformanceCounter} clock type
    is used instead.
    The TickCounter clock type is the only clock type that may overflow.
    Windows Vista and Windows Server 2008 support the extended 64-bit tick
    counter, which allows avoiding the overflow.
    On Windows systems, the clock overflows after 2^32 milliseconds, which
    corresponds to roughly 49.7 days. This means two processes's reckoning of
    the time since the reference may be different by multiples of 2^32
    milliseconds. When comparing such values, it's recommended that the high
    32 bits of the millisecond count be masked off.
    On Symbian systems, the overflow happens after 2^32 ticks, the duration
    of which can be obtained from the platform HAL using the constant
    HAL::ENanoTickPeriod. When comparing values between processes, it's
    necessary to divide the value by the tick duration and mask off the high
    32 bits.
    \section2 MachAbsoluteTime
    This clock type is based on the absolute time presented by Mach kernels,
    such as that found on Mac OS X. This clock type is presented separately
    from MonotonicClock since Mac OS X is also a Unix system and may support
    a POSIX monotonic clock with values differing from the Mach absolute
    time.
    This clock is monotonic and does not overflow.
    \section2 PerformanceCounter
    This clock uses the Windows functions \tt{QueryPerformanceCounter} and
    \tt{QueryPerformanceFrequency} to access the system's high-precision
    performance counter. Since this counter may not be available on all
    systems, QElapsedTimer will fall back to the \tt{TickCounter} clock
    automatically, if this clock cannot be used.
    This clock is monotonic and does not overflow.
    \sa clockType(), isMonotonic()
*/

/*!
    \fn bool QElapsedTimer::operator ==(const QElapsedTimer &other) const
    Returns true if this object and \a other contain the same time.
*/

/*!
    \fn bool QElapsedTimer::operator !=(const QElapsedTimer &other) const
    Returns true if this object and \a other contain different times.
*/

#include "platform/qt_qelapsedtimer.h"
#include <sys/select.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if defined(QT_NO_CLOCK_MONOTONIC) || defined(QT_BOOTSTRAPPED)
// turn off the monotonic clock
# ifdef _POSIX_MONOTONIC_CLOCK
#  undef _POSIX_MONOTONIC_CLOCK
# endif
# define _POSIX_MONOTONIC_CLOCK -1
#endif

namespace bubblefs {
namespace myqt {
  
#if (_POSIX_MONOTONIC_CLOCK-0 != 0)
static const bool monotonicClockChecked = true;
static const bool monotonicClockAvailable = _POSIX_MONOTONIC_CLOCK > 0;
#else
static int monotonicClockChecked = false;
static int monotonicClockAvailable = false;
#endif

#ifdef __GNUC__
# define is_likely(x) __builtin_expect((x), 1)
#else
# define is_likely(x) (x)
#endif
#define load_acquire(x) ((volatile const int&)(x))
#define store_release(x,v) ((volatile int&)(x) = (v))

static const qint64 invalidData = Q_INT64_C(0x8000000000000000);

/*!
    Marks this QElapsedTimer object as invalid.
    An invalid object can be checked with isValid(). Calculations of timer
    elapsed since invalid data are undefined and will likely produce bizarre
    results.
    \sa isValid(), start(), restart()
*/
void QElapsedTimer::invalidate()
{
     t1 = t2 = invalidData;
}

/*!
    Returns false if this object was invalidated by a call to invalidate() and
    has not been restarted since.
    \sa invalidate(), start(), restart()
*/
bool QElapsedTimer::isValid() const
{
    return t1 != invalidData && t2 != invalidData;
}

/*!
    Returns true if this QElapsedTimer has already expired by \a timeout
    milliseconds (that is, more than \a timeout milliseconds have elapsed).
    The value of \a timeout can be -1 to indicate that this timer does not
    expire, in which case this function will always return false.
    \sa elapsed()
*/
bool QElapsedTimer::hasExpired(qint64 timeout) const
{
    // if timeout is -1, quint64(timeout) is LLINT_MAX, so this will be
    // considered as never expired
    return quint64(elapsed()) > quint64(timeout);
}

static void unixCheckClockType()
{
#if (_POSIX_MONOTONIC_CLOCK-0 == 0)
    if (is_likely(load_acquire(monotonicClockChecked)))
        return;

# if defined(_SC_MONOTONIC_CLOCK)
    // detect if the system support monotonic timers
    long x = sysconf(_SC_MONOTONIC_CLOCK);
    store_release(monotonicClockAvailable, x >= 200112L);
# endif

    store_release(monotonicClockChecked, true);
#endif
}

static inline qint64 fractionAdjustment()
{
    // disabled, but otherwise indicates bad usage of QElapsedTimer
    //Q_ASSERT(monotonicClockChecked);

    if (monotonicClockAvailable) {
        // the monotonic timer is measured in nanoseconds
        // 1 ms = 1000000 ns
        return 1000*1000ull;
    } else {
        // gettimeofday is measured in microseconds
        // 1 ms = 1000 us
        return 1000;
    }
}

bool QElapsedTimer::isMonotonic()
{
    unixCheckClockType();
    return monotonicClockAvailable;
}

QElapsedTimer::ClockType QElapsedTimer::clockType()
{
    unixCheckClockType();
    return monotonicClockAvailable ? MonotonicClock : SystemTime;
}

static inline void do_gettime(qint64 *sec, qint64 *frac)
{
#if (_POSIX_MONOTONIC_CLOCK-0 >= 0)
    unixCheckClockType();
    if (is_likely(monotonicClockAvailable)) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        *sec = ts.tv_sec;
        *frac = ts.tv_nsec;
        return;
    }
#endif
    // use gettimeofday
    struct timeval tv;
    ::gettimeofday(&tv, 0);
    *sec = tv.tv_sec;
    *frac = tv.tv_usec;
}

// used in qcore_unix.cpp and qeventdispatcher_unix.cpp
timeval qt_gettime()
{
    qint64 sec, frac;
    do_gettime(&sec, &frac);

    timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = frac;
    if (monotonicClockAvailable)
        tv.tv_usec /= 1000;

    return tv;
}

static qint64 elapsedAndRestart(qint64 sec, qint64 frac,
                                qint64 *nowsec, qint64 *nowfrac)
{
    do_gettime(nowsec, nowfrac);
    sec = *nowsec - sec;
    frac = *nowfrac - frac;
    return sec * Q_INT64_C(1000) + frac / fractionAdjustment();
}

void QElapsedTimer::start()
{
    do_gettime(&t1, &t2);
}

qint64 QElapsedTimer::restart()
{
    return elapsedAndRestart(t1, t2, &t1, &t2);
}

qint64 QElapsedTimer::nsecsElapsed() const
{
    qint64 sec, frac;
    do_gettime(&sec, &frac);
    sec = sec - t1;
    frac = frac - t2;
    if (!monotonicClockAvailable)
        frac *= 1000;
    return sec * Q_INT64_C(1000000000) + frac;
}

qint64 QElapsedTimer::elapsed() const
{
    qint64 sec, frac;
    return elapsedAndRestart(t1, t2, &sec, &frac);
}

qint64 QElapsedTimer::msecsSinceReference() const
{
    return t1 * Q_INT64_C(1000) + t2 / fractionAdjustment();
}

qint64 QElapsedTimer::msecsTo(const QElapsedTimer &other) const
{
    qint64 secs = other.t1 - t1;
    qint64 fraction = other.t2 - t2;
    return secs * Q_INT64_C(1000) + fraction / fractionAdjustment();
}

qint64 QElapsedTimer::secsTo(const QElapsedTimer &other) const
{
    return other.t1 - t1;
}

bool operator<(const QElapsedTimer &v1, const QElapsedTimer &v2)
{
    return v1.t1 < v2.t1 || (v1.t1 == v2.t1 && v1.t2 < v2.t2);
}

} // namespace myqt
} // namespace bubblefs