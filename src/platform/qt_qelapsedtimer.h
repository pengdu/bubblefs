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

// qt/src/corelib/tools/qelapsedtimer.h

#ifndef BUBBLEFS_PLATFORM_QT_QTIMESTAMP_H_
#define BUBBLEFS_PLATFORM_QT_QTIMESTAMP_H_

#include "platform/qt_qglobal.h"

namespace bubblefs {
namespace myqt {
  
class QElapsedTimer
{
public:
    enum ClockType {
        SystemTime,
        MonotonicClock,
        TickCounter,
        MachAbsoluteTime,
        PerformanceCounter
    };
    static ClockType clockType();
    static bool isMonotonic();

    void start();
    qint64 restart();
    void invalidate();
    bool isValid() const;

    qint64 nsecsElapsed() const;
    qint64 elapsed() const;
    bool hasExpired(qint64 timeout) const;

    qint64 msecsSinceReference() const;
    qint64 msecsTo(const QElapsedTimer &other) const;
    qint64 secsTo(const QElapsedTimer &other) const;

    bool operator==(const QElapsedTimer &other) const
    { return t1 == other.t1 && t2 == other.t2; }
    bool operator!=(const QElapsedTimer &other) const
    { return !(*this == other); }

    friend bool operator<(const QElapsedTimer &v1, const QElapsedTimer &v2);

private:
    qint64 t1;
    qint64 t2;
};

} // namespace myqt
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_QT_QTIMESTAMP_H_