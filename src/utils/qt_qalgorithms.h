/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the documentation of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:FDL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see http://www.qt.io/terms-conditions. For further
** information use the contact form at http://www.qt.io/contact-us.
**
** GNU Free Documentation License Usage
** Alternatively, this file may be used under the terms of the GNU Free
** Documentation License version 1.3 as published by the Free Software
** Foundation and appearing in the file included in the packaging of
** this file.  Please review the following information to ensure
** the GNU Free Documentation License version 1.3 requirements
** will be met: http://www.gnu.org/copyleft/fdl.html.
** $QT_END_LICENSE$
**
****************************************************************************/

// qt/src/corelib/tools/qalgorithms.h

/*!
    \headerfile <QtAlgorithms>
    \title Generic Algorithms
    \ingroup funclists

    \brief The <QtAlgorithms> header includes the generic, template-based algorithms.

    Qt provides a number of global template functions in \c
    <QtAlgorithms> that work on containers and perform well-know
    algorithms. You can use these algorithms with any \l {container
    class} that provides STL-style iterators, including Qt's QList,
    QLinkedList, QVector, QMap, and QHash classes.

    These functions have taken their inspiration from similar
    functions available in the STL \c <algorithm> header. Most of them
    have a direct STL equivalent; for example, qCopyBackward() is the
    same as STL's copy_backward() algorithm.

    If STL is available on all your target platforms, you can use the
    STL algorithms instead of their Qt counterparts. One reason why
    you might want to use the STL algorithms is that STL provides
    dozens and dozens of algorithms, whereas Qt only provides the most
    important ones, making no attempt to duplicate functionality that
    is already provided by the C++ standard.

    Most algorithms take \l {STL-style iterators} as parameters. The
    algorithms are generic in the sense that they aren't bound to a
    specific iterator class; you can use them with any iterators that
    meet a certain set of requirements.

    Let's take the qFill() algorithm as an example. Unlike QVector,
    QList has no fill() function that can be used to fill a list with
    a particular value. If you need that functionality, you can use
    qFill():

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 0

    qFill() takes a begin iterator, an end iterator, and a value.
    In the example above, we pass \c list.begin() and \c list.end()
    as the begin and end iterators, but this doesn't have to be
    the case:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 1

    Different algorithms can have different requirements for the
    iterators they accept. For example, qFill() accepts two 
    \l {forward iterators}. The iterator types required are specified
    for each algorithm. If an iterator of the wrong type is passed (for
    example, if QList::ConstIterator is passed as an \l {output
    iterator}), you will always get a compiler error, although not
    necessarily a very informative one.

    Some algorithms have special requirements on the value type
    stored in the containers. For example, qEqual() requires that the
    value type supports operator==(), which it uses to compare items.
    Similarly, qDeleteAll() requires that the value type is a
    non-const pointer type (for example, QWidget *). The value type
    requirements are specified for each algorithm, and the compiler
    will produce an error if a requirement isn't met.

    \target binaryFind example

    The generic algorithms can be used on other container classes
    than those provided by Qt and STL. The syntax of STL-style
    iterators is modeled after C++ pointers, so it's possible to use
    plain arrays as containers and plain pointers as iterators. A
    common idiom is to use qBinaryFind() together with two static
    arrays: one that contains a list of keys, and another that
    contains a list of associated values. For example, the following
    code will look up an HTML entity (e.g., \c &amp;) in the \c
    name_table array and return the corresponding Unicode value from
    the \c value_table if the entity is recognized:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 2

    This kind of code is for advanced users only; for most
    applications, a QMap- or QHash-based approach would work just as
    well:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 3

    \section1 Types of Iterators

    The algorithms have certain requirements on the iterator types
    they accept, and these are specified individually for each
    function. The compiler will produce an error if a requirement
    isn't met.

    \section2 Input Iterators

    An \e{input iterator} is an iterator that can be used for reading
    data sequentially from a container. It must provide the following
    operators: \c{==} and \c{!=} for comparing two iterators, unary
    \c{*} for retrieving the value stored in the item, and prefix
    \c{++} for advancing to the next item.

    The Qt containers' iterator types (const and non-const) are all
    input iterators.

    \section2 Output Iterators

    An \e{output iterator} is an iterator that can be used for
    writing data sequentially to a container or to some output
    stream. It must provide the following operators: unary \c{*} for
    writing a value (i.e., \c{*it = val}) and prefix \c{++} for
    advancing to the next item.

    The Qt containers' non-const iterator types are all output
    iterators.

    \section2 Forward Iterators

    A \e{forward iterator} is an iterator that meets the requirements
    of both input iterators and output iterators.

    The Qt containers' non-const iterator types are all forward
    iterators.

    \section2 Bidirectional Iterators

    A \e{bidirectional iterator} is an iterator that meets the
    requirements of forward iterators but that in addition supports
    prefix \c{--} for iterating backward.

    The Qt containers' non-const iterator types are all bidirectional
    iterators.

    \section2 Random Access Iterators

    The last category, \e{random access iterators}, is the most
    powerful type of iterator. It supports all the requirements of a
    bidirectional iterator, and supports the following operations:

    \table
    \row \i \c{i += n} \i advances iterator \c i by \c n positions
    \row \i \c{i -= n} \i moves iterator \c i back by \c n positions
    \row \i \c{i + n} or \c{n + i} \i returns the iterator for the item \c
       n positions ahead of iterator \c i
    \row \i \c{i - n} \i returns the iterator for the item \c n positions behind of iterator \c i
    \row \i \c{i - j} \i returns the number of items between iterators \c i and \c j
    \row \i \c{i[n]} \i same as \c{*(i + n)}
    \row \i \c{i < j} \i returns true if iterator \c j comes after iterator \c i
    \endtable

    QList and QVector's non-const iterator types are random access iterators.

    \sa {container classes}, <QtGlobal>
*/

/*! \fn OutputIterator qCopy(InputIterator begin1, InputIterator end1, OutputIterator begin2)
    \relates <QtAlgorithms>

    Copies the items from range [\a begin1, \a end1) to range [\a
    begin2, ...), in the order in which they appear.

    The item at position \a begin1 is assigned to that at position \a
    begin2; the item at position \a begin1 + 1 is assigned to that at
    position \a begin2 + 1; and so on.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 4

    \sa qCopyBackward(), {input iterators}, {output iterators}
*/

/*! \fn BiIterator2 qCopyBackward(BiIterator1 begin1, BiIterator1 end1, BiIterator2 end2)
    \relates <QtAlgorithms>

    Copies the items from range [\a begin1, \a end1) to range [...,
    \a end2).

    The item at position \a end1 - 1 is assigned to that at position
    \a end2 - 1; the item at position \a end1 - 2 is assigned to that
    at position \a end2 - 2; and so on.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 5

    \sa qCopy(), {bidirectional iterators}
*/

/*! \fn bool qEqual(InputIterator1 begin1, InputIterator1 end1, InputIterator2 begin2)
    \relates <QtAlgorithms>

    Compares the items in the range [\a begin1, \a end1) with the
    items in the range [\a begin2, ...). Returns true if all the
    items compare equal; otherwise returns false.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 6

    This function requires the item type (in the example above,
    QString) to implement \c operator==().

    \sa {input iterators}
*/

/*! \fn void qFill(ForwardIterator begin, ForwardIterator end, const T &value)
    \relates <QtAlgorithms>

    Fills the range [\a begin, \a end) with \a value.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 7

    \sa qCopy(), {forward iterators}
*/

/*! \fn void qFill(Container &container, const T &value)
    \relates <QtAlgorithms>

    \overload

    This is the same as qFill(\a{container}.begin(), \a{container}.end(), \a value);
*/

/*! \fn InputIterator qFind(InputIterator begin, InputIterator end, const T &value)
    \relates <QtAlgorithms>

    Returns an iterator to the first occurrence of \a value in a
    container in the range [\a begin, \a end). Returns \a end if \a
    value isn't found.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 8

    This function requires the item type (in the example above,
    QString) to implement \c operator==().

    If the items in the range are in ascending order, you can get
    faster results by using qLowerBound() or qBinaryFind() instead of
    qFind().

    \sa qBinaryFind(), {input iterators}
*/

/*! \fn void qFind(const Container &container, const T &value)
    \relates <QtAlgorithms>

    \overload

    This is the same as qFind(\a{container}.constBegin(), \a{container}.constEnd(), value);
*/

/*! \fn void qCount(InputIterator begin, InputIterator end, const T &value, Size &n)
    \relates <QtAlgorithms>

    Returns the number of occurrences of \a value in the range [\a begin, \a end),
    which is returned in \a n. \a n is never initialized, the count is added to \a n.
    It is the caller's responsibility to initialize \a n.

    Example:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 9

    This function requires the item type (in the example above,
    \c int) to implement \c operator==().

    \sa {input iterators}
*/

/*! \fn void qCount(const Container &container, const T &value, Size &n)
\relates <QtAlgorithms>

\overload

Instead of operating on iterators, as in the other overload, this function
operates on the specified \a container to obtain the number of instances
of \a value in the variable passed as a reference in argument \a n.
*/

/*! \fn void qSwap(T &var1, T &var2)
    \relates <QtAlgorithms>

    Exchanges the values of variables \a var1 and \a var2.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 10
*/

/*! \fn void qSort(RandomAccessIterator begin, RandomAccessIterator end)
    \relates <QtAlgorithms>

    Sorts the items in range [\a begin, \a end) in ascending order
    using the quicksort algorithm.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 11

    The sort algorithm is efficient on large data sets. It operates
    in \l {linear-logarithmic time}, O(\e{n} log \e{n}).

    This function requires the item type (in the example above,
    \c{int}) to implement \c operator<().

    If neither of the two items is "less than" the other, the items are
    taken to be equal. It is then undefined which one of the two
    items will appear before the other after the sort.

    \sa qStableSort(), {random access iterators}
*/

/*! \fn void qSort(RandomAccessIterator begin, RandomAccessIterator end, LessThan lessThan)
    \relates <QtAlgorithms>

    \overload

    Uses the \a lessThan function instead of \c operator<() to
    compare the items.

    For example, here's how to sort the strings in a QStringList
    in case-insensitive alphabetical order:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 12

    To sort values in reverse order, pass
    \l{qGreater()}{qGreater<T>()} as the \a lessThan parameter. For
    example:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 13

    If neither of the two items is "less than" the other, the items are
    taken to be equal. It is then undefined which one of the two
    items will appear before the other after the sort.

    An alternative to using qSort() is to put the items to sort in a
    QMap, using the sort key as the QMap key. This is often more
    convenient than defining a \a lessThan function. For example, the
    following code shows how to sort a list of strings case
    insensitively using QMap:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 14

    \sa QMap
*/

/*! \fn void qSort(Container &container)
    \relates <QtAlgorithms>

    \overload

    This is the same as qSort(\a{container}.begin(), \a{container}.end());
*/

/*! 
    \fn void qStableSort(RandomAccessIterator begin, RandomAccessIterator end)
    \relates <QtAlgorithms>

    Sorts the items in range [\a begin, \a end) in ascending order
    using a stable sorting algorithm.

    If neither of the two items is "less than" the other, the items are
    taken to be equal. The item that appeared before the other in the
    original container will still appear first after the sort. This
    property is often useful when sorting user-visible data.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 15

    The sort algorithm is efficient on large data sets. It operates
    in \l {linear-logarithmic time}, O(\e{n} log \e{n}).

    This function requires the item type (in the example above,
    \c{int}) to implement \c operator<().

    \sa qSort(), {random access iterators}
*/

/*! 
    \fn void qStableSort(RandomAccessIterator begin, RandomAccessIterator end, LessThan lessThan)
    \relates <QtAlgorithms>

    \overload

    Uses the \a lessThan function instead of \c operator<() to
    compare the items.

    For example, here's how to sort the strings in a QStringList
    in case-insensitive alphabetical order:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 16
    
    Note that earlier versions of Qt allowed using a lessThan function that took its
    arguments by non-const reference. From 4.3 and on this is no longer possible,
    the arguments has to be passed by const reference or value.

    To sort values in reverse order, pass
    \l{qGreater()}{qGreater<T>()} as the \a lessThan parameter. For
    example:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 17

    If neither of the two items is "less than" the other, the items are
    taken to be equal. The item that appeared before the other in the
    original container will still appear first after the sort. This
    property is often useful when sorting user-visible data.
*/

/*! 
    \fn void qStableSort(Container &container)
    \relates <QtAlgorithms>

    \overload

    This is the same as qStableSort(\a{container}.begin(), \a{container}.end());
*/

/*! \fn RandomAccessIterator qLowerBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
    \relates <QtAlgorithms>

    Performs a binary search of the range [\a begin, \a end) and
    returns the position of the first ocurrence of \a value. If no
    such item is found, returns the position where it should be
    inserted.

    The items in the range [\a begin, \e end) must be sorted in
    ascending order; see qSort().

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 18

    This function requires the item type (in the example above,
    \c{int}) to implement \c operator<().

    qLowerBound() can be used in conjunction with qUpperBound() to
    iterate over all occurrences of the same value:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 19

    \sa qUpperBound(), qBinaryFind()
*/

/*!
    \fn RandomAccessIterator qLowerBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
    \relates <QtAlgorithms>

    \overload

    Uses the \a lessThan function instead of \c operator<() to
    compare the items.

    Note that the items in the range must be sorted according to the order
    specified by the \a lessThan object.
*/

/*! 
    \fn void qLowerBound(const Container &container, const T &value)
    \relates <QtAlgorithms>

    \overload

    For read-only iteration over containers, this function is broadly equivalent to
    qLowerBound(\a{container}.begin(), \a{container}.end(), value). However, since it
    returns a const iterator, you cannot use it to modify the container; for example,
    to insert items.
*/

/*! \fn RandomAccessIterator qUpperBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
    \relates <QtAlgorithms>

    Performs a binary search of the range [\a begin, \a end) and
    returns the position of the one-past-the-last occurrence of \a
    value. If no such item is found, returns the position where the
    item should be inserted.

    The items in the range [\a begin, \e end) must be sorted in
    ascending order; see qSort().

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 20

    This function requires the item type (in the example above,
    \c{int}) to implement \c operator<().

    qUpperBound() can be used in conjunction with qLowerBound() to
    iterate over all occurrences of the same value:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 21

    \sa qLowerBound(), qBinaryFind()
*/

/*!
    \fn RandomAccessIterator qUpperBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
    \relates <QtAlgorithms>

    \overload

    Uses the \a lessThan function instead of \c operator<() to
    compare the items.

    Note that the items in the range must be sorted according to the order
    specified by the \a lessThan object.
*/

/*! 
    \fn void qUpperBound(const Container &container, const T &value)
    \relates <QtAlgorithms>

    \overload

    This is the same as qUpperBound(\a{container}.begin(), \a{container}.end(), value);
*/


/*! \fn RandomAccessIterator qBinaryFind(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
    \relates <QtAlgorithms>

    Performs a binary search of the range [\a begin, \a end) and
    returns the position of an occurrence of \a value. If there are
    no occurrences of \a value, returns \a end.

    The items in the range [\a begin, \a end) must be sorted in
    ascending order; see qSort().

    If there are many occurrences of the same value, any one of them
    could be returned. Use qLowerBound() or qUpperBound() if you need
    finer control.

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 22

    This function requires the item type (in the example above,
    QString) to implement \c operator<().

    See the \l{<QtAlgorithms>#binaryFind example}{detailed
    description} for an example usage.

    \sa qLowerBound(), qUpperBound(), {random access iterators}
*/

/*! \fn RandomAccessIterator qBinaryFind(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
    \relates <QtAlgorithms>

    \overload

    Uses the \a lessThan function instead of \c operator<() to
    compare the items.

    Note that the items in the range must be sorted according to the order
    specified by the \a lessThan object.
*/

/*! 
    \fn void qBinaryFind(const Container &container, const T &value)
    \relates <QtAlgorithms>

    \overload

    This is the same as qBinaryFind(\a{container}.begin(), \a{container}.end(), value);
*/


/*! 
    \fn void qDeleteAll(ForwardIterator begin, ForwardIterator end)
    \relates <QtAlgorithms>

    Deletes all the items in the range [\a begin, \a end) using the
    C++ \c delete operator. The item type must be a pointer type (for
    example, \c{QWidget *}).

    Example:
    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 23

    Notice that qDeleteAll() doesn't remove the items from the
    container; it merely calls \c delete on them. In the example
    above, we call clear() on the container to remove the items.

    This function can also be used to delete items stored in
    associative containers, such as QMap and QHash. Only the objects
    stored in each container will be deleted by this function; objects
    used as keys will not be deleted.

    \sa {forward iterators}
*/

/*! 
    \fn void qDeleteAll(const Container &c)
    \relates <QtAlgorithms>

    \overload

    This is the same as qDeleteAll(\a{c}.begin(), \a{c}.end()).
*/

/*! \fn LessThan qLess()
    \relates <QtAlgorithms>

    Returns a functional object, or functor, that can be passed to qSort()
    or qStableSort().

    Example:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 24

    \sa {qGreater()}{qGreater<T>()}
*/

/*! \fn LessThan qGreater()
    \relates <QtAlgorithms>

    Returns a functional object, or functor, that can be passed to qSort()
    or qStableSort().

    Example:

    \snippet doc/src/snippets/code/doc_src_qalgorithms.cpp 25

    \sa {qLess()}{qLess<T>()}
*/

#ifndef BUBBLEFS_UTILS_QT_QALGORITHMS_H_
#define BUBBLEFS_UTILS_QT_QALGORITHMS_H_

#include "platform/qt_qglobal.h"

namespace bubblefs {
namespace myqt {
/*
    Warning: The contents of QAlgorithmsPrivate is not a part of the public Qt API
    and may be changed from version to version or even be completely removed.
*/
namespace QAlgorithmsPrivate {

template <typename RandomAccessIterator, typename T, typename LessThan>
inline void qSortHelper(RandomAccessIterator start, RandomAccessIterator end, const T &t, LessThan lessThan);
template <typename RandomAccessIterator, typename T>
inline void qSortHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &dummy);

template <typename RandomAccessIterator, typename T, typename LessThan>
inline void qStableSortHelper(RandomAccessIterator start, RandomAccessIterator end, const T &t, LessThan lessThan);
template <typename RandomAccessIterator, typename T>
inline void qStableSortHelper(RandomAccessIterator, RandomAccessIterator, const T &);

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qLowerBoundHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan);
template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qUpperBoundHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan);
template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qBinaryFindHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan);

}

template <typename InputIterator, typename OutputIterator>
inline OutputIterator qCopy(InputIterator begin, InputIterator end, OutputIterator dest)
{
    while (begin != end)
        *dest++ = *begin++;
    return dest;
}

template <typename BiIterator1, typename BiIterator2>
inline BiIterator2 qCopyBackward(BiIterator1 begin, BiIterator1 end, BiIterator2 dest)
{
    while (begin != end)
        *--dest = *--end;
    return dest;
}

template <typename InputIterator1, typename InputIterator2>
inline bool qEqual(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
    for (; first1 != last1; ++first1, ++first2)
        if (!(*first1 == *first2))
            return false;
    return true;
}

template <typename ForwardIterator, typename T>
inline void qFill(ForwardIterator first, ForwardIterator last, const T &val)
{
    for (; first != last; ++first)
        *first = val;
}

template <typename Container, typename T>
inline void qFill(Container &container, const T &val)
{
    qFill(container.begin(), container.end(), val);
}

template <typename InputIterator, typename T>
inline InputIterator qFind(InputIterator first, InputIterator last, const T &val)
{
    while (first != last && !(*first == val))
        ++first;
    return first;
}

template <typename Container, typename T>
inline typename Container::const_iterator qFind(const Container &container, const T &val)
{
    return qFind(container.constBegin(), container.constEnd(), val);
}

template <typename InputIterator, typename T, typename Size>
inline void qCount(InputIterator first, InputIterator last, const T &value, Size &n)
{
    for (; first != last; ++first)
        if (*first == value)
            ++n;
}

template <typename Container, typename T, typename Size>
inline void qCount(const Container &container, const T &value, Size &n)
{
    qCount(container.constBegin(), container.constEnd(), value, n);
}

#ifdef qdoc
template <typename T>
LessThan qLess()
{
}

template <typename T>
LessThan qGreater()
{
}
#else
template <typename T>
class qLess
{
public:
    inline bool operator()(const T &t1, const T &t2) const
    {
        return (t1 < t2);
    }
};

template <typename T>
class qGreater
{
public:
    inline bool operator()(const T &t1, const T &t2) const
    {
        return (t2 < t1);
    }
};
#endif

template <typename RandomAccessIterator>
inline void qSort(RandomAccessIterator start, RandomAccessIterator end)
{
    if (start != end)
        QAlgorithmsPrivate::qSortHelper(start, end, *start);
}

template <typename RandomAccessIterator, typename LessThan>
inline void qSort(RandomAccessIterator start, RandomAccessIterator end, LessThan lessThan)
{
    if (start != end)
        QAlgorithmsPrivate::qSortHelper(start, end, *start, lessThan);
}

template<typename Container>
inline void qSort(Container &c)
{
#ifdef Q_CC_BOR
    // Work around Borland 5.5 optimizer bug
    c.detach();
#endif
    if (!c.empty())
        QAlgorithmsPrivate::qSortHelper(c.begin(), c.end(), *c.begin());
}

template <typename RandomAccessIterator>
inline void qStableSort(RandomAccessIterator start, RandomAccessIterator end)
{
    if (start != end)
        QAlgorithmsPrivate::qStableSortHelper(start, end, *start);
}

template <typename RandomAccessIterator, typename LessThan>
inline void qStableSort(RandomAccessIterator start, RandomAccessIterator end, LessThan lessThan)
{
    if (start != end)
        QAlgorithmsPrivate::qStableSortHelper(start, end, *start, lessThan);
}

template<typename Container>
inline void qStableSort(Container &c)
{
#ifdef Q_CC_BOR
    // Work around Borland 5.5 optimizer bug
    c.detach();
#endif
    if (!c.empty())
        QAlgorithmsPrivate::qStableSortHelper(c.begin(), c.end(), *c.begin());
}

template <typename RandomAccessIterator, typename T>
inline RandomAccessIterator qLowerBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
{
    // Implementation is duplicated from QAlgorithmsPrivate to keep existing code
    // compiling. We have to allow using *begin and value with different types,
    // and then implementing operator< for those types.
    RandomAccessIterator middle;
    int n = end - begin;
    int half;

    while (n > 0) {
        half = n >> 1;
        middle = begin + half;
        if (*middle < value) {
            begin = middle + 1;
            n -= half + 1;
        } else {
            n = half;
        }
    }
    return begin;
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qLowerBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    return QAlgorithmsPrivate::qLowerBoundHelper(begin, end, value, lessThan);
}

template <typename Container, typename T>
inline typename Container::const_iterator qLowerBound(const Container &container, const T &value)
{
    return QAlgorithmsPrivate::qLowerBoundHelper(container.constBegin(), container.constEnd(), value, qLess<T>());
}

template <typename RandomAccessIterator, typename T>
inline RandomAccessIterator qUpperBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
{
    // Implementation is duplicated from QAlgorithmsPrivate.
    RandomAccessIterator middle;
    int n = end - begin;
    int half;

    while (n > 0) {
        half = n >> 1;
        middle = begin + half;
        if (value < *middle) {
            n = half;
        } else {
            begin = middle + 1;
            n -= half + 1;
        }
    }
    return begin;
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qUpperBound(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    return QAlgorithmsPrivate::qUpperBoundHelper(begin, end, value, lessThan);
}

template <typename Container, typename T>
inline typename Container::const_iterator qUpperBound(const Container &container, const T &value)
{
    return QAlgorithmsPrivate::qUpperBoundHelper(container.constBegin(), container.constEnd(), value, qLess<T>());
}

template <typename RandomAccessIterator, typename T>
inline RandomAccessIterator qBinaryFind(RandomAccessIterator begin, RandomAccessIterator end, const T &value)
{
    // Implementation is duplicated from QAlgorithmsPrivate.
    RandomAccessIterator it = qLowerBound(begin, end, value);

    if (it == end || value < *it)
        return end;

    return it;
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qBinaryFind(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    return QAlgorithmsPrivate::qBinaryFindHelper(begin, end, value, lessThan);
}

template <typename Container, typename T>
inline typename Container::const_iterator qBinaryFind(const Container &container, const T &value)
{
    return QAlgorithmsPrivate::qBinaryFindHelper(container.constBegin(), container.constEnd(), value, qLess<T>());
}

template <typename ForwardIterator>
inline void qDeleteAll(ForwardIterator begin, ForwardIterator end)
{
    while (begin != end) {
        delete *begin;
        ++begin;
    }
}

template <typename Container>
inline void qDeleteAll(const Container &c)
{
    qDeleteAll(c.begin(), c.end());
}

/*
    Warning: The contents of QAlgorithmsPrivate is not a part of the public Qt API
    and may be changed from version to version or even be completely removed.
*/
namespace QAlgorithmsPrivate {

template <typename RandomAccessIterator, typename T, typename LessThan>
inline void qSortHelper(RandomAccessIterator start, RandomAccessIterator end, const T &t, LessThan lessThan)
{
top:
    int span = int(end - start);
    if (span < 2)
        return;

    --end;
    RandomAccessIterator low = start, high = end - 1;
    RandomAccessIterator pivot = start + span / 2;

    if (lessThan(*end, *start))
        qSwap(*end, *start);
    if (span == 2)
        return;

    if (lessThan(*pivot, *start))
        qSwap(*pivot, *start);
    if (lessThan(*end, *pivot))
        qSwap(*end, *pivot);
    if (span == 3)
        return;

    qSwap(*pivot, *end);

    while (low < high) {
        while (low < high && lessThan(*low, *end))
            ++low;

        while (high > low && lessThan(*end, *high))
            --high;

        if (low < high) {
            qSwap(*low, *high);
            ++low;
            --high;
        } else {
            break;
        }
    }

    if (lessThan(*low, *end))
        ++low;

    qSwap(*end, *low);
    qSortHelper(start, low, t, lessThan);

    start = low + 1;
    ++end;
    goto top;
}

template <typename RandomAccessIterator, typename T>
inline void qSortHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &dummy)
{
    qSortHelper(begin, end, dummy, qLess<T>());
}

template <typename RandomAccessIterator>
inline void qReverse(RandomAccessIterator begin, RandomAccessIterator end)
{
    --end;
    while (begin < end)
        qSwap(*begin++, *end--);
}

template <typename RandomAccessIterator>
inline void qRotate(RandomAccessIterator begin, RandomAccessIterator middle, RandomAccessIterator end)
{
    qReverse(begin, middle);
    qReverse(middle, end);
    qReverse(begin, end);
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline void qMerge(RandomAccessIterator begin, RandomAccessIterator pivot, RandomAccessIterator end, T &t, LessThan lessThan)
{
    const int len1 = pivot - begin;
    const int len2 = end - pivot;

    if (len1 == 0 || len2 == 0)
        return;

    if (len1 + len2 == 2) {
        if (lessThan(*(begin + 1), *(begin)))
            qSwap(*begin, *(begin + 1));
        return;
    }

    RandomAccessIterator firstCut;
    RandomAccessIterator secondCut;
    int len2Half;
    if (len1 > len2) {
        const int len1Half = len1 / 2;
        firstCut = begin + len1Half;
        secondCut = qLowerBound(pivot, end, *firstCut, lessThan);
        len2Half = secondCut - pivot;
    } else {
        len2Half = len2 / 2;
        secondCut = pivot + len2Half;
        firstCut = qUpperBound(begin, pivot, *secondCut, lessThan);
    }

    qRotate(firstCut, pivot, secondCut);
    const RandomAccessIterator newPivot = firstCut + len2Half;
    qMerge(begin, firstCut, newPivot, t, lessThan);
    qMerge(newPivot, secondCut, end, t, lessThan);
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline void qStableSortHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &t, LessThan lessThan)
{
    const int span = end - begin;
    if (span < 2)
       return;

    const RandomAccessIterator middle = begin + span / 2;
    qStableSortHelper(begin, middle, t, lessThan);
    qStableSortHelper(middle, end, t, lessThan);
    qMerge(begin, middle, end, t, lessThan);
}

template <typename RandomAccessIterator, typename T>
inline void qStableSortHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &dummy)
{
    qStableSortHelper(begin, end, dummy, qLess<T>());
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qLowerBoundHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    RandomAccessIterator middle;
    int n = int(end - begin);
    int half;

    while (n > 0) {
        half = n >> 1;
        middle = begin + half;
        if (lessThan(*middle, value)) {
            begin = middle + 1;
            n -= half + 1;
        } else {
            n = half;
        }
    }
    return begin;
}


template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qUpperBoundHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    RandomAccessIterator middle;
    int n = end - begin;
    int half;

    while (n > 0) {
        half = n >> 1;
        middle = begin + half;
        if (lessThan(value, *middle)) {
            n = half;
        } else {
            begin = middle + 1;
            n -= half + 1;
        }
    }
    return begin;
}

template <typename RandomAccessIterator, typename T, typename LessThan>
inline RandomAccessIterator qBinaryFindHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    RandomAccessIterator it = qLowerBoundHelper(begin, end, value, lessThan);

    if (it == end || lessThan(value, *it))
        return end;

    return it;
}

} //namespace QAlgorithmsPrivate
} // namespace myqt
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_QT_QALGORITHMS_H_