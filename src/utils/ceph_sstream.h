
// ceph/src/include/types.h

#ifndef BUBBLEFS_UTILS_CEPH_SSTREAM_H_
#define BUBBLEFS_UTILS_CEPH_SSTREAM_H_

#include <deque>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

//namespace bubblefs {
//namespace myceph {

// -- io helpers --
// Forward declare all the I/O helpers so strict ADL can find them in
// the case of containers of containers. I'm tempted to abstract this
// stuff using template templates like I did for denc.
// Koening lookup/argument-dependent lookup ? 

template<class A, class B>
inline std::ostream& operator<<(std::ostream&out, const std::pair<A,B>& v);
template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::vector<A,Alloc>& v);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::deque<A,Alloc>& v);
template<class A, class B, class C>
inline std::ostream& operator<<(std::ostream&out, const std::tuple<A, B, C> &t);
template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::list<A,Alloc>& ilist);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::set<A, Comp, Alloc>& iset);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multiset<A,Comp,Alloc>& iset);
template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::map<A,B,Comp,Alloc>& m);
template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multimap<A,B,Comp,Alloc>& m);

template<class A, class B>
inline std::ostream& operator<<(std::ostream& out, const std::pair<A,B>& v) {
  return out << "pair(" << v.first << ", " << v.second << ")";
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::vector<A,Alloc>& v) {
  out << "vector[";
  for (auto p = v.begin(); p != v.end(); ++p) {
    if (p != v.begin()) out << ", ";
    out << *p;
  }
  out << "]";
  return out;
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::deque<A,Alloc>& v) {
  out << "deque<";
  for (auto p = v.begin(); p != v.end(); ++p) {
    if (p != v.begin()) out << ", ";
    out << *p;
  }
  out << ">";
  return out;
}

template<class A, class B, class C>
inline std::ostream& operator<<(std::ostream& out, const std::tuple<A, B, C> &t) {
  out << std::get<0>(t) <<"," << std::get<1>(t) << "," << std::get<2>(t);
  return out;
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::list<A,Alloc>& ilist) {
  out << "list[";
  for (auto it = ilist.begin();
       it != ilist.end();
       ++it) {
    if (it != ilist.begin()) out << ", ";
    out << *it;
  }
  out << "]";
  return out;
}

template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::set<A, Comp, Alloc>& iset) {
  out << "set(";
  for (auto it = iset.begin();
       it != iset.end();
       ++it) {
    if (it != iset.begin()) out << ", ";
    out << *it;
  }
  out << ")";
  return out;
}

template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multiset<A,Comp,Alloc>& iset) {
  out << "multiset(";
  for (auto it = iset.begin();
       it != iset.end();
       ++it) {
    if (it != iset.begin()) out << ", ";
    out << *it;
  }
  out << ")";
  return out;
}

template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::map<A,B,Comp,Alloc>& m)
{
  out << "map{";
  for (auto it = m.begin();
       it != m.end();
       ++it) {
    if (it != m.begin()) out << ", ";
    out << it->first << "=" << it->second;
  }
  out << "}";
  return out;
}

template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multimap<A,B,Comp,Alloc>& m)
{
  out << "multimap{{";
  for (auto it = m.begin();
       it != m.end();
       ++it) {
    if (it != m.begin()) out << ", ";
    out << it->first << "=" << it->second;
  }
  out << "}}";
  return out;
}

//} // namespace myceph
//} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CEPH_SSTREAM_H_