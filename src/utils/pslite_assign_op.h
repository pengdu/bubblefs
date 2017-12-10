/**
 *  Copyright (c) 2015 by Contributors
 * \file   assign_op.h
 * \brief  assignment operator
 * http://en.cppreference.com/w/cpp/language/operator_assignment
 */

// ps-lite/include/ps/internal/assign_op.h

#ifndef BUBBLEFS_UTILS_PSLITE_ASSIGN_OP_H_
#define BUBBLEFS_UTILS_PSLITE_ASSIGN_OP_H_

#include "platform/base_error.h"
#include "utils/pslite_env.h"

namespace bubblefs {
namespace mypslite {

enum AssignOp {
  ASSIGN,  // a = b
  PLUS,    // a += b
  MINUS,   // a -= b
  TIMES,   // a *= b
  DIVIDE,  // a -= b
  AND,     // a &= b
  OR,      // a |= b
  XOR      // a ^= b
};

/**
 * \brief return an assignment function: right op= left
 */
template<typename T>
inline void AssignFunc(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN:
      *rhs = lhs; break;
    case PLUS:
      *rhs += lhs; break;
    case MINUS:
      *rhs -= lhs; break;
    case TIMES:
      *rhs *= lhs; break;
    case DIVIDE:
      *rhs /= lhs; break;
    default:
      PANIC("use AssignOpInt..");
  }
}

/**
 * \brief return an assignment function including bit operations, only
 * works for integers
 */
template<typename T>
inline void AssignFuncInt(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN:
      *rhs = lhs; break;
    case PLUS:
      *rhs += lhs; break;
    case MINUS:
      *rhs -= lhs; break;
    case TIMES:
      *rhs *= lhs; break;
    case DIVIDE:
      *rhs /= lhs; break;
    case AND:
      *rhs &= lhs; break;
    case OR:
      *rhs |= lhs; break;
    case XOR:
      *rhs ^= lhs; break;
    default:
      PANIC("use AssignFuncInt..");
  }
}

}  // namespace mypslite
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PSLITE_ASSIGN_OP_H_