/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/math/TensorExpression.h
// Paddle/paddle/math/TensorApply.h
// Paddle/paddle/math/TensorEvaluate.h
// Paddle/paddle/math/TensorAssign.h

#pragma once


#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <boost/iterator/iterator_concepts.hpp>
#include "hl_tensor_ops.h"

namespace bubblefs {
namespace mypaddle {

template <class OP, typename ExprType, class T>
class TensorConstant;
template <class OP, typename ExprType, class T>
class TensorUnaryOp;
template <class OP, typename LhsType, typename RhsType, class T>
class TensorBinaryOp;
template <typename ExprType1, typename ExprType2, typename ExprType3, class T>
class TensorTernaryOp;

template <typename LhsType, typename RhsType, class T>
class TensorAssignOp;

/**
 * \brief Tensor base class.
 *
 * This is the base class of all Tensor and Expression class.
 */
template <typename Derived, class T>
class TensorExpression {
public:
  /**
   * Element wise unary expression.
   */
  template <typename UnaryOp>
  const TensorUnaryOp<UnaryOp, const Derived, T> unaryExpression(
      const UnaryOp& op) const {
    return TensorUnaryOp<UnaryOp, const Derived, T>(op, derived());
  }

  const TensorUnaryOp<hppl::unary::add_scale<T>, const Derived, T> operator+(
      T p) const {
    return unaryExpression(hppl::unary::add_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::sub_scale<T>, const Derived, T> operator-(
      T p) const {
    return unaryExpression(hppl::unary::sub_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::mul_scale<T>, const Derived, T> operator*(
      T p) const {
    return unaryExpression(hppl::unary::mul_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::div_scale<T>, const Derived, T> operator/(
      T p) const {
    return unaryExpression(hppl::unary::div_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::neg<T>, const Derived, T> operator-() const {
    return unaryExpression(hppl::unary::neg<T>());
  }

  const TensorUnaryOp<hppl::unary::exp_op<T>, const Derived, T> exp() const {
    return unaryExpression(hppl::unary::exp_op<T>());
  }

  const TensorUnaryOp<hppl::unary::log_op<T>, const Derived, T> log() const {
    return unaryExpression(hppl::unary::log_op<T>());
  }

  const TensorUnaryOp<hppl::unary::sqrt_op<T>, const Derived, T> sqrt() const {
    return unaryExpression(hppl::unary::sqrt_op<T>());
  }

  const TensorUnaryOp<hppl::unary::square<T>, const Derived, T> square() const {
    return unaryExpression(hppl::unary::square<T>());
  }

  const TensorUnaryOp<hppl::unary::reciprocal<T>, const Derived, T> reciprocal()
      const {
    return unaryExpression(hppl::unary::reciprocal<T>());
  }

  const TensorUnaryOp<hppl::unary::abs<T>, const Derived, T> abs() const {
    return unaryExpression(hppl::unary::abs<T>());
  }

  const TensorUnaryOp<hppl::unary::sign<T>, const Derived, T> sign() const {
    return unaryExpression(hppl::unary::sign<T>());
  }

  const TensorUnaryOp<hppl::unary::pow_op<T>, const Derived, T> pow(T p) const {
    return unaryExpression(hppl::unary::pow_op<T>(p));
  }

  const TensorUnaryOp<hppl::unary::min<T>, const Derived, T> min(T p) const {
    return unaryExpression(hppl::unary::min<T>(p));
  }

  const TensorUnaryOp<hppl::unary::max<T>, const Derived, T> max(T p) const {
    return unaryExpression(hppl::unary::max<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_eq<T>, const Derived, T> operator==(
      T p) const {
    return unaryExpression(hppl::unary::cmp_eq<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_ne<T>, const Derived, T> operator!=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_ne<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_le<T>, const Derived, T> operator<=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_le<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_lt<T>, const Derived, T> operator<(
      T p) const {
    return unaryExpression(hppl::unary::cmp_lt<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_ge<T>, const Derived, T> operator>=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_ge<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_gt<T>, const Derived, T> operator>(
      T p) const {
    return unaryExpression(hppl::unary::cmp_gt<T>(p));
  }

  const TensorUnaryOp<hppl::unary::and_op<T>, const Derived, T> operator&&(
      T p) const {
    return unaryExpression(hppl::unary::and_op<T>(p));
  }

  const TensorUnaryOp<hppl::unary::or_op<T>, const Derived, T> operator||(
      T p) const {
    return unaryExpression(hppl::unary::or_op<T>(p));
  }

  /**
   * Element wise binary expression.
   */
  template <typename BinaryOp, typename ExpressionType>
  const TensorBinaryOp<BinaryOp, const Derived, const ExpressionType, T>
  binaryExpression(const BinaryOp& op, const ExpressionType& expr) const {
    return TensorBinaryOp<BinaryOp, const Derived, const ExpressionType, T>(
        op, derived(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_eq<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator==(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_eq<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_ne<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator!=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_ne<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_le<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator<=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_le<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_lt<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator<(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_lt<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_ge<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator>=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_ge<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_gt<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator>(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_gt<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::and_op<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator&&(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::and_op<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::or_op<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator||(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::or_op<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::add<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator+(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::add<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::sub<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator-(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::sub<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::mul<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator*(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::mul<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::div<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator/(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::div<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::min<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  min(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::min<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::max<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  max(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::max<T>(), expr);
  }

  /**
   * Element wise ternary expression.
   *
   * ternary conditional operator(?: operator).
   * The conditional expression returns one of two values depending on
   * the result of derived expression.
   * If derived expression evaluates to true, then expression1 is evaluated.
   * If derived expression evaluates to false, then expression2 is evaluated.
   */
  template <typename ExprType1, typename ExprType2>
  const TensorTernaryOp<const Derived, const ExprType1, const ExprType2, T>
  condition(const ExprType1& expr1, const ExprType2& expr2) const {
    return TensorTernaryOp<const Derived, const ExprType1, const ExprType2, T>(
        derived(), expr1, expr2);
  }

  template <typename ExprType>
  const TensorTernaryOp<
      const Derived,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      const ExprType,
      T>
  condition(T p, const ExprType& expr) const {
    return condition(constant(p), expr);
  }

  template <typename ExprType>
  const TensorTernaryOp<
      const Derived,
      const ExprType,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      T>
  condition(const ExprType& expr, T p) const {
    return condition(expr, constant(p));
  }

  const TensorTernaryOp<
      const Derived,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      T>
  condition(T p1, T p2) const {
    return condition(constant(p1), constant(p2));
  }

  /**
   * return a TensorConstant. A TensorConstant object hold a constant value.
   */
  const TensorConstant<hppl::unary::constant<T>, const Derived, T> constant(
      T p) const {
    return TensorConstant<hppl::unary::constant<T>, const Derived, T>(
        hppl::unary::constant<T>(p), derived());
  }

  /**
   * return a TensorAssignOp, and use AssignEvaluate to evaluate one or more
   * TensorAssignOp objects.
   */
  template <typename ExpressionType>
  TensorAssignOp<Derived, ExpressionType, T> lazyAssign(
      const ExpressionType& expr) const {
    return TensorAssignOp<Derived, ExpressionType, T>(derived(), expr);
  }

protected:
  const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

/**
 * \brief Unary Operator Expression
 */
template <class OP, typename ExprType, class T>
class TensorUnaryOp
    : public TensorExpression<TensorUnaryOp<OP, ExprType, T>, T> {
public:
  explicit TensorUnaryOp(const OP op, const ExprType& expr)
      : op_(op), expr_(expr) {}

  const OP op_;
  const ExprType expr_;
};

/**
 * \brief Binary Operator Expression
 */
template <class OP, typename LhsType, typename RhsType, class T>
class TensorBinaryOp
    : public TensorExpression<TensorBinaryOp<OP, LhsType, RhsType, T>, T> {
public:
  explicit TensorBinaryOp(const OP op, const LhsType& lhs, const RhsType& rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}

  const OP op_;
  const LhsType lhs_;
  const RhsType rhs_;
};

/**
 * \brief Ternary Operator Expression
 */
template <typename ExprType1, typename ExprType2, typename ExprType3, class T>
class TensorTernaryOp : public TensorExpression<
                            TensorTernaryOp<ExprType1, ExprType2, ExprType3, T>,
                            T> {
public:
  explicit TensorTernaryOp(const ExprType1& expr1,
                           const ExprType2& expr2,
                           const ExprType3& expr3)
      : expr1_(expr1), expr2_(expr2), expr3_(expr3) {}

  const ExprType1 expr1_;
  const ExprType2 expr2_;
  const ExprType3 expr3_;
};

/**
 * \brief Constant Expression
 */
template <class OP, typename ExprType, class T>
class TensorConstant
    : public TensorExpression<TensorConstant<OP, ExprType, T>, T> {
public:
  explicit TensorConstant(const OP op, const ExprType& expr)
      : op_(op), expr_(expr) {}

  const OP op_;
  const ExprType expr_;
};

/**
 * \brief operator+ overload
 * \return a unary operator expression
 */
template <typename Derived, class T>
const TensorUnaryOp<hppl::unary::add_scale<T>, const Derived, T> operator+(
    T p, const TensorExpression<Derived, T>& expr) {
  return expr + p;
}

/**
 * \brief operator* overload
 * \return a unary operator expression
 */
template <typename Derived, class T>
const TensorUnaryOp<hppl::unary::mul_scale<T>, const Derived, T> operator*(
    T p, const TensorExpression<Derived, T>& expr) {
  return expr * p;
}

/**
 * \brief The tensor evaluator classes.
 */
template <typename Derived, class T>
class TensorApply {
public:
  explicit INLINE TensorApply(const Derived& p)
      : data_(p.data_),
        stride_(p.stride_),
        height_(p.height_),
        width_(p.width_),
        useGpu_(p.useGpu_) {}

  INLINE T apply(int i, int j) const { return data_[i * stride_ + j]; }
  INLINE T apply(int index) const { return data_[index]; }
  INLINE T& applyRef(int i, int j) { return data_[i * stride_ + j]; }
  INLINE T& applyRef(int index) { return data_[index]; }

  INLINE size_t getWidth() const { return width_; }
  INLINE size_t getHeight() const { return height_; }
  INLINE bool isContiguous() const { return stride_ == width_ || height_ == 1; }
  INLINE bool useGpu() const { return useGpu_; }

  T* data_;
  size_t stride_;
  size_t height_;
  size_t width_;
  bool useGpu_;
};

/**
 * \brief The tensor evaluator classes.
 * evaluator for rvalues
 */
template <typename Derived, class T>
class TensorApply<const Derived, T> {
public:
  explicit INLINE TensorApply(const Derived& p)
      : data_(p.data_),
        stride_(p.stride_),
        height_(p.height_),
        width_(p.width_),
        useGpu_(p.useGpu_) {}

  INLINE T apply(int i, int j) const { return data_[i * stride_ + j]; }
  INLINE T apply(int index) const { return data_[index]; }

  INLINE size_t getWidth() const { return width_; }
  INLINE size_t getHeight() const { return height_; }
  INLINE bool isContiguous() const { return stride_ == width_ || height_ == 1; }
  INLINE bool useGpu() const { return useGpu_; }

  const T* data_;
  size_t stride_;
  size_t height_;
  size_t width_;
  bool useGpu_;
};

template <typename Derived, class T>
class TensorApply<const TensorExpression<Derived, T>, T> {
public:
  explicit TensorApply(const TensorExpression<Derived, T>& expr)
      : expr_(expr.derived()) {}

  INLINE T apply(int i, int j) const { return expr_.apply(i, j); }
  INLINE T apply(int index) const { return expr_.apply(index); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return expr_.isContiguous(); }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  TensorApply<const Derived, T> expr_;
};

/**
 * \brief The unary expression evaluator classes.
 */
template <class OP, typename ArgType, class T>
class TensorApply<const TensorUnaryOp<OP, ArgType, T>, T> {
public:
  explicit INLINE TensorApply(const TensorUnaryOp<OP, ArgType, T>& expr)
      : op_(expr.op_), expr_(expr.expr_) {}

  INLINE T apply(int i, int j) const { return op_(expr_.apply(i, j)); }
  INLINE T apply(int index) const { return op_(expr_.apply(index)); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return expr_.isContiguous(); }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  const OP op_;
  TensorApply<ArgType, T> expr_;
};

/**
 * \brief The binary expression evaluator classes.
 */
template <class OP, typename LhsType, typename RhsType, class T>
class TensorApply<const TensorBinaryOp<OP, LhsType, RhsType, T>, T> {
public:
  explicit INLINE TensorApply(
      const TensorBinaryOp<OP, LhsType, RhsType, T>& expr)
      : op_(expr.op_), lhs_(expr.lhs_), rhs_(expr.rhs_) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
    CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
    CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());
#endif
  }

  INLINE T apply(int i, int j) const {
    return op_(lhs_.apply(i, j), rhs_.apply(i, j));
  }
  INLINE T apply(int index) const {
    return op_(lhs_.apply(index), rhs_.apply(index));
  }

  INLINE size_t getWidth() const { return lhs_.getWidth(); }
  INLINE size_t getHeight() const { return rhs_.getHeight(); }
  INLINE bool isContiguous() const {
    return lhs_.isContiguous() && rhs_.isContiguous();
  }
  INLINE bool useGpu() const { return lhs_.useGpu(); }

  const OP op_;
  TensorApply<LhsType, T> lhs_;
  TensorApply<RhsType, T> rhs_;
};

/**
 * \brief The ternary expression evaluator classes.
 */
template <typename ArgType1, typename ArgType2, typename ArgType3, class T>
class TensorApply<const TensorTernaryOp<ArgType1, ArgType2, ArgType3, T>, T> {
public:
  explicit INLINE TensorApply(
      const TensorTernaryOp<ArgType1, ArgType2, ArgType3, T>& expr)
      : expr1_(expr.expr1_), expr2_(expr.expr2_), expr3_(expr.expr3_) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(expr1_.getWidth(), expr2_.getWidth());
    CHECK_EQ(expr1_.getWidth(), expr3_.getWidth());
    CHECK_EQ(expr1_.getHeight(), expr2_.getHeight());
    CHECK_EQ(expr1_.getHeight(), expr3_.getHeight());
    CHECK_EQ(expr1_.useGpu(), expr2_.useGpu());
    CHECK_EQ(expr1_.useGpu(), expr3_.useGpu());
#endif
  }

  INLINE T apply(int i, int j) const {
    return expr1_.apply(i, j) ? expr2_.apply(i, j) : expr3_.apply(i, j);
  }
  INLINE T apply(int index) const {
    return expr1_.apply(index) ? expr2_.apply(index) : expr3_.apply(index);
  }

  INLINE size_t getWidth() const { return expr1_.getWidth(); }
  INLINE size_t getHeight() const { return expr1_.getHeight(); }
  INLINE bool isContiguous() const {
    return expr1_.isContiguous() && expr2_.isContiguous() &&
           expr3_.isContiguous();
  }
  INLINE bool useGpu() const { return expr1_.useGpu(); }

  TensorApply<ArgType1, T> expr1_;
  TensorApply<ArgType2, T> expr2_;
  TensorApply<ArgType3, T> expr3_;
};

/**
 * \brief The const expression evaluator classes.
 */
template <class OP, typename ArgType, class T>
class TensorApply<const TensorConstant<OP, ArgType, T>, T> {
public:
  explicit INLINE TensorApply(const TensorConstant<OP, ArgType, T>& expr)
      : op_(expr.op_), expr_(expr.expr_) {}

  INLINE T apply(int i, int j) const { return op_(i, j); }
  INLINE T apply(int index) const { return op_(index); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return true; }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  const OP op_;
  TensorApply<ArgType, T> expr_;
};

/**
 * \brief The tensor cpu evaluate api.
 */
template <class T, typename LeftType, typename RightType>
inline void TensorCpuApply(LeftType& lhs, const RightType& rhs) {
  TensorApply<LeftType, T> lhs_(lhs);
  TensorApply<const RightType, T> rhs_(rhs);
  CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
  CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
  CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());

  int height = lhs_.getHeight();
  int width = lhs_.getWidth();
  if (lhs_.isContiguous() && rhs_.isContiguous()) {
    int size = height * width;
    for (int index = 0; index < size; index++) {
      lhs_.applyRef(index) = rhs_.apply(index);
    }
  } else {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        lhs_.applyRef(i, j) = rhs_.apply(i, j);
      }
    }
  }
}

#ifdef __NVCC__
template <typename LeftType, typename RightType>
__global__ void TensorElementWiseOp(LeftType lhs,
                                    RightType rhs,
                                    const int border) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    lhs.applyRef(idx) = rhs.apply(idx);
  }
}

template <typename LeftType, typename RightType>
__global__ void TensorElementWiseOp(LeftType lhs, RightType rhs) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < lhs.getHeight(); i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < lhs.getWidth(); j += gridDim.x * blockDim.x) {
      lhs.applyRef(i, j) = rhs.apply(i, j);
    }
  }
}

/**
 * \brief The tensor gpu evaluate api.
 */
template <class T, typename LeftType, typename RightType>
inline void TensorGpuApply(LeftType& lhs, const RightType& rhs) {
  TensorApply<LeftType, T> lhs_(lhs);
  TensorApply<const RightType, T> rhs_(rhs);
  CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
  CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
  CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());

  int dimM = lhs_.getHeight();
  int dimN = lhs_.getWidth();

  if (lhs_.isContiguous() && rhs_.isContiguous()) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    TensorElementWiseOp<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
        lhs_, rhs_, size);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    TensorElementWiseOp<<<grid, threads, 0, STREAM_DEFAULT>>>(lhs_, rhs_);
  }

  CHECK_SYNC("TensorGpuApply failed");
}
#else
template <class T, typename LeftType, typename RightType>
inline void TensorGpuApply(LeftType& lhs, RightType& rhs) {
  LOG(FATAL) << "Since it is gcc compiled, "
                "this calculation does not support GPU implementation.";
}
#endif

/**
 * \brief Tensor Assign Expression(return by lazyAssign,
 * and evaluated by AssignEvaluate)
 */
template <typename LhsType, typename RhsType, class T>
class TensorAssignOp {
public:
  explicit TensorAssignOp(const LhsType& lhs, const RhsType& rhs)
      : lhs_(lhs), rhs_(rhs) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
    CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
    CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());
#endif
  }

  INLINE void apply(const int i, const int j) {
    lhs_.applyRef(i, j) = rhs_.apply(i, j);
  }
  INLINE void apply(const int index) {
    lhs_.applyRef(index) = rhs_.apply(index);
  }

  INLINE size_t getWidth() const { return lhs_.getWidth(); }
  INLINE size_t getHeight() const { return rhs_.getHeight(); }
  INLINE bool isContiguous() const {
    return lhs_.isContiguous() && rhs_.isContiguous();
  }
  INLINE bool useGpu() const { return lhs_.useGpu(); }

private:
  TensorApply<LhsType, T> lhs_;
  TensorApply<const RhsType, T> rhs_;
};

template <typename Assign, typename... AssignOp>
void AssignCpuEvaluate(int height,
                       int width,
                       bool isContiguous,
                       Assign&& assign,
                       AssignOp&&... args) {
  if (isContiguous) {
    int size = height * width;
    for (int index = 0; index < size; index++) {
      assign.apply(index);
      __attribute__((unused)) int dummy[] = {(((args)).apply(index), 0)...};
    }
  } else {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        assign.apply(i, j);
        __attribute__((unused)) int dummy[] = {(((args)).apply(i, j), 0)...};
      }
    }
  }
}

#ifdef __NVCC__
template <typename Assign, typename... AssignOp>
__global__ void AssignGpuEvaluate1(const int border,
                                   Assign assign,
                                   AssignOp... args) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    assign.apply(idx);
    __attribute__((unused)) int dummy[] = {(((args)).apply(idx), 0)...};
  }
}

template <typename Assign, typename... AssignOp>
__global__ void AssignGpuEvaluate2(const int height,
                                   const int width,
                                   Assign assign,
                                   AssignOp... args) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < height; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < width; j += gridDim.x * blockDim.x) {
      assign.apply(i, j);
      __attribute__((unused)) int dummy[] = {(((args)).apply(i, j), 0)...};
    }
  }
}
#endif

/**
 * \brief Evaluate one or more TensorAssignOp objects.
 *
 * \note At least one assignment expression is required
 */
template <typename Assign, typename... AssignOp>
void AssignEvaluate(Assign&& assign, AssignOp&&... args) {
  const bool useGpu_ = assign.useGpu();
  bool isContiguous_ = assign.isContiguous();
  const size_t height = assign.getHeight();
  const size_t width = assign.getWidth();

  const int packSize = sizeof...(args);
  const bool packUseGpu[] = {((args)).useGpu()...};
  const bool packIsContiguous[] = {((args)).isContiguous()...};
  const size_t packHeight[] = {((args)).getHeight()...};
  const size_t packWidth[] = {((args)).getWidth()...};

  for (int i = 0; i < packSize; i++) {
    CHECK_EQ(useGpu_, packUseGpu[i]);
    CHECK_EQ(height, packHeight[i]);
    CHECK_EQ(width, packWidth[i]);
    isContiguous_ = isContiguous_ && packIsContiguous[i];
  }

  if (useGpu_) {
#ifdef __NVCC__
    if (isContiguous_) {
      int size = height * width;
      int blockSize = size <= 1024 ? size : 1024;
      int gridSize = (size + 1024 - 1) / 1024;
      AssignGpuEvaluate1<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
          size, assign, args...);
    } else {
      int blockSizeY = std::min(32, (int)height);
      int blockSizeX = (32 / blockSizeY) * 32;
      int gridSizeX = std::min(32, (int)(width + blockSizeX - 1) / blockSizeX);
      int gridSizeY = std::min(32, (int)(height + blockSizeY - 1) / blockSizeY);
      dim3 threads(blockSizeX, blockSizeY);
      dim3 grid(gridSizeX, gridSizeY);
      AssignGpuEvaluate2<<<grid, threads, 0, STREAM_DEFAULT>>>(
          height, width, assign, args...);
    }

    CHECK_SYNC("AssignEvaluate failed");
#endif
  } else {
    AssignCpuEvaluate(height, width, isContiguous_, assign, args...);
  }
}

}  // namespace mypaddle
}  // namespace bubblefs