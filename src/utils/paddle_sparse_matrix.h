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

// Paddle/paddle/math/CpuSparseMatrix.h
// Paddle/paddle/math/CpuSparseMatrix.cpp
// Paddle/paddle/math/SparseMatrix.h
// Paddle/paddle/math/SparseMatrix.cpp
// Paddle/paddle/math/SparseRowMatrix.h
// Paddle/paddle/math/SparseRowMatrix.cpp

#pragma once

#include <string.h>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <vector>
#include "utils/paddle_matrix.h"
#include "utils/paddle_memory_handle.h"
#include "utils/paddle_simd_functions.h"
#include "utils/paddle_thread.h"

#include "hl_gpu.h"
#include "hl_top_k.h"

namespace bubblefs {
namespace mypaddle {

class CpuSparseMatrix : public Matrix {
public:
  CpuSparseMatrix(size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format = SPARSE_CSR,
                  bool trans = false);

  CpuSparseMatrix(CpuMemHandlePtr memHandle,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  CpuSparseMatrix(real* data,
                  int* rows,
                  int* cols,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  ~CpuSparseMatrix() {}

  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format);
  void resize(size_t newHeight, size_t newWidth);

  MatrixPtr getTranspose();

  SparseValueType getValueType();

  real* getRowValues(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return value_ + rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  int* getRowCols(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return cols_ + rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  /// fill row indices of each value in CSR matrix
  void fillRowIndices(IVectorPtr& outVec) const;

  size_t getColNum(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return rows_[i + 1] - rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  real* getColumn(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return value_ + cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  size_t getColStartIdx(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  size_t getRowStartIdx(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  size_t getRowNum(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return cols_[i + 1] - cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  virtual real getSum() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return elementCnt_;
    }
    double sum = 0;
    for (size_t i = 0; i < elementCnt_; ++i) {
      sum += value_[i];
    }
    return sum;
  }

  virtual void square2() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return;
    }
    for (size_t i = 0; i < elementCnt_; ++i) {
      value_[i] = value_[i] * value_[i];
    }
  }

  /**
   * only consider nonzero values.
   * the actual min value should compare with 0.0.
   */
  virtual real getMin() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return (elementCnt_ > 0 ? 1.0 : 0.0);
    }
    real min = value_[0];
    for (size_t i = 1; i < elementCnt_; ++i) {
      min = value_[i] < min ? value_[i] : min;
    }
    return min;
  }

  /**
   * only consider nonzero values.
   * the actual max value should compare with 0.0.
   */
  virtual real getMax() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return (elementCnt_ > 0 ? 1.0 : 0.0);
    }
    real max = value_[0];
    for (size_t i = 1; i < elementCnt_; ++i) {
      max = value_[i] > max ? value_[i] : max;
    }
    return max;
  }

  void rowMax(IVector& maxIds, Matrix& maxVal);
  int* getRows() const { return rows_; }
  int* getCols() const { return cols_; }
  real* getValue() const { return value_; }
  SparseFormat getFormat() const { return format_; }
  SparseValueType getValueType() const { return valueType_; }

  /**
   * @brief return value_ of sparse matrix
   *
   * Some times CpuSparseMatrix maybe Matrix,
   * if getValue, must dynamic_cast to CpuSparseMatrix,
   * getData is convenient to get value
   */
  real* getData() { return getValue(); }
  const real* getData() const { return getValue(); }

  /**
   * @brief only set value_ of FLOAT_VALUE sparse matrix to zero
   */
  void zeroMem();

  /// mem MUST be alloced outside (memAlloc=false)
  void transpose(MatrixPtr& matTrans, bool memAlloc);

  void mul(const Matrix& A, const Matrix& B, real alpha, real beta);

  /**
   * @brief sparseMatrix += denseMatrix
   *
   *  Named add3 just because add/add2 has been used in BaseMatrix.cu
   *  and they are not virtual function.
   *
   *  Only add value of same (row, col) index in dense matrix
   *  and do not use others values whoes postions are not in sparse matirx.
   *
   * @param[in]  b   dense matrix
   */
  void add3(CpuMatrix* b);
  void add3(MatrixPtr b);

  /**
   * @brief sparseMatrix[i,j] += bias[j], (j is the col index of sparse matrix)
   *
   * @param[in]  b      bias, dense matrix and height = 1
   * @param[in]  scale  scale of b
   */
  void addBias(Matrix& b, real scale);

  void print(std::ostream& os) const;

  void printOneRow(std::ostream& os, size_t idx) const;

  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values);

  /**
   * @brief this_row = b_row * c_row[cCol]
   *
   * @param[in]  cCol   the column of matrix c used to scale each row of b
   * @param[in]  b      CpuSparseMatrix
   * @param[in]  c      Matrix
   */
  void rowScale(size_t cCol, CpuSparseMatrix& b, Matrix& c);

  void randomizeUniform();

  void copyFrom(const GpuSparseMatrix& src, hl_stream_t stream);

  void copyFrom(const Matrix& src, hl_stream_t stream = HPPL_STREAM_DEFAULT);

  void copyFrom(const Matrix& src);

  /**
   * Get a temporary matrix. This is threadsafe. It should be only used
   * temporarily, i.e. do not store it or use it as return value.
   *
   * @note  Do NOT use large amount of tmp matrix.
   */
  CpuSparseMatrixPtr getTmpSparseMatrix(size_t height, size_t width);

  virtual MatrixPtr subMatrix(size_t startRow, size_t numRows);

  void copyFrom(std::vector<int>& rows,
                std::vector<int>& cols,
                std::vector<real>& values);

  void copyFrom(const CpuMatrix& src);

  void copyFrom(const CpuSparseMatrix& src);

  // trim the large size
  void trimFrom(const CpuSparseMatrix& src);

  void copyRow(int offsets, size_t colNum, const sparse_non_value_t* row);

  void copyRow(int offsets, size_t colNum, const sparse_float_value_t* row);

  template <class T>
  void copyFrom(int64_t* ids, int64_t* indices, T* data);

  template <class T>
  void copyFrom(int64_t* indices, T* data);

  void copyFrom(const real* data, size_t len) {
    LOG(FATAL) << "not supported!";
  }

private:
  MatrixPtr clone(size_t height = 0, size_t width = 0, bool useGpu = false);

protected:
  void sparseResize();
  /*for csr , record row start position, for csc, record row index for every no
   * zero value*/
  int* rows_;
  /*for csc , record col start position, for csr, record col index for every no
   * zero value*/
  int* cols_;
  real* value_;               /*nonzero value*/
  SparseFormat format_;       /* matrix format */
  SparseValueType valueType_; /*with value or not  */
  static const size_t DEFAULT_AVG_WIDTH = 20;

  static ThreadLocal<std::vector<CpuSparseMatrixPtr>> cpuLocalMats_;

  // BaseMatrixT interface
public:
  bool isSparse() const { return true; }

private:
  using Matrix::mul;
  using Matrix::copyFrom;
  using Matrix::rowMax;
  using Matrix::print;
  using Matrix::subMatrix;
};

typedef std::shared_ptr<_hl_sparse_matrix_s> hl_sparse_matrix_s_ptr;

class GpuSparseMatrix : public Matrix {
public:
  MemoryHandlePtr sMemoryHandle_;
  int* rows_;
  int* cols_;
  real* value_;
  const char* end_; /* point to the end of sMemoryHandle_ */

  hl_sparse_matrix_s_ptr sMatrix_;
  SparseValueType valueType_;
  SparseFormat format_;

public:
  GpuSparseMatrix(size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format_ = SPARSE_CSR,
                  bool trans = false);

  GpuSparseMatrix(GpuMemHandlePtr dataHandle,
                  hl_sparse_matrix_s_ptr sMatrix,
                  size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format_ = SPARSE_CSR,
                  bool trans = false,
                  MemoryHandlePtr sMemoryHandle = NULL);

  GpuSparseMatrix(real* value,
                  int* rows,
                  int* cols,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  GpuSparseMatrix(hl_sparse_matrix_s_ptr sMatrix,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans,
                  MemoryHandlePtr sMemoryHandle);

protected:
  struct Element {
    int row;
    int col;
    real val;
    Element(int rowIn, int colIn, real valIn)
        : row(rowIn), col(colIn), val(valIn) {}
  };

public:
  ~GpuSparseMatrix() {}

  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format);

  void resize(size_t newHeight, size_t newWidth);

  void sparseResizeCSR();

  void sparseResizeCSC();

  void resizeCSR(size_t newHeight,
                 size_t newWidth,
                 size_t newNnz,
                 SparseValueType valueType);

  void resizeCSC(size_t newHeight,
                 size_t newWidth,
                 size_t newNnz,
                 SparseValueType valueType);

  void mul(const GpuMatrix& a, const GpuMatrix& b, real scaleAB, real scaleT);
  /// B = A , B.trans = !A.trans
  MatrixPtr getTranspose();

  /// B = A'
  void transpose(MatrixPtr& matTrans, bool memAlloc);

  void copyFrom(const Matrix& src);
  void copyFrom(const Matrix& src, hl_stream_t stream);
  void copyFromCSR(CpuSparseMatrix& src, hl_stream_t stream);
  void copyFromCSC(CpuSparseMatrix& src, hl_stream_t stream);

  void copyFrom(const IVector& src) { LOG(FATAL) << "not implemented"; }
  void copyFrom(const IVector& src, hl_stream_t stream) {
    LOG(FATAL) << "not implemented";
  }

  template <class T>
  void copyFrom(int64_t* ids, int64_t* indices, T* data, hl_stream_t stream);

  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values);
  SparseValueType getValueType() const;
  SparseFormat getFormat() const { return format_; }

  const int* getRowCols(size_t x) const { return cols_ + rows_[x]; }
  const real* getRowValues(size_t x) const { return value_ + rows_[x]; }
  size_t getColNum(size_t x) const { return rows_[x + 1] - rows_[x]; }
  void print(std::ostream& os) const;

  /**
   * @brief only set value_ of FLOAT_VALUE sparse matrix to zero
   */
  void zeroMem();

  /**
   * @brief sparseMatrix += denseMatrix
   *
   * Named add3 just because add/add2 has been used in BaseMatrix.cu
   * and they are not virtual function.
   *
   * Only add value of same (row, col) index in dense matrix
   * and do not use others values.
   *
   * @param[in]  b   dense matrix
   */
  void add3(GpuMatrix* b);
  void add3(MatrixPtr b);

  /**
   * @brief sparseMatrix[i,j] += bias[j], (j is the col index of sparse matrix)
   *
   * @param[in]  b      bias, dense matrix and height = 1
   * @param[in]  scale  scale of b
   */
  void addBias(Matrix& b, real scale);

  /**
   * @brief return rows, which is gpu address
   */
  int* getRows() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_rows(sMatrix_.get());
  }

  /**
   * @brief return cols, which is gpu address
   */
  int* getCols() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_cols(sMatrix_.get());
  }

  /**
   * @brief return value, which is gpu address
   */
  real* getValue() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_value(sMatrix_.get());
  }

  /**
   * @brief return value_ of sparse matrix
   *
   * Some times CpuSparseMatrix maybe Matrix,
   * if getValue, must dynamic_cast to CpuSparseMatrix,
   * getData is convenient to get value
   */
  real* getData() { return getValue(); }
  const real* getData() const { return getValue(); }

  /**
   * @brief  Get top k value of each row in sparse matrix.
   *
   * Store the value in maxVal and theirs index in maxIds.
   * k = maxVal.width
   *
   * @param[out]  maxIds    index of top k
   * @param[out]  maxVal    value of top k
   */
  void rowMax(IVector& maxIds, Matrix& maxVal);

protected:
  void sparseResize();

  void copyRow(int offsets, size_t colNum, const sparse_non_value_t* row);
  void copyRow(int offsets, size_t colNum, const sparse_float_value_t* row);

public:
  void mul(const Matrix& a, const Matrix& b, real scaleAB, real scaleT);

  void copyFrom(CpuSparseMatrix& src, hl_stream_t stream);
  void copyFrom(GpuSparseMatrix& src, hl_stream_t stream);

  void trimFrom(const CpuSparseMatrix& src);
  void trimFromCSR(const CpuSparseMatrix& src);
  void trimFromCSC(const CpuSparseMatrix& src);

  // BaseMatrixT interface
public:
  bool isSparse() const { return true; }

private:
  using Matrix::mul;
  using Matrix::copyFrom;
  using Matrix::rowMax;
  using Matrix::print;
  using Matrix::subMatrix;
};

/**
 * Sparse Row
 */
class SparseRowCpuMatrix : public CpuMatrix {
public:
  struct IndexDict {
    // In the following, global id means the row id in the original matrix.
    // Local id means the row id in the local storage which only contains
    // the sparse rows.
    std::vector<unsigned int> localIndices;   // local id -> global id
    std::vector<unsigned int> globalIndices;  // global id -> local id
  };
  typedef std::shared_ptr<IndexDict> IndexDictPtr;

  /// heightStore is max number of rows of the sparse matrix.
  SparseRowCpuMatrix(CpuMemHandlePtr dataHandle,
                     size_t height,
                     size_t width,
                     IndexDictPtr indexDictHandle = nullptr,
                     bool trans = false)
      : CpuMatrix(nullptr, height, width, trans),
        indexDictHandle_(indexDictHandle) {
    init(height, width);
    buf_.reset(new RowBuffer(dataHandle, width));
  }

  virtual ~SparseRowCpuMatrix() {}

public:
  /**
   *  Get the row buf
   *
   *  @param row row id in the original matrix
   */
  real* getRow(size_t row) {
    CHECK_NE(globalIndices_[row], kUnusedId_);
    return getLocalRow(globalIndices_[row]);
  }

  /**
   *  Get the row buf
   *
   *  @param row row id in local storage
   */
  real* getLocalRow(size_t row) { return buf_->getWithAutoGrowth(row); }

  /**
   *  reserve the storage for rows according to current size of
   * indexDictHandle.
   *
   *  This is only used when SparseRowCpuMatrix is constructed with
   *  indexDictHandle.
   */
  void reserveStore() { buf_->resize(localIndices_->size()); }

  // row is the row id in the original matrix
  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

  /**
   * Fill data according to row indexs added, setup indices inside.
   *
   * *src* and *size* are data and size of normal dense CpuMatrix.
   */
  virtual void copyFrom(const real* src, size_t size);
  virtual void zeroMem();

  /**
   * apply L1 to all sparse rows, should be apply after indices ready.
   */
  virtual void applyL1(real learningRate, real decayRate);

  void clearIndices() { clearRows(); }
  void zeroMemThread(size_t tid, size_t numThreads);

  /**
   *  value -= grad * learningRate,  this is gradient.
   *
   * If L1 decay set use L1, else if L2 set use L2, otherwise no decay atall.
   *
   * t0 is a int vector used by L1/L2 decay, size = height of parameter
   * matrix,
   * store the time that each weight row last updated.
   *
   * Time is batchId, currentTime is current batchId.
   *
   * While pass finished, caller should call this func one more time
   *  with (fini=true) to let weight decay catch up current time.
   */
  void sgdUpdate(BaseMatrix& value,
                 IVector& t0,
                 real learningRate,
                 int currentTime,
                 real decayRate,
                 bool useL1,
                 bool fini = false);

  /**
   *  merge rows in *this* to *dest* for designated thread
   *
   *  values add to *dest* matrix
   *
   *  ids occured in *this* append to *ids*
   *  filtered by  (id % numThreads == tid)
   */
  void addTo(BaseMatrix& dest,
             std::vector<uint32_t>& ids,
             size_t tid,
             size_t numThreads);

  /**
   *  the second version addTo(), *dest* is a SparseRowCpuMatrix.
   *
   *  The dest's indices should be setup already, addTo() will
   *  check src ids is exist in dest's indices.
   */
  void addTo(SparseRowCpuMatrix& dest, size_t tid, size_t numThreads);

  const IndexDictPtr& getIndexDictHandle() const { return indexDictHandle_; }

  /**
   *  check all local and global indices consistency
   */
  void checkIndices();
  /**
   *  check whether row *i* exist in indices
   */
  void checkIndex(size_t i) {
    size_t localId = globalIndices_[i];
    CHECK_LT(localId, localIndices_->size());
    CHECK_EQ((*localIndices_)[localId], i);
  }

  std::vector<unsigned int>& getLocalIndices() const {
    return indexDictHandle_->localIndices;
  }

protected:
  template <typename Func>
  void apply(Func f) {
    f(buf_->data(), localIndices_->size() * width_);
  }

  void init(size_t height, size_t width);

  /// clear row indices.
  void clearRows() {
    for (auto id : *localIndices_) {
      globalIndices_[id] = kUnusedId_;
    }
    localIndices_->clear();
    buf_->clear();
  }

  inline void checkStoreSize() {
    if (buf_->isAutoGrowth()) {
      if (buf_->getRowCount() > 0.5 * height_) {
        LOG(WARNING) << "There are more than 0.5*height ("
                     << localIndices_->size() << ") rows are used for sparse "
                     << "update, which is not efficient. Considering not use "
                     << "sparse_update.";
      }
    } else {
      CHECK_LE(localIndices_->size(), buf_->getRowCount());
    }
  }

  std::unique_ptr<RowBuffer> buf_;
  IndexDictPtr indexDictHandle_;
  std::vector<unsigned int>* localIndices_;  // =&indexDictHandle_->localIndices
  unsigned int* globalIndices_;  // =indexDictHandle_->globalIndices.data();
  static const unsigned int kUnusedId_;
};

class SyncThreadPool;

/// For prefetching parameters from remote Parameter server
class SparsePrefetchRowCpuMatrix : public SparseRowCpuMatrix {
public:
  SparsePrefetchRowCpuMatrix(CpuMemHandlePtr dataHandle,
                             size_t height,
                             size_t width,
                             IndexDictPtr indexDictHandle = nullptr,
                             SyncThreadPool* pool = nullptr,
                             bool trans = false)
      : SparseRowCpuMatrix(dataHandle, height, width, indexDictHandle, trans),
        pool_(pool) {}

  /**
   * Extract feature ids from *input*, to fill row indexs.
   *
   * *input* must be sparse matrix.
   *
   * Can call many times before setup.
   */
  void addRows(MatrixPtr input);
  void addRows(IVectorPtr ids);

  /**
   * setup global indices of SparseRowMatrix after finish add rows.
   */
  void setupIndices();

protected:
  void addRows(const unsigned int* ids, size_t len);
  SyncThreadPool* pool_;
};

class SparseAutoGrowRowCpuMatrix : public SparseRowCpuMatrix {
public:
  SparseAutoGrowRowCpuMatrix(size_t height,
                             size_t width,
                             IndexDictPtr indexDictHandle = nullptr,
                             bool trans = false)
      : SparseRowCpuMatrix(nullptr, height, width, indexDictHandle, trans) {}

  real* getRow(size_t row) {
    auto id = globalIndices_[row];
    if (id == kUnusedId_) {
      id = globalIndices_[row] = localIndices_->size();
      localIndices_->push_back(row);
      checkStoreSize();
    }
    return getLocalRow(id);
  }

  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);
};

class CacheRowCpuMatrix : public SparseAutoGrowRowCpuMatrix {
public:
  CacheRowCpuMatrix(size_t height,
                    size_t width,
                    IndexDictPtr indexDictHandle = nullptr,
                    bool trans = false)
      : SparseAutoGrowRowCpuMatrix(height, width, indexDictHandle, trans),
        sourceData_(nullptr) {}

  void setSourceData(CpuVectorPtr sourceVec) {
    sourceDataVec_ = sourceVec;
    sourceData_ = sourceVec->getData();
  }

  real* getRow(size_t row) {
    auto id = globalIndices_[row];
    if (id == kUnusedId_) {
      id = globalIndices_[row] = localIndices_->size();
      localIndices_->push_back(row);
      checkStoreSize();
      memcpy(
          getLocalRow(id), sourceData_ + width_ * row, sizeof(float) * width_);
    }
    return getLocalRow(id);
  }

  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

public:
  CpuVectorPtr sourceDataVec_;
  real* sourceData_;
};

/**
 * Sparse Row Ids Matrix.
 *
 * mostly same as CpuMatrix, but maintain sparse row ids occured,
 * ids are hashed by worker thread id.
 */
class SparseRowIdsCpuMatrix : public CpuMatrix {
public:
  SparseRowIdsCpuMatrix(CpuMemHandlePtr dataHandle,
                        size_t height,
                        size_t width,
                        bool trans = false)
      : CpuMatrix(dataHandle, height, width, trans) {}

  void setNumOfThreads(size_t numOfThreads) { idsArray_.resize(numOfThreads); }

  std::vector<uint32_t>& getIds(size_t threadId) { return idsArray_[threadId]; }

private:
  std::vector<std::vector<uint32_t>> idsArray_;
};

const size_t CpuSparseMatrix::DEFAULT_AVG_WIDTH;

CpuSparseMatrix::CpuSparseMatrix(size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, false) {
  resize(height, width, nnz, valueType, format);
}

CpuSparseMatrix::CpuSparseMatrix(CpuMemHandlePtr dataHandle,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(dataHandle, height, width, trans, false) {
  resize(height, width, nnz, valueType, format);
}

CpuSparseMatrix::CpuSparseMatrix(real* data,
                                 int* rows,
                                 int* cols,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, false) {
  cols_ = cols;
  rows_ = rows;
  value_ = data;
  height_ = height;
  width_ = width;
  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;
}

void CpuSparseMatrix::resize(size_t newHeight,
                             size_t newWidth,
                             size_t newNnz,
                             SparseValueType valueType,
                             SparseFormat format) {
  CHECK_LE(newNnz, newHeight * newWidth);
  size_t newSize = 0;
  if (format == SPARSE_CSR) {
    newSize = (newHeight + 1) * sizeof(int) + newNnz * sizeof(int);
  } else {
    newSize = (newWidth + 1) * sizeof(int) + newNnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = format;
  sparseResize();
}
void CpuSparseMatrix::sparseResize() {
  if (format_ == SPARSE_CSR) {
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()));
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(memoryHandle_->getBuf()) +
          (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }
  } else {
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()));
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(memoryHandle_->getBuf()) +
          (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }
  }
}

void CpuSparseMatrix::resize(size_t newHeight, size_t newWidth) {
  resize(newHeight,
         newWidth,
         newHeight * std::min(DEFAULT_AVG_WIDTH, newWidth),
         valueType_,
         format_);
}

MatrixPtr CpuSparseMatrix::getTranspose() {
  if (!memoryHandle_ && !value_) {
    MatrixPtr dest(new CpuSparseMatrix(
        height_, width_, elementCnt_, valueType_, format_, true));
    return dest;
  } else if (memoryHandle_) {
    MatrixPtr dest(new CpuSparseMatrix(
        std::dynamic_pointer_cast<CpuMemoryHandle>(memoryHandle_),
        height_,
        width_,
        elementCnt_,
        valueType_,
        format_,
        true));
    return dest;
  } else if (value_) {
    MatrixPtr dest(new CpuSparseMatrix(value_,
                                       rows_,
                                       cols_,
                                       height_,
                                       width_,
                                       elementCnt_,
                                       valueType_,
                                       format_,
                                       true));
    return dest;
  } else {
    return NULL;
  }
}

SparseValueType CpuSparseMatrix::getValueType() { return valueType_; }

void CpuSparseMatrix::mul(const Matrix& a,
                          const Matrix& b,
                          real scaleAB,
                          real scaleT) {
  CHECK(!isTransposed()) << "Not supported";
  const auto a_ptr = dynamic_cast<const CpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const CpuMatrix*>(&b);

  if (a_ptr && b_ptr) {
    CpuMatrix::mul((CpuMatrix*)a_ptr, (CpuMatrix*)b_ptr, this, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::add3(CpuMatrix* b) {
  CHECK(getFormat() != SPARSE_CSC) << "Not supported";
  CHECK(height_ == b->getHeight());
  CHECK(width_ == b->getWidth());
  real* A = getValue();
  real* B = b->getData();
  int* cols = getCols();
  for (size_t i = 0; i < height_; i++) {
    size_t start = getRowStartIdx(i);
    size_t end = getRowStartIdx(i + 1);
    for (size_t j = start; j < end; j++) {
      A[j] = B[i * width_ + cols[j]];
    }
  }
}

void CpuSparseMatrix::add3(MatrixPtr b) {
  if (dynamic_cast<CpuMatrix*>(b.get())) {
    add3(dynamic_cast<CpuMatrix*>(b.get()));
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::addBias(Matrix& b, real scale) {
  CHECK_EQ(b.getHeight(), (size_t)1);
  CHECK_EQ(width_, b.getWidth());
  real* A = getValue();
  real* B = b.getData();
  int* cols = getCols();
  size_t nnz = getElementCnt();
  for (size_t i = 0; i < nnz; i++) {
    A[i] += scale * B[cols[i]];
  }
}

template <class T>
void printBuf(std::ostream& os, T* a, size_t len, const char* name) {
  os << "\n: " << name << " [";
  for (size_t i = 0; i < len; i++) {
    os << a[i] << " ";
  }
  os << "]\n";
}

void CpuSparseMatrix::print(std::ostream& os) const {
  size_t rowSize = format_ == SPARSE_CSC ? elementCnt_ : height_ + 1;
  size_t colSize = format_ == SPARSE_CSC ? width_ + 1 : elementCnt_;
  printBuf(os, rows_, rowSize, "row");
  printBuf(os, cols_, colSize, "col");
  if (valueType_ == FLOAT_VALUE) {
    printBuf(os, value_, elementCnt_, "value");
  }
  return;
}

void CpuSparseMatrix::printOneRow(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, height_);
  if (format_ == SPARSE_CSC) {
    LOG(FATAL) << "SPARSE_CSC not supported";
    return;
  }

  const int* col = getRowCols(idx);
  size_t num = getColNum(idx);
  if (num > 0) {
    if (valueType_ == FLOAT_VALUE) {
      const real* data = getRowValues(idx);
      os << col[0] << ":" << data[0];
      for (size_t i = 1; i < num; ++i) {
        os << " " << col[i] << ":" << data[i];
      }
    } else {
      os << col[0];
      for (size_t i = 1; i < num; ++i) {
        os << " " << col[i];
      }
    }
  }
  os << ";";
}

void CpuSparseMatrix::rowScale(size_t cCol, CpuSparseMatrix& b, Matrix& c) {
  CHECK(getFormat() != SPARSE_CSC) << "Not supported";
  CHECK_EQ(height_, b.getHeight());
  CHECK_EQ(width_, b.getWidth());
  real* A = getValue();
  real* B = b.getValue();
  if (b.getValueType() == FLOAT_VALUE) {
    for (size_t i = 0; i < height_; i++) {
      size_t start = getRowStartIdx(i);
      size_t end = getRowStartIdx(i + 1);
      CHECK_EQ(start, b.getRowStartIdx(i));
      CHECK_EQ(end, b.getRowStartIdx(i + 1));
      for (size_t j = start; j < end; j++) {
        A[j] = B[j] * c.getElement(i, cCol);
      }
    }
  } else if (b.getValueType() == NO_VALUE) {
    for (size_t i = 0; i < height_; i++) {
      size_t start = getRowStartIdx(i);
      size_t end = getRowStartIdx(i + 1);
      CHECK_EQ(start, b.getRowStartIdx(i));
      CHECK_EQ(end, b.getRowStartIdx(i + 1));
      for (size_t j = start; j < end; j++) {
        A[j] = c.getElement(i, cCol);
      }
    }
  }
}

void CpuSparseMatrix::randomizeUniform() {
  CHECK_LE(elementCnt_, height_ * width_);
  if (valueType_ == FLOAT_VALUE) {
    real* data = getValue();
    for (size_t i = 0; i < elementCnt_; ++i) {
      *data++ = rand() / static_cast<real>(RAND_MAX);  // NOLINT
    }
  }
  if (format_ == SPARSE_CSR) {
    sparseRand(rows_, cols_, elementCnt_, height_ + 1, width_, false);
  } else {
    sparseRand(cols_, rows_, elementCnt_, width_ + 1, height_, false);
  }
}

void CpuSparseMatrix::copyFrom(std::vector<int>& rows,
                               std::vector<int>& cols,
                               std::vector<real>& values) {
  size_t size = format_ == SPARSE_CSR ? cols.size() : rows.size();
  resize(height_, width_, size, valueType_, format_);
  if (valueType_ == FLOAT_VALUE) {
    memcpy(&value_[0], &values[0], sizeof(real) * values.size());
  }
  memcpy(&cols_[0], &cols[0], sizeof(int) * cols.size());
  memcpy(&rows_[0], &rows[0], sizeof(int) * rows.size());
}

// Copy from a CpuMatrix, only supported in sparse_float_value_t
// SparseMatrix.
void CpuSparseMatrix::copyFrom(const CpuMatrix& src) {
  CHECK_EQ(getHeight(), src.getHeight());
  CHECK_EQ(getWidth(), src.getWidth());
  CHECK(!src.trans_ && !trans_);
  if (format_ == SPARSE_CSR) {
    std::vector<int> rows(getHeight() + 1);
    std::vector<int> cols;
    std::vector<real> values;
    rows[0] = 0;
    for (size_t r = 0; r < getHeight(); ++r) {
      for (size_t c = 0; c < getWidth(); ++c) {
        real v = src.getElement(r, c);
        if (fabs(v) > FLT_EPSILON) {
          cols.push_back(c);
          values.push_back(v);
        }
      }
      rows[r + 1] = values.size();
    }
    copyFrom(rows, cols, values);
  } else {
    std::vector<int> cols(getWidth() + 1);
    std::vector<int> rows;
    std::vector<real> values;
    cols[0] = 0;
    for (size_t r = 0; r < getWidth(); ++r) {
      for (size_t c = 0; c < getHeight(); ++c) {
        real v = src.getElement(c, r);
        if (fabs(v) > FLT_EPSILON) {
          rows.push_back(c);
          values.push_back(v);
        }
      }
      cols[r + 1] = values.size();
    }
    copyFrom(rows, cols, values);
  }
}

MatrixPtr CpuSparseMatrix::clone(size_t height, size_t width, bool useGpu) {
  if (height == 0 && width == 0) {
    height = height_;
    width = width_;
  }
  CHECK(width && height);
  if (!useGpu) {
    return std::make_shared<CpuSparseMatrix>(
        height, width, 0, valueType_, format_);
  } else {
    return std::make_shared<GpuSparseMatrix>(
        height, width, elementCnt_, valueType_, format_);
  }
}

MatrixPtr CpuSparseMatrix::subMatrix(size_t startRow, size_t numRows) {
  CHECK_LE(startRow + numRows, height_);
  CHECK_EQ(format_, SPARSE_CSR);
  if (valueType_ == NO_VALUE) {
    return std::make_shared<CpuSparseMatrix>(
        nullptr,
        rows_ + startRow,
        cols_,
        numRows,
        width_,
        rows_[startRow + numRows] - rows_[startRow],
        valueType_,
        format_,
        trans_);
  } else {
    return std::make_shared<CpuSparseMatrix>(
        value_,
        rows_ + startRow,
        cols_,
        numRows,
        width_,
        rows_[startRow + numRows] - rows_[startRow],
        valueType_,
        format_,
        trans_);
  }
}

/* mem MUST be alloced outside (memAlloc=false) */
void CpuSparseMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  CHECK(!memAlloc);
  CpuSparseMatrix* mat = dynamic_cast<CpuSparseMatrix*>(matTrans.get());
  if (format_ == SPARSE_CSR) {
    /*statistic element number in each col*/
    int* colCounters = mat->getRows() + 1;
    memset(colCounters, 0, sizeof(int) * width_);
    for (size_t i = 0; i < elementCnt_; ++i) {
      int col = cols_[i];
      colCounters[col]++;
    }
    /*fill mat rows */
    mat->getRows()[0] = 0;
    for (size_t i = 1; i < width_ + 1; i++) {
      mat->getRows()[i] = mat->getRows()[i - 1] + mat->getRows()[i];
    }
    /*fill mat values and cols*/
    std::vector<int> colNumVec(width_, 0);
    if (valueType_ == FLOAT_VALUE) {
      for (size_t i = 0; i < height_; i++) {
        for (int j = rows_[i]; j < rows_[i + 1]; j++) {
          int colIdx = cols_[j];
          int index = mat->getRows()[colIdx] + colNumVec[colIdx];
          mat->getCols()[index] = i;
          mat->getValue()[index] = value_[j];
          colNumVec[colIdx]++;
        }
      }
    } else {
      for (size_t i = 0; i < height_; i++) {
        for (int j = rows_[i]; j < rows_[i + 1]; j++) {
          int colIdx = cols_[j];
          int index = mat->getRows()[colIdx] + colNumVec[colIdx];
          mat->getCols()[index] = i;
          colNumVec[colIdx]++;
        }
      }
    }
  } else {
    /*statistic element number in each row*/
    int* rowCounters = mat->getCols() + 1;
    memset(rowCounters, 0, sizeof(int) * height_);
    for (size_t i = 0; i < elementCnt_; ++i) {
      int row = rows_[i];
      rowCounters[row]++;
    }

    /*fill mat cols */
    mat->getCols()[0] = 0;
    for (size_t i = 1; i < height_ + 1; i++) {
      mat->getCols()[i] = mat->getCols()[i - 1] + mat->getCols()[i];
    }
    /*fill mat values and rows*/
    std::vector<int> rowNumVec(height_, 0);
    if (valueType_ == FLOAT_VALUE) {
      for (size_t i = 0; i < width_; i++) {
        for (int j = cols_[i]; j < cols_[i + 1]; j++) {
          int rowIdx = rows_[j];
          int index = mat->getCols()[rowIdx] + rowNumVec[rowIdx];
          mat->getRows()[index] = i;
          mat->getValue()[index] = value_[j];
          rowNumVec[rowIdx]++;
        }
      }
    } else {
      for (size_t i = 0; i < width_; i++) {
        for (int j = cols_[i]; j < cols_[i + 1]; j++) {
          int rowIdx = rows_[j];
          int index = mat->getCols()[rowIdx] + rowNumVec[rowIdx];
          mat->getRows()[index] = i;
          rowNumVec[rowIdx]++;
        }
      }
    }
  }
}

void CpuSparseMatrix::setRow(size_t row,
                             size_t colNum,
                             const unsigned int* cols,
                             const real* values) {
  if (format_ == SPARSE_CSR) {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    if (0 == row) {
      rows_[row] = 0;
    }
    rows_[row + 1] = rows_[row] + colNum;
    for (size_t i = 0; i < colNum; ++i) {
      cols_[rows_[row] + i] = cols[i];
    }
    if (valueType_ == NO_VALUE) {
      CHECK(!values);
    } else {
      for (size_t i = 0; i < colNum; ++i) {
        value_[rows_[row] + i] = values[i];
      }
    }
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::fillRowIndices(IVectorPtr& outVec) const {
  if (format_ == SPARSE_CSR) {
    auto nnz = getElementCnt();
    IVector::resizeOrCreate(outVec, nnz, false);
    auto out = outVec->getData();
    int* rows = getRows();
    for (size_t i = 0; i < height_; i++) {
      for (int j = rows[i]; j < rows[i + 1]; j++) {
        out[j] = i;
      }
    }
  } else {
    LOG(FATAL) << "SPARSE_CSC not supported";
  }
}

ThreadLocal<std::vector<CpuSparseMatrixPtr>> CpuSparseMatrix::cpuLocalMats_;

CpuSparseMatrixPtr CpuSparseMatrix::getTmpSparseMatrix(size_t height,
                                                       size_t width) {
  std::vector<CpuSparseMatrixPtr>* localMats = cpuLocalMats_.get();
  auto it = localMats->begin();
  while (it != localMats->end()) {
    if (it->unique()) {
      (*it)->resize(height, width, elementCnt_, valueType_, format_);
      return *it;
    }
  }
  localMats->emplace_back(std::make_shared<CpuSparseMatrix>(
      height, width, elementCnt_, valueType_, format_, false));
  return localMats->back();
}

void CpuSparseMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  if (dynamic_cast<const GpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const GpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc, stream);
  } else if (dynamic_cast<const CpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else if (dynamic_cast<const CpuMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else {
    LOG(FATAL) << "not implemented";
  }
}

void CpuSparseMatrix::copyFrom(const Matrix& src) {
  if (dynamic_cast<const CpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else if (dynamic_cast<const CpuMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else {
    LOG(FATAL) << "not implemented";
  }
}

void CpuSparseMatrix::copyFrom(const GpuSparseMatrix& src, hl_stream_t stream) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_EQ(width_, src.getWidth());
  CHECK_EQ(size_t(elementCnt_), src.getElementCnt());
  size_t valSize = valueType_ == NO_VALUE ? 0 : elementCnt_;
  if (format_ == SPARSE_CSC)
    hl_memcpy_from_csc_matrix(value_,
                              valSize,
                              rows_,
                              elementCnt_,
                              cols_,
                              width_ + 1,
                              src.sMatrix_.get(),
                              stream);
  else
    hl_memcpy_from_csr_matrix(value_,
                              valSize,
                              rows_,
                              height_ + 1,
                              cols_,
                              elementCnt_,
                              src.sMatrix_.get(),
                              stream);
}

void CpuSparseMatrix::copyFrom(const CpuSparseMatrix& src) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_EQ(width_, src.getWidth());
  CHECK_EQ(format_, src.getFormat());
  int start = format_ == SPARSE_CSR ? src.getRows()[0] : src.getCols()[0];
  if (format_ == SPARSE_CSR) {
    size_t totalColNum = 0;
    for (size_t i = 0; i < height_; ++i) {
      totalColNum += src.getColNum(i);
    }
    resize(height_, width_, totalColNum, valueType_, format_);
    rows_[0] = 0;
    for (size_t i = 0; i < height_; ++i) {
      rows_[i + 1] = rows_[i] + src.getColNum(i);
    }
    memcpy(cols_, src.getCols() + start, totalColNum * sizeof(int));
  } else {
    size_t totalColNum = 0;
    for (size_t i = 0; i < width_; ++i) {
      totalColNum += src.getRowNum(i);
    }
    resize(height_, width_, totalColNum, valueType_, format_);
    cols_[0] = 0;
    for (size_t i = 0; i < width_; ++i) {
      cols_[i + 1] = cols_[i] + src.getRowNum(i);
    }
    memcpy(rows_, src.getRows() + start, totalColNum * sizeof(int));
  }

  // if have different value type, only copy rows and cols
  if (valueType_ == FLOAT_VALUE && src.getValueType() == FLOAT_VALUE) {
    memcpy(value_, src.getValue() + start, elementCnt_ * sizeof(real));
  }
}

void CpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_non_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
  }
}

void CpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_float_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
    value_[offsets + j] = row[j].value;
  }
}

template <class T>
void CpuSparseMatrix::copyFrom(int64_t* ids, int64_t* indices, T* data) {
  size_t totalColNum = 0;
  for (size_t i = 0; i < height_; ++i) {
    int64_t id = ids[i];
    totalColNum += indices[id + 1] - indices[id];
  }
  valueType_ = typeid(T) == typeid(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE;

  resize(height_, width_, totalColNum, valueType_, format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; ++i) {
    int64_t id = ids[i];
    T* row = data + indices[id];
    size_t colNum = indices[id + 1] - indices[id];
    rows_[i + 1] = rows_[i] + colNum;
    copyRow(rows_[i], colNum, row);
  }
}

template <class T>
void CpuSparseMatrix::copyFrom(int64_t* indices, T* data) {
  CHECK(format_ == SPARSE_CSR);
  size_t totalColNum = indices[height_] - indices[0];
  valueType_ = typeid(T) == typeid(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE;
  resize(height_, width_, totalColNum, valueType_, format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; ++i) {
    T* row = data + indices[i];
    size_t colNum = indices[i + 1] - indices[i];
    rows_[i + 1] = rows_[i] + colNum;
    copyRow(rows_[i], colNum, row);
  }
}

void CpuSparseMatrix::trimFrom(const CpuSparseMatrix& src) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_LE(width_, src.getWidth());
  CHECK_EQ(format_, src.getFormat());
  CHECK_EQ(valueType_, src.getValueType());
  if (format_ == SPARSE_CSR) {
    int* srcCols = src.getCols();
    size_t numLessWidth =
        std::count_if(srcCols, srcCols + src.getElementCnt(), [this](size_t n) {
          return n < this->width_;
        });
    resize(height_, width_, numLessWidth, valueType_, format_);
    rows_[0] = 0;
    size_t index = 0;
    for (size_t r = 0; r < height_; ++r) {
      for (int i = src.getRows()[r]; i < src.getRows()[r + 1]; ++i) {
        if (srcCols[i] < static_cast<int>(width_)) {
          cols_[index] = srcCols[i];
          if (valueType_ == FLOAT_VALUE) {
            value_[index] = src.getValue()[i];
          }
          ++index;
        }
      }
      rows_[r + 1] = index;
    }
    CHECK_EQ(index, numLessWidth);
  } else {
    size_t numLessWidth = src.getCols()[width_] - src.getCols()[0];
    resize(height_, width_, numLessWidth, valueType_, format_);
    cols_[0] = 0;
    size_t index = 0;
    // note: c < width_, not src.getWidth();
    for (size_t c = 0; c < width_; ++c) {
      for (int i = src.getCols()[c]; i < src.getCols()[c + 1]; ++i) {
        rows_[index] = src.getRows()[i];
        if (valueType_ == FLOAT_VALUE) {
          value_[index] = src.getValue()[i];
        }
        ++index;
      }
      cols_[c + 1] = index;
    }
    CHECK_EQ(index, numLessWidth);
  }
}

void CpuSparseMatrix::zeroMem() {
  CHECK(valueType_ == FLOAT_VALUE);
  memset(value_, 0, elementCnt_ * sizeof(real));
}

template void CpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_non_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_float_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* indices,
                                        sparse_non_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* indices,
                                        sparse_float_value_t* data);

void CpuSparseMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  maxVal.zeroMem();
  int* outids = maxIds.getData();
  real* outvalues = maxVal.getData();

  typedef std::pair<real, size_t> valuepair;
  std::vector<valuepair> vec;
  for (size_t i = 0; i < numSamples; i++) {
    vec.clear();

    auto num = getColNum(i);
    auto ids = getRowCols(i);
    auto values = getRowValues(i);
    for (size_t j = 0; j < num; j++) {
      vec.push_back(std::make_pair(values[j], ids[j]));
    }

    size_t outsize = std::min(num, beam);
    std::partial_sort(vec.begin(),
                      vec.begin() + outsize,
                      vec.end(),
                      [](const valuepair& a, const valuepair& b) {
                        return a.first > b.first;
                      });
    for (size_t j = 0; j < outsize; j++) {
      outids[i * beam + j] = vec[j].second;
      outvalues[i * beam + j] = vec[j].first;
    }
    if (outsize < beam) {
      // if the number of values to sort are less than the output size,
      // use -1 to indicate the end of valid sorted values.
      outids[i * beam + outsize] = -1;
    }
  }
}

GpuSparseMatrix::GpuSparseMatrix(size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, true) {
  resize(height, width, nnz, valueType, format);
}

GpuSparseMatrix::GpuSparseMatrix(GpuMemHandlePtr dataHandle,
                                 hl_sparse_matrix_s_ptr sMatrix,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans,
                                 MemoryHandlePtr sMemoryHandle)
    : Matrix(dataHandle, height, width, trans, true) {
  CHECK(dataHandle && sMatrix) << "Invalid argument pointer";

  size_t size = 0;
  if (format == SPARSE_CSR) {
    size = (height + 1) * sizeof(int) + nnz * sizeof(int);
  } else {
    size = (width + 1) * sizeof(int) + nnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    size += nnz * sizeof(real);
  }
  CHECK_LE(size, dataHandle->getSize());

  sMatrix_ = sMatrix;

  if (sMemoryHandle == NULL) {
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(dataHandle->getSize());
  } else {
    CHECK_EQ(sMemoryHandle->getSize(), dataHandle->getSize());
    sMemoryHandle_ = sMemoryHandle;
  }

  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;
  if (format_ == SPARSE_CSR)
    sparseResizeCSR();
  else
    sparseResizeCSC();
}

GpuSparseMatrix::GpuSparseMatrix(hl_sparse_matrix_s_ptr sMatrix,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans,
                                 MemoryHandlePtr sMemoryHandle)
    : Matrix(NULL, height, width, trans, true) {
  CHECK(sMatrix) << "Invalid argument pointer";
  sMatrix_ = sMatrix;
  sMemoryHandle_ = sMemoryHandle;
  elementCnt_ = nnz;
  format_ = format;
  valueType_ = valueType;
}

GpuSparseMatrix::GpuSparseMatrix(real* value,
                                 int* rows,
                                 int* cols,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, true) {
  size_t size = 0;
  if (format == SPARSE_CSR) {
    size = (height + 1) * sizeof(int) + nnz * sizeof(int);
  } else {
    size = (width + 1) * sizeof(int) + nnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    size += nnz * sizeof(real);
  }
  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;

  sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(size);
  if (format_ == SPARSE_CSR) {
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
          (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }

    if (sMatrix_ == NULL) {
      /* construct hl_sparse_matrix_s */
      hl_sparse_matrix_s tmp;
      hl_construct_sparse_matrix(
          &tmp,
          value,
          rows,
          cols,
          HL_SPARSE_CSR,
          valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
          height_,
          width_,
          elementCnt_);
      hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
      sMatrix_ = tmp2;
    }

  } else {
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
          (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }

    if (sMatrix_ == NULL) {
      /* construct hl_sparse_matrix_s */
      hl_sparse_matrix_s tmp;
      hl_construct_sparse_matrix(
          &tmp,
          value,
          rows,
          cols,
          HL_SPARSE_CSC,
          valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
          height_,
          width_,
          elementCnt_);
      hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
      sMatrix_ = tmp2;
    }
  }
}

void GpuSparseMatrix::sparseResizeCSR() {
  rows_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
  cols_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
                             (height_ + 1) * sizeof(int));
  if (NO_VALUE != valueType_) {
    value_ = reinterpret_cast<real*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
  } else {
    value_ = NULL;
  }

  if (sMatrix_ == NULL) {
    /* construct hl_sparse_matrix_s */
    hl_sparse_matrix_s tmp;
    hl_construct_sparse_matrix(
        &tmp,
        data_,
        memoryHandle_->getSize(),
        HL_SPARSE_CSR,
        valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
        height_,
        width_,
        elementCnt_);
    hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
    sMatrix_ = tmp2;
  }
}

void GpuSparseMatrix::sparseResizeCSC() {
  cols_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
  rows_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
                             (width_ + 1) * sizeof(int));
  if (NO_VALUE != valueType_) {
    value_ = reinterpret_cast<real*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
  } else {
    value_ = NULL;
  }

  if (sMatrix_ == NULL) {
    /* construct hl_sparse_matrix_s */
    hl_sparse_matrix_s tmp;
    hl_construct_sparse_matrix(
        &tmp,
        memoryHandle_->getBuf(),
        memoryHandle_->getSize(),
        HL_SPARSE_CSC,
        valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
        height_,
        width_,
        elementCnt_);
    hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
    sMatrix_ = tmp2;
  }
}

void GpuSparseMatrix::resize(size_t newHeight,
                             size_t newWidth,
                             size_t newNnz,
                             SparseValueType valueType,
                             SparseFormat format) {
  if (format == SPARSE_CSR) {
    resizeCSR(newHeight, newWidth, newNnz, valueType);
  } else {
    resizeCSC(newHeight, newWidth, newNnz, valueType);
  }
}

void GpuSparseMatrix::resizeCSR(size_t newHeight,
                                size_t newWidth,
                                size_t newNnz,
                                SparseValueType valueType) {
  size_t newSize = (newHeight + 1) * sizeof(int) + newNnz * sizeof(int);
  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<GpuMemoryHandle>(newSize);
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
    end_ = reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
           sMemoryHandle_->getSize();
    sMatrix_ = NULL;
  } else if (valueType != valueType_) {
    sMatrix_ = NULL;
  } else {
    /*
     * newNnz > elementCnt_ is necessary for the following condition:
     * Firstly, height_ is 9 elementCnt_ is 56
     * Secondly, height_ is 11 elementCnt_ is 44
     *   ==> height_ is bigger, sMatrix_ will resize, and total item is 44 now
     * Then, height_ is 10 elementCnt_ is 52
     *   ==> Without newNnz > elementCnt_ condition, sMatrix_ will fail
     */
    if ((ssize_t)((newHeight + 1) * sizeof(int)) >
            ((char*)cols_ - (char*)rows_) ||
        newNnz > static_cast<size_t>(sMatrix_->nnz)) {
      sMatrix_ = NULL;
    } else if (NO_VALUE == valueType) {
      if ((ssize_t)(newNnz * sizeof(int)) > (end_ - (char*)cols_)) {
        sMatrix_ = NULL;
      }
    } else {
      if ((ssize_t)(newNnz * sizeof(int)) > ((char*)value_ - (char*)cols_) ||
          (ssize_t)(newNnz * sizeof(real)) > (end_ - (char*)value_)) {
        sMatrix_ = NULL;
      }
    }
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = SPARSE_CSR;

  if (sMatrix_ == NULL) {
    sparseResizeCSR();
  }
}

void GpuSparseMatrix::resizeCSC(size_t newHeight,
                                size_t newWidth,
                                size_t newNnz,
                                SparseValueType valueType) {
  size_t newSize = (newWidth + 1) * sizeof(int) + newNnz * sizeof(int);
  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<GpuMemoryHandle>(newSize);
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
    end_ = reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
           sMemoryHandle_->getSize();
    sMatrix_ = NULL;
  } else if (valueType != valueType_) {
    sMatrix_ = NULL;
  } else {
    /*
     * newNnz > elementCnt_ is necessary for the following condition:
     * Firstly, height_ is 9 elementCnt_ is 56
     * Secondly, height_ is 11 elementCnt_ is 44
     *   ==> height_ is bigger, sMatrix_ will resize,
     *       and total item is 44 now
     * Then, height_ is 10 elementCnt_ is 52
     *   ==> Without newNnz > elementCnt_ condition, sMatrix_ will fail
     */
    if ((ssize_t)((newWidth + 1) * sizeof(int)) >
            ((char*)rows_ - (char*)cols_) ||
        newNnz > static_cast<size_t>(sMatrix_->nnz)) {
      sMatrix_ = NULL;
    } else if (NO_VALUE == valueType) {
      if ((ssize_t)(newNnz * sizeof(int)) > (end_ - (char*)rows_)) {
        sMatrix_ = NULL;
      }
    } else {
      if ((ssize_t)(newNnz * sizeof(int)) > ((char*)value_ - (char*)rows_) ||
          (ssize_t)(newNnz * sizeof(real)) > (end_ - (char*)value_)) {
        sMatrix_ = NULL;
      }
    }
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = SPARSE_CSC;

  if (sMatrix_ == NULL) {
    sparseResizeCSC();
  }
}

void GpuSparseMatrix::resize(size_t newHeight, size_t newWidth) {
  resize(newHeight, newWidth, elementCnt_, valueType_, format_);
}

MatrixPtr GpuSparseMatrix::getTranspose() {
  CHECK(memoryHandle_.get() || sMatrix_) << "not supported";
  if (memoryHandle_.get()) {
    MatrixPtr copy_T(new GpuSparseMatrix(
        std::dynamic_pointer_cast<GpuMemoryHandle>(memoryHandle_),
        sMatrix_,
        height_,
        width_,
        elementCnt_,
        valueType_,
        format_,
        true,
        sMemoryHandle_));
    return copy_T;
  } else {
    MatrixPtr copy_T(new GpuSparseMatrix(sMatrix_,
                                         height_,
                                         width_,
                                         elementCnt_,
                                         valueType_,
                                         format_,
                                         true,
                                         sMemoryHandle_));
    return copy_T;
  }
}

void GpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_non_value_t* row) {
  memcpy(cols_ + offsets, row, sizeof(int) * colNum);
}

void GpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_float_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
    value_[offsets + j] = row[j].value;
  }
}

void GpuSparseMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  if (auto mat = dynamic_cast<const CpuSparseMatrix*>(&src)) {
    copyFrom(*(const_cast<CpuSparseMatrix*>(mat)), stream);
  } else if (auto mat = dynamic_cast<const GpuSparseMatrix*>(&src)) {
    copyFrom(*(const_cast<GpuSparseMatrix*>(mat)), stream);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

void GpuSparseMatrix::copyFrom(const Matrix& src) {
  copyFrom(src, HPPL_STREAM_1);
  hl_stream_synchronize(HPPL_STREAM_1);
}

template <class T>
void GpuSparseMatrix::copyFrom(int64_t* ids,
                               int64_t* indices,
                               T* data,
                               hl_stream_t stream) {
  CHECK_EQ(format_, SPARSE_CSR);
  size_t nnz = 0;
  for (size_t i = 0; i < height_; i++) {
    int64_t id = ids[i];
    nnz += indices[id + 1] - indices[id];
  }

  resize(height_,
         width_,
         nnz,
         sizeof(T) == sizeof(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE,
         format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; i++) {
    int64_t id = ids[i];
    size_t colNum = indices[id + 1] - indices[id];
    rows_[i + 1] = rows_[i] + colNum;

    T* row = data + indices[id];
    copyRow(rows_[i], colNum, row);
  }

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;
  hl_memcpy_csr_matrix(sMatrix_.get(), value_, rows_, cols_, stream);
}

void GpuSparseMatrix::setRow(size_t row,
                             size_t colNum,
                             const unsigned int* cols,
                             const real* values) {
  CHECK_EQ(format_, SPARSE_CSR);
  if (NO_VALUE == valueType_) {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    CHECK(NULL == values);
  } else {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    CHECK(NULL != values);
  }
  if (0 == row) {
    rows_[row] = 0;
  }
  rows_[row + 1] = rows_[row] + colNum;

  memcpy(cols_ + rows_[row], cols, sizeof(*cols) * colNum);
  if (FLOAT_VALUE == valueType_) {
    memcpy(value_ + rows_[row], values, sizeof(*values) * colNum);
  }

  if (height_ - 1 == row) {
    sMatrix_->format = HL_SPARSE_CSR;
    sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
    sMatrix_->rows = height_;
    sMatrix_->cols = width_;
    sMatrix_->nnz = elementCnt_;
    hl_memcpy_csr_matrix(
        sMatrix_.get(), value_, rows_, cols_, HPPL_STREAM_DEFAULT);
  }
}

SparseValueType GpuSparseMatrix::getValueType() const { return valueType_; }

void GpuSparseMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  CHECK_EQ(format_, SPARSE_CSC);
  int nnz = sMatrix_->nnz;
  if (memAlloc) {
    matTrans = std::make_shared<GpuSparseMatrix>(
        width_, height_, nnz, valueType_, format_, false);
  } else {
    CHECK(matTrans != nullptr);
  }

  CpuIVector rows(nnz);
  CpuIVector cols(width_ + 1);
  CpuIVector cols_full(nnz);
  CpuVector value(nnz);
  hl_stream_t stream = HPPL_STREAM_1;
  hl_memcpy_from_csc_matrix(value.getData(),
                            nnz,
                            rows.getData(),
                            nnz,
                            cols.getData(),
                            width_ + 1,
                            sMatrix_.get(),
                            stream);

  hl_stream_synchronize(stream);

  /*for every non zero number, get its column index*/
  std::vector<Element> dataVec;
  for (size_t i = 0; i < width_; i++) {
    for (int j = cols.getData()[i]; j < cols.getData()[i + 1]; j++) {
      cols_full.getData()[j] = i;
    }
  }

  /*sort row index and column index by the ascending order*/
  for (int i = 0; i < nnz; i++) {
    dataVec.emplace_back(
        rows.getData()[i], cols_full.getData()[i], value.getData()[i]);
  }
  std::sort(dataVec.begin(), dataVec.end(), [](Element a, Element b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  /*get sorted data, row index, and col index, put them in the right place*/
  cols.resize(height_ + 1);
  rows.resize(nnz);
  value.resize(nnz);

  cols.getData()[0] = 0;
  rows.getData()[0] = dataVec[0].col;
  value.getData()[0] = dataVec[0].val;
  for (int i = 1; i < nnz; i++) {
    if (dataVec[i].row != dataVec[i - 1].row) {
      for (int j = dataVec[i - 1].row + 1; j <= dataVec[i].row; j++) {
        cols.getData()[j] = i;
      }
    }
    rows.getData()[i] = dataVec[i].col;
    value.getData()[i] = dataVec[i].val;
  }
  cols.getData()[height_] = nnz;

  /*copy back from cpu*/
  GpuSparseMatrixPtr dest =
      std::dynamic_pointer_cast<GpuSparseMatrix>(matTrans);
  hl_memcpy_csc_matrix((dest->sMatrix_).get(),
                       value.getData(),
                       rows.getData(),
                       cols.getData(),
                       stream);
  hl_stream_synchronize(stream);
}

void GpuSparseMatrix::mul(const GpuMatrix& a,
                          const GpuMatrix& b,
                          real scaleAB,
                          real scaleT) {
  CHECK(a.useGpu_ && b.useGpu_) << "type not match";
  CHECK(!trans_) << "trans not supported";
  real* A_d = (real*)a.getData();
  real* B_d = (real*)b.getData();
  hl_sparse_matrix_s C_d = sMatrix_.get();
  hl_trans_op_t a_trans = a.trans_ ? HPPL_OP_T : HPPL_OP_N;
  hl_trans_op_t b_trans = b.trans_ ? HPPL_OP_T : HPPL_OP_N;

  if (!a.trans_ && !b.trans_) {
    CHECK(height_ == a.getHeight());
    CHECK(width_ == b.getWidth());
    CHECK(a.getWidth() == b.getHeight());
  } else if (a.trans_ && !b.trans_) {
    CHECK(height_ == a.getWidth());
    CHECK(width_ == b.getWidth());
    CHECK(a.getHeight() == b.getHeight());
  } else if (!a.trans_ && b.trans_) {
    CHECK(height_ == a.getHeight());
    CHECK(width_ == b.getHeight());
    CHECK(a.getWidth() == b.getWidth());
  } else {
    LOG(INFO) << "Not support";
  }
  int dimM = height_;
  int dimN = width_;
  int dimK = !b.trans_ ? b.getHeight() : b.getWidth();
  hl_sparse_matrix_mul(
      A_d, a_trans, B_d, b_trans, C_d, dimM, dimN, dimK, scaleAB, scaleT);
}

void GpuSparseMatrix::mul(const Matrix& a,
                          const Matrix& b,
                          real scaleAB,
                          real scaleT) {
  const auto a_ptr = dynamic_cast<const GpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const GpuMatrix*>(&b);
  if (a_ptr && b_ptr) {
    mul(*a_ptr, *b_ptr, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "not supported";
  }
}

template <class T>
void printBuf(std::ostream& os, T* a, size_t len, const char* name) {
  os << "\n: " << name << " [";
  for (size_t i = 0; i < len; i++) {
    os << a[i] << " ";
  }
  os << "]\n";
}

void GpuSparseMatrix::print(std::ostream& os) const {
  if (format_ == SPARSE_CSC) {
    int nnz = sMatrix_->nnz;
    IVectorPtr rows = IVector::create(nnz, false);
    IVectorPtr cols = IVector::create(width_ + 1, false);
    VectorPtr value = Vector::create(nnz, false);
    hl_stream_t stream = HPPL_STREAM_DEFAULT;
    hl_memcpy_from_csc_matrix(value->getData(),
                              value->getSize(),
                              rows->getData(),
                              rows->getSize(),
                              cols->getData(),
                              cols->getSize(),
                              sMatrix_.get(),
                              stream);
    hl_stream_synchronize(stream);

    printBuf(os, cols->getData(), width_ + 1, "col idx");
    printBuf(os, rows->getData(), elementCnt_, "row idx");
    printBuf(os, value->getData(), elementCnt_, "value");
  }
}

void GpuSparseMatrix::copyFromCSR(CpuSparseMatrix& src, hl_stream_t stream) {
  trans_ = src.trans_;
  size_t nnz = src.getElementCnt();

  resize(src.getHeight(), src.getWidth(), nnz, valueType_, src.getFormat());
  // if have different value type, only copy rows and cols
  SparseValueType vType =
      valueType_ != src.getValueType() ? NO_VALUE : valueType_;

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = vType == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csr_matrix(sMatrix_.get(),
                       vType == NO_VALUE ? NULL : src.getValue(),
                       src.getRows(),
                       src.getCols(),
                       stream);

  // restore type of sMatrix_
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
}

void GpuSparseMatrix::copyFromCSC(CpuSparseMatrix& src, hl_stream_t stream) {
  trans_ = src.trans_;
  size_t nnz = src.getElementCnt();

  resize(src.getHeight(), src.getWidth(), nnz, valueType_, src.getFormat());

  // if have different value type, only copy rows and cols
  SparseValueType vType =
      valueType_ != src.getValueType() ? NO_VALUE : valueType_;

  sMatrix_->format = HL_SPARSE_CSC;
  sMatrix_->type = vType == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csc_matrix(sMatrix_.get(),
                       vType == NO_VALUE ? NULL : src.getValue(),
                       src.getRows(),
                       src.getCols(),
                       stream);

  // restore type of sMatrix_
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
}

void GpuSparseMatrix::copyFrom(GpuSparseMatrix& src, hl_stream_t stream) {
  CHECK(trans_ == src.trans_);
  CHECK(format_ == src.getFormat());
  resize(src.getHeight(),
         src.getWidth(),
         elementCnt_,
         valueType_,
         src.getFormat());

  size_t rowSize = format_ == SPARSE_CSC ? elementCnt_ : height_ + 1;
  size_t colSize = format_ == SPARSE_CSC ? width_ + 1 : elementCnt_;

  if (valueType_ == FLOAT_VALUE && src.getValueType() == FLOAT_VALUE) {
    hl_memcpy_async(
        getValue(), src.getValue(), sizeof(real) * elementCnt_, stream);
  }
  CHECK(getRows());
  CHECK(src.getRows());

  hl_memcpy_async(getRows(), src.getRows(), sizeof(int) * rowSize, stream);
  hl_memcpy_async(getCols(), src.getCols(), sizeof(int) * colSize, stream);
}

void GpuSparseMatrix::copyFrom(CpuSparseMatrix& src, hl_stream_t stream) {
  if (format_ == SPARSE_CSR) {
    copyFromCSR(src, stream);
  } else {
    copyFromCSC(src, stream);
  }
}

void GpuSparseMatrix::trimFromCSR(const CpuSparseMatrix& src) {
  trans_ = src.trans_;
  int* srcCols = src.getCols();
  size_t nnz = std::count_if(srcCols,
                             srcCols + src.getElementCnt(),
                             [this](size_t n) { return n < this->width_; });
  resize(height_, width_, nnz, valueType_, format_);

  rows_[0] = 0;
  size_t index = 0;
  for (size_t r = 0; r < height_; ++r) {
    for (int i = src.getRows()[r]; i < src.getRows()[r + 1]; ++i) {
      if (srcCols[i] < (int)width_) {
        cols_[index] = srcCols[i];
        if (valueType_ == FLOAT_VALUE) {
          value_[index] = src.getValue()[i];
        }
        ++index;
      }
    }
    rows_[r + 1] = index;
  }
  CHECK_EQ(index, nnz);

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csr_matrix(sMatrix_.get(),
                       valueType_ == NO_VALUE ? NULL : value_,
                       rows_,
                       cols_,
                       /*default stream = */ HPPL_STREAM_DEFAULT);
}

void GpuSparseMatrix::trimFromCSC(const CpuSparseMatrix& src) {
  trans_ = src.trans_;
  size_t nnz = src.getCols()[width_] - src.getCols()[0];
  resize(height_, width_, nnz, valueType_, format_);

  cols_[0] = 0;
  for (size_t i = 0; i < width_; i++) {
    cols_[i + 1] = cols_[i] + (int)(src.getRowNum(i));
  }
  memcpy(rows_, src.getRows() + src.getCols()[0], sizeof(int) * nnz);
  if (valueType_ == FLOAT_VALUE) {
    memcpy(value_, src.getValue() + src.getCols()[0], sizeof(real) * nnz);
  }

  sMatrix_->format = HL_SPARSE_CSC;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csc_matrix(sMatrix_.get(),
                       valueType_ == NO_VALUE ? NULL : value_,
                       rows_,
                       cols_,
                       /*default stream = */ HPPL_STREAM_DEFAULT);
}

void GpuSparseMatrix::trimFrom(const CpuSparseMatrix& src) {
  if (format_ == SPARSE_CSR) {
    trimFromCSR(src);
  } else {
    trimFromCSC(src);
  }
}

void GpuSparseMatrix::addBias(Matrix& b, real scale) {
  CHECK(b.getHeight() == 1) << "the Bias should be a vector";
  hl_sparse_matrix_s A_d = sMatrix_.get();
  hl_sparse_matrix_add_bias(A_d, b.getData(), scale);
}

void GpuSparseMatrix::add3(GpuMatrix* b) {
  CHECK(getFormat() != SPARSE_CSC) << "Not supported";
  CHECK(height_ == b->getHeight());
  CHECK(width_ == b->getWidth());
  real* B_d = b->getData();
  hl_sparse_matrix_s A_d = sMatrix_.get();
  hl_sparse_matrix_add_dense(A_d, B_d, height_, width_, 1, 0);
}

void GpuSparseMatrix::add3(MatrixPtr b) {
  if (dynamic_cast<GpuMatrix*>(b.get())) {
    add3(dynamic_cast<GpuMatrix*>(b.get()));
  } else {
    LOG(FATAL) << "not supported";
  }
}

void GpuSparseMatrix::zeroMem() {
  CHECK(valueType_ == FLOAT_VALUE);
  real* value = getValue();
  if (value == NULL) {
    LOG(FATAL) << "value is nullptr";
  }
  hl_matrix_zero_mem(value, elementCnt_);
}

void GpuSparseMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
#ifdef PADDLE_WITH_CUDA
  CHECK(maxIds.useGpu() && maxVal.useGpu()) << "Matrix type are not equal";
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  CHECK_EQ(format_, SPARSE_CSR) << "Only support SPARSE_CSR";

  hl_sparse_matrix_top_k(maxVal.getData(),
                         maxVal.getStride(),
                         maxIds.getData(),
                         sMatrix_.get(),
                         beam,
                         numSamples);
#endif
}

template void GpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_non_value_t* data,
                                        hl_stream_t stream);
template void GpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_float_value_t* data,
                                        hl_stream_t stream);

const unsigned int SparseRowCpuMatrix::kUnusedId_ = -1U;

void SparseRowCpuMatrix::init(size_t height, size_t width) {
  height_ = height;
  if (!indexDictHandle_) {
    indexDictHandle_.reset(new IndexDict);
    indexDictHandle_->globalIndices.assign(height, kUnusedId_);
  }
  localIndices_ = &indexDictHandle_->localIndices;
  globalIndices_ = indexDictHandle_->globalIndices.data();
}

void SparseRowCpuMatrix::mul(CpuSparseMatrix* a,
                             CpuMatrix* b,
                             real scaleAB,
                             real scaleT) {
  CpuMatrix::mul<CpuMatrix, SparseRowCpuMatrix>(a, b, this, scaleAB, scaleT);
}

void SparseRowCpuMatrix::copyFrom(const real* src, size_t size) {
  LOG(FATAL) << "This should not be called";
}

void SparseRowCpuMatrix::zeroMem() {
  apply([](real* buf, size_t len) { memset(buf, 0, sizeof(real) * len); });
  clearRows();
}

void SparseRowCpuMatrix::applyL1(real learningRate, real decayRate) {
  apply([=](real* buf, size_t len) {
    CpuVector value(0, nullptr);
    value.subVecFrom(buf, 0, len);
    value.applyL1(learningRate, decayRate);
  });
}

void SparseRowCpuMatrix::sgdUpdate(BaseMatrix& value,
                                   IVector& t0,
                                   real learningRate,
                                   int currentTime,
                                   real decayRate,
                                   bool useL1,
                                   bool fini) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;

  // t0 and value are vectors
  CHECK_EQ(t0.getSize(), this->height_);
  CHECK_EQ(value.width_, this->height_ * this->width_);

  if (decayRate == 0.0f) {
    if (fini) {
      return;
    }

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] -= learningRate * g[j];
      }
    }
    return;
  }  // else

  if (useL1) {  // L1 decay
    if (fini) {
      for (size_t i = 0; i < this->height_; ++i) {
        real* v = value.rowBuf(i);
        int* t = t0.getData() + i;
        if (t[0] < currentTime) {
          // W(t0) -> W(t+1)
          int tDiff = currentTime - t[0];
          real delta = tDiff * learningRate * decayRate;
          simd::decayL1(v, v, delta, this->width_);
        }
      }
      return;
    }  // else

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      int* t = t0.getData() + localIndices[i];
      if (t[0] < currentTime) {
        // W(t0) -> W(t)
        int tDiff = currentTime - t[0];
        real delta = tDiff * learningRate * decayRate;
        simd::decayL1(v, v, delta, this->width_);
      }

      // W(t) -> W(t+1)
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] -= learningRate * g[j];
      }
      simd::decayL1(v, v, learningRate * decayRate, this->width_);

      // state update to t+1
      t[0] = currentTime + 1;
    }

  } else {  // L2 decay
    if (fini) {
      for (size_t i = 0; i < this->height_; ++i) {
        real* v = value.rowBuf(i);
        int* t = t0.getData() + i;
        if (t[0] < currentTime) {
          // W(t0) -> W(t+1)
          int tDiff = currentTime - t[0];
          real recip = 1.0f / (1.0f + tDiff * learningRate * decayRate);
          for (size_t j = 0; j < this->width_; ++j) {
            v[j] *= recip;
          }
        }
      }
      return;
    }  // else

    real recipDecay = 1.0f / (1.0f + learningRate * decayRate);

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      int* t = t0.getData() + localIndices[i];
      if (t[0] < currentTime) {
        // W(t0) -> W(t)
        int tDiff = currentTime - t[0];
        real recip = 1.0f / (1.0f + tDiff * learningRate * decayRate);
        for (size_t j = 0; j < this->width_; ++j) {
          v[j] *= recip;
        }
      }

      // W(t) -> W(t+1)
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] = recipDecay * (v[j] - learningRate * g[j]);
      }

      // state update to t+1
      t[0] = currentTime + 1;
    }
  }
}

void SparseRowCpuMatrix::addTo(BaseMatrix& dest,
                               std::vector<uint32_t>& ids,
                               size_t tid,
                               size_t numThreads) {
  CHECK(!dest.useGpu_);
  CHECK_EQ(dest.height_ * dest.width_, this->height_ * this->width_);

  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      simd::addTo(dest.rowBuf(id), getLocalRow(i), this->width_);
      ids.push_back(id);
    }
  }
}

void SparseRowCpuMatrix::addTo(SparseRowCpuMatrix& dest,
                               size_t tid,
                               size_t numThreads) {
  CHECK(!dest.useGpu_);
  CHECK_EQ(dest.height_ * dest.width_, this->height_ * this->width_);

  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      dest.checkIndex(id);
      simd::addTo(dest.getRow(id), getLocalRow(i), this->width_);
    }
  }
}

void SparseRowCpuMatrix::zeroMemThread(size_t tid, size_t numThreads) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      memset(this->getLocalRow(i), 0, this->width_ * sizeof(real));
    }
  }
}

void SparseAutoGrowRowCpuMatrix::mul(CpuSparseMatrix* a,
                                     CpuMatrix* b,
                                     real scaleAB,
                                     real scaleT) {
  CpuMatrix::mul<CpuMatrix, SparseAutoGrowRowCpuMatrix>(
      a, b, this, scaleAB, scaleT);
}

void CacheRowCpuMatrix::mul(CpuSparseMatrix* a,
                            CpuMatrix* b,
                            real scaleAB,
                            real scaleT) {
  CpuMatrix::mul<CpuMatrix, CacheRowCpuMatrix>(a, b, this, scaleAB, scaleT);
}

void SparsePrefetchRowCpuMatrix::addRows(const unsigned int* ids, size_t len) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < len; i++) {
    CHECK_LT(*(ids + i), this->getHeight())
        << "id:" << *(ids + i) << "Height:" << this->getHeight()
        << "sparse id value exceeds the max input dimension, "
        << "it could be caused invalid input data samples";
  }
  localIndices.insert(localIndices.end(), ids, ids + len);
}

void SparsePrefetchRowCpuMatrix::addRows(MatrixPtr input) {
  CpuSparseMatrix* mat = dynamic_cast<CpuSparseMatrix*>(input.get());
  CHECK(mat) << "only support sparse matrix";
  addRows(reinterpret_cast<const unsigned int*>(mat->getCols()),
          mat->getElementCnt());
}

void SparsePrefetchRowCpuMatrix::addRows(IVectorPtr ids) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  size_t numSamples = ids->getSize();
  int* index = ids->getData();
  for (size_t i = 0; i < numSamples; ++i) {
    if (index[i] == -1) continue;

    unsigned int id = (unsigned int)index[i];
    CHECK_LT(id, this->getHeight())
        << "id:" << id << "Height:" << this->getHeight()
        << "sparse id value exceeds the max input dimension, "
        << "it could be caused invalid input data samples";
    localIndices.push_back(id);
  }
}

void SparsePrefetchRowCpuMatrix::setupIndices() {
  auto& localIndices = indexDictHandle_->localIndices;
  uniqueIds(localIndices);
  // for each sparse row
  for (size_t id = 0; id < localIndices.size(); ++id) {
    globalIndices_[localIndices[id]] = id;  // sparse row -> local id
  }
  checkStoreSize();
}

void SparseRowCpuMatrix::checkIndices() {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    CHECK_EQ(globalIndices_[localIndices[i]], i);
  }
  checkStoreSize();
}

}  // namespace mypaddle
}  // namespace bubblefs