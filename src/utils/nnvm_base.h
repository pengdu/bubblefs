/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Configuration of nnvm as well as basic data structure.
 */

// nnvm/include/nnvm/base.h

#ifndef BUBBLEFS_UTILS_NNVM_BASE_H_
#define BUBBLEFS_UTILS_NNVM_BASE_H_

#include "platform/base_error.h"
#include "utils/dmlc_base.h"
#include "utils/dmlc_any.h"
#include "utils/dmlc_json.h"
#include "utils/dmlc_memory.h"
#include "utils/dmlc_registry.h"
#include "utils/dmlc_array_view.h"

namespace bubblefs {
namespace mynnvm {

/*! \brief any type */
using mydmlc::any;

/*! \brief array_veiw type  */
using mydmlc::array_view;

/*!\brief getter function of any type */
using mydmlc::get;

}  // namespace mynnvm
}  // namespace bubblefs

// describe op registration point
#define NNVM_STRINGIZE_DETAIL(x) #x
#define NNVM_STRINGIZE(x) NNVM_STRINGIZE_DETAIL(x)
#define NNVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" NNVM_STRINGIZE(__LINE__))
#define NNVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" NNVM_STRINGIZE(__LINE__)

#endif  // BUBBLEFS_UTILS_NNVM_BASE_H_