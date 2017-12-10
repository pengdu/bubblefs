/**
 *  Copyright (c) 2015 by Contributors
 */

// ps-lite/include/ps/base.h

#ifndef BUBBLEFS_UTILS_PSLITE_BASE_H_
#define BUBBLEFS_UTILS_PSLITE_BASE_H_

#include <limits>
#include "utils/pslite_env.h"

namespace bubblefs {
namespace mypslite {

#if USE_KEY32
/*! \brief Use unsigned 32-bit int as the key type */
using Key = uint32_t;
#else
/*! \brief Use unsigned 64-bit int as the key type */
using Key = uint64_t;
#endif
/*! \brief The maximal allowed key value */
static const Key kMaxKey = std::numeric_limits<Key>::max();
/** \brief node ID for the scheduler */
static const int kScheduler = 1;
/**
 * \brief the server node group ID
 *
 * group id can be combined:
 * - kServerGroup + kScheduler means all server nodes and the scheuduler
 * - kServerGroup + kWorkerGroup means all server and worker nodes
 */
static const int kServerGroup = 2;
/** \brief the worker node group ID */
static const int kWorkerGroup = 4;

}  // namespace mypslite
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PSLITE_BASE_H_