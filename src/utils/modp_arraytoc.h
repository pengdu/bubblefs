/* -*- mode: c; c-basic-offset: 4; indent-tabs-mode: nil; tab-width: 4 -*- */
/* vi: set expandtab shiftwidth=4 tabstop=4: */

#ifndef BUBBLEFS_UTILS_MODP_ARRAYTOC_H_
#define BUBBLEFS_UTILS_MODP_ARRAYTOC_H_

#include "platform/types.h"

/** \brief output a uint32_t array into source code
 *
 *
 * \param[in] ary the input array
 * \param[in] size number of elements in array
 * \param[in] name the name of the struct for the source code
 *
 */
void uint32_array_to_c(const uint32_t* ary, size_t size, const char* name);

/** \brief output an uint32_t array into source code as hex values
 *
 * \param[in] ary the input array
 * \param[in] size number of elements in array
 * \param[in] name the name of the struct for source code
 *
 */
void uint32_array_to_c_hex(const uint32_t* ary, size_t size, const char* name);

/** \brief output a char array into source code
 *
 * \param[in] ary the input array
 * \param[in] size number of elements in array
 * \param[in] name the name of the struct for source code
 */
void char_array_to_c(const char* ary, size_t size, const char* name);

#endif // BUBBLEFS_UTILS_MODP_ARRAYTOC_H_