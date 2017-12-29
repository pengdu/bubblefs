/**
 * @file ErrorHandling.cpp
 * @author Minmin Gong
 *
 * @section DESCRIPTION
 *
 * This source file is part of KFL, a subproject of KlayGE
 * For the latest info, see http://www.klayge.org
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * You may alternatively use this source under the terms of
 * the KlayGE Proprietary License (KPL). You can obtained such a license
 * from http://www.klayge.org/licensing/.
 */

// KlayGE/KFL/src/Kernel/ErrorHandling.cpp

#include "utils/klayge_kfl_error_handling.h"
#include <system_error>
#include "platform/base_error.h"

#include "boost/lexical_cast.hpp"

namespace bubblefs {
namespace myklayge {
  
        std::string CombineFileLine(std::string_view file, int line)
        {
                return std::string(file) + ": " + boost::lexical_cast<std::string>(line);
        }

        void Verify(bool x)
        {
                if (!x)
                {
                        TERRC(std::errc::function_not_supported);
                }
        }

        void KFLUnreachableInternal(char const * msg, char const * file, uint32_t line)
        {
                if (msg)
                {
                        PRINTF_ERROR("%s\n", msg);
                }
                if (file)
                {
                        PRINTF_ERROR("UNREACHABLE executed at %s: %d.\n", file, line);
                }
                else
                {
                        PRINTF_ERROR("UNREACHABLE executed.\n");
                }

                TMSG("Unreachable.");
        }

} // namespace myklayge
} // namespace bubblefs