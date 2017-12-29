/**
 * @file CustomizedStreamBuf.hpp
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

// KlayGE/KFL/include/KFL/CustomizedStreamBuf.hpp

#ifndef BUBBLEFS_UTILS_KLAYGE_KFL_CUSTOMIZED_STREAMBUF_H_
#define BUBBLEFS_UTILS_KLAYGE_KFL_CUSTOMIZED_STREAMBUF_H_

#include <streambuf>
#include "platform/macros.h"

namespace bubblefs {
namespace myklayge {
  
        class MemStreamBuf : public std::streambuf
        {
        public:
                MemStreamBuf(void const * begin, void const * end);

        protected:
                virtual int_type uflow() override;
                virtual int_type underflow() override;

                virtual std::streamsize xsgetn(char_type* s, std::streamsize count) override;

                virtual int_type pbackfail(int_type ch) override;
                virtual std::streamsize showmanyc() override;

                virtual pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
                virtual pos_type seekpos(pos_type sp, std::ios_base::openmode which) override;
                
        private: 
                char_type const * const begin_;
                char_type const * const end_;
                char_type const * current_;
                
                DISALLOW_COPY_AND_ASSIGN(MemStreamBuf);
        };
        
} // namespace myklayge
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_KLAYGE_KFL_CUSTOMIZED_STREAMBUF_H_