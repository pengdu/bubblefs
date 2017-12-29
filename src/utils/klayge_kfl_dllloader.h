/**
 * @file DllLoader.hpp
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

// KlayGE/KFL/include/KFL/DllLoader.hpp

#ifndef BUBBLEFS_UTILS_KLAYGE_KFL_DLLLOADER_H_
#define BUBBLEFS_UTILS_KLAYGE_KFL_DLLLOADER_H_

#include <string>

#if defined(COMPILER_MSVC) || defined(COMPILER_CLANGC2)
        #define DLL_PREFIX ""
#else
        #define DLL_PREFIX "lib"
#endif
#if defined(PLATFORM_WINDOWS)
        #define DLL_EXT_NAME "dll"
#elif defined(PLATFORM_DARWIN)
        #define DLL_EXT_NAME "dylib"
#else
        #define DLL_EXT_NAME "so"
#endif

#define DLL_SUFFIX "." DLL_EXT_NAME

namespace bubblefs {
namespace myklayge {
  
        class DllLoader
        {
        public:
                DllLoader();
                ~DllLoader();

                bool Load(std::string const & dll_name);
                void Free();

                void* GetProcAddress(std::string const & proc_name);

        private:
                void* dll_handle_;
        };
        
} // namespace myklayge
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_KLAYGE_KFL_DLLLOADER_H_