/**
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

// KlayGE/KFL/include/KFL/CXX17/any.hpp
// KlayGE/KFL/include/KFL/CXX17/iterator.hpp
// KlayGE/KFL/include/KFL/CXX17/optional.hpp
// KlayGE/KFL/include/KFL/CXX17/string_view.hpp

#ifndef BUBBLEFS_PLATFORM_KLAYGE_KFL_CXX17_H_
#define BUBBLEFS_PLATFORM_KLAYGE_KFL_CXX17_H_

#if defined(KLAYGE_CXX17_LIBRARY_ANY_SUPPORT)
        #include <any>
#elif defined(KLAYGE_TS_LIBRARY_ANY_SUPPORT)
        #include <experimental/any>
        namespace std
        {
                using experimental::any;
                using experimental::any_cast;
                using experimental::bad_any_cast;
        }
#else
        #include <boost/any.hpp>
        namespace std
        {
                using boost::any;
                using boost::any_cast;
                using boost::bad_any_cast;
        }
#endif

#if defined(KLAYGE_CXX17_LIBRARY_SIZE_AND_MORE_SUPPORT)
        #include <iterator>
#else
        namespace std
        {
                template <typename T>
                inline constexpr size_t size(T const & t)
                {
                        return t.size();
                }

                template <typename T, size_t N>
                inline constexpr size_t size(T const (&)[N]) noexcept
                {
                        return N;
                }
        }
#endif

#if defined(KLAYGE_CXX17_LIBRARY_OPTIONAL_SUPPORT)
        #include <optional>
#elif defined(KLAYGE_TS_LIBRARY_OPTIONAL_SUPPORT)
        #include <experimental/optional>
        namespace std
        {
                using experimental::optional;
        }
#else
#ifdef KLAYGE_COMPILER_CLANG
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunused-parameter" // Ignore unused parameter 'out', 'v'
#endif
#include <boost/optional.hpp>
#ifdef KLAYGE_COMPILER_CLANG
        #pragma clang diagnostic pop
#endif
        namespace std
        {
                using boost::optional;
        }
#endif

#if defined(KLAYGE_CXX17_LIBRARY_STRING_VIEW_SUPPORT)
        #include <string_view>
#else
        #include <boost/utility/string_view.hpp>

        namespace std
        {
                using boost::basic_string_view;
                using boost::string_view;
                using boost::wstring_view;
        }
#endif

#endif // BUBBLEFS_PLATFORM_KLAYGE_KFL_CXX17_H_