/**
 * @file PreDeclare.hpp
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

// KlayGE/KFL/include/KFL/CXX17/any.hpp
// KlayGE/KFL/include/KFL/CXX17/iterator.hpp
// KlayGE/KFL/include/KFL/CXX17/optional.hpp
// KlayGE/KFL/include/KFL/CXX17/string_view.hpp
// KlayGE/KFL/include/KFL/PreDeclare.hpp
// KlayGE/KFL/include/KFL/Types.hpp
// KlayGE/KFL/include/KFL/Trace.hpp

#ifndef BUBBLEFS_UTILS_KLAYGE_KFL_PREDECLARE_H_
#define BUBBLEFS_UTILS_KLAYGE_KFL_PREDECLARE_H_

#include <memory>
#include <type_traits>
#include "platform/base_error.h"
#include "platform/macros.h"

#include "boost/operators.hpp"

/*
namespace boost {  

//  <boost::noncopyable>
//  Private copy constructor and copy assignment ensure classes derived from  
//  class noncopyable cannot be copied.  
  
//  Contributed by Dave Abrahams  
  
namespace noncopyable_  // protection from unintended ADL  
{  
  class noncopyable  
  {  
   protected:  
      noncopyable() {}  
      ~noncopyable() {}  
   private:  // emphasize the following members are private  
      noncopyable( const noncopyable& );  
      const noncopyable& operator=( const noncopyable& );  
  };  
}  
  
typedef noncopyable_::noncopyable noncopyable;  
  
} // namespace boost  
*/

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
        #include "boost/any.hpp"
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
#include "boost/optional.hpp"
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
        #include "boost/utility/string_view.hpp"

        namespace std
        {
                using boost::basic_string_view;
                using boost::string_view;
                using boost::wstring_view;
        }
#endif

namespace bubblefs {
namespace myklayge {
  
        class ResIdentifier;
        typedef std::shared_ptr<ResIdentifier> ResIdentifierPtr;
        class DllLoader;

        class XMLDocument;
        typedef std::shared_ptr<XMLDocument> XMLDocumentPtr;
        class XMLNode;
        typedef std::shared_ptr<XMLNode> XMLNodePtr;
        class XMLAttribute;
        typedef std::shared_ptr<XMLAttribute> XMLAttributePtr;

        class bad_join;
        template <typename ResultType>
        class joiner;
        class threader;
        class thread_pool;

        class half;
        template <typename T, int N>
        class Vector_T;
        typedef Vector_T<int32_t, 1> Veci1;
        typedef Vector_T<int32_t, 2> Veci2;
        typedef Vector_T<int32_t, 3> Veci3;
        typedef Vector_T<int32_t, 4> Veci4;
        typedef Vector_T<uint32_t, 1> Vecui1;
        typedef Vector_T<uint32_t, 2> Vecui2;
        typedef Vector_T<uint32_t, 3> Vecui3;
        typedef Vector_T<uint32_t, 4> Vecui4;
        typedef Vector_T<float, 1> Vecf1;
        typedef Vector_T<float, 2> Vecf2;
        typedef Vector_T<float, 3> Vecf3;
        typedef Vector_T<float, 4> Vecf4;
        template <typename T>
        class Matrix4_T;
        typedef Matrix4_T<float> float4x4;
        typedef Matrix4_T<float> Matf4;
        template <typename T>
        class Quaternion_T;
        typedef Quaternion_T<float> Quaternion;
        template <typename T>
        class Plane_T;
        typedef Plane_T<float> Plane;
        template <typename T>
        class Color_T;
        typedef Color_T<float> Color;
        template <typename T>
        class Size_T;
        typedef Size_T<float> Size;
        typedef Size_T<int32_t> ISize;
        typedef Size_T<uint32_t> UISize;
        typedef std::shared_ptr<Size> SizePtr;
        typedef std::shared_ptr<ISize> ISizePtr;
        typedef std::shared_ptr<UISize> UISizePtr;
        template <typename T>
        class Rect_T;
        typedef Rect_T<float> Rect;
        typedef Rect_T<int32_t> IRect;
        typedef Rect_T<uint32_t> UIRect;
        typedef std::shared_ptr<Rect> RectPtr;
        typedef std::shared_ptr<IRect> IRectPtr;
        typedef std::shared_ptr<UIRect> UIRectPtr;
        template <typename T>
        class Bound_T;
        typedef Bound_T<float> Bound;
        typedef std::shared_ptr<Bound> BoundPtr;
        template <typename T>
        class Sphere_T;
        typedef Sphere_T<float> Sphere;
        typedef std::shared_ptr<Sphere> SpherePtr;
        template <typename T>
        class AABBox_T;
        typedef AABBox_T<float> AABBox;
        typedef std::shared_ptr<AABBox> AABBoxPtr;
        template <typename T>
        class Frustum_T;
        typedef Frustum_T<float> Frustum;
        typedef std::shared_ptr<Frustum> FrustumPtr;
        template <typename T>
        class OBBox_T;
        typedef OBBox_T<float> OBBox;
        typedef std::shared_ptr<OBBox> OBBoxPtr;
        
        typedef uint32_t FourCC;
        
        class Trace
        {
        public:
                Trace(char const * func, int line = 0, char const * file = nullptr)
                        : func_(func), line_(line), file_(file)
                {
                        PRINTF_TRACE("Enter %s in file %s (line %d)", func_, file_ != nullptr ? file_ : "", line_);
                }

                ~Trace()
                {
                        PRINTF_TRACE("Leave %s in file %s (line %d)", func_, file_ != nullptr ? file_ : "", line_);
                }

        private:
                char const * func_;
                int line_;
                char const * file_;
        };

} // namespace myklayge
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_KLAYGE_KFL_PREDECLARE_H_