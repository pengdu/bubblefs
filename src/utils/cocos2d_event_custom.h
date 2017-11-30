/****************************************************************************
 Copyright (c) 2013-2017 Chukong Technologies Inc.
 
 http://www.cocos2d-x.org
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ****************************************************************************/

// cocos2d-x/cocos/base/CCEventCustom.h
// cocos2d-x/cocos/base/CCEventCustom.cpp

#ifndef BUBBLEFS_UTILS_COCOS2D_EVENT_CUSTOM_H_
#define BUBBLEFS_UTILS_COCOS2D_EVENT_CUSTOM_H_

#include <string>
#include "utils/cocos2d_event.h"

/**
 * @addtogroup base
 * @{
 */

namespace bubblefs {
namespace mycocos2d {

/** @class EventCustom
 * @brief Custom event.
 */
class EventCustom : public Event
{
public:
    /** Constructor.
     *
     * @param eventName A given name of the custom event.
     * @js ctor
     */
    EventCustom::EventCustom(const std::string& eventName)
    : Event(Type::CUSTOM)
    , _userData(nullptr)
    , _eventName(eventName)
    { }
    
    /** Sets user data.
     *
     * @param data The user data pointer, it's a void*.
     */
    void setUserData(void* data) { _userData = data; }
    
    /** Gets user data.
     *
     * @return The user data pointer, it's a void*.
     */
    void* getUserData() const { return _userData; }
    
    /** Gets event name.
     *
     * @return The name of the event.
     */
    const std::string& getEventName() const { return _eventName; }
protected:
    void* _userData;       ///< User data
    std::string _eventName;
};

} // namespace mycocos2d
} // namespace bubblefs

// end of base group
/// @}

#endif // BUBBLEFS_UTILS_COCOS2D_EVENT_CUSTOM_H_