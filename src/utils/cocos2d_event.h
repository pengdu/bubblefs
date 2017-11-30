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

// cocos2d-x/cocos/base/CCEvent.h
// cocos2d-x/cocos/base/CCEvent.cpp

#ifndef BUBBLEFS_UTILS_COCOS2D_EVENT_H_
#define BUBBLEFS_UTILS_COCOS2D_EVENT_H_

#include "platform/cocos2d_macros.h"
#include "utils/cocos2d_ref.h"

/**
 * @addtogroup base
 * @{
 */

namespace bubblefs {
namespace mycocos2d {

class Node;

/** @class Event
 * @brief Base class of all kinds of events.
 */
class Event : public Ref
{
public:
    /** Type Event type.*/
    enum class Type
    {
        TOUCH,
        KEYBOARD,
        ACCELERATION,
        MOUSE,
        FOCUS,
        GAME_CONTROLLER,
        CUSTOM
    };
    
protected:
    /** Constructor */
    Event::Event(Type type)
    : _type(type)
    , _isStopped(false)
    , _currentTarget(nullptr)
    { }
    
public:
    /** Destructor.
     */
    virtual ~Event() { };

    /** Gets the event type.
     *
     * @return The event type.
     */
    Type getType() const { return _type; }
    
    /** Stops propagation for current event.
     */
    void stopPropagation() { _isStopped = true; }
    
    /** Checks whether the event has been stopped.
     *
     * @return True if the event has been stopped.
     */
    bool isStopped() const { return _isStopped; }
    
    /** Gets current target of the event.
     * @return The target with which the event associates.
     * @note It's only available when the event listener is associated with node.
     *        It returns 0 when the listener is associated with fixed priority.
     */
    Node* getCurrentTarget() { return _currentTarget; }
    
protected:
    /** Sets current target */
    void setCurrentTarget(Node* target) { _currentTarget = target; }
    
    Type _type;     ///< Event type
    
    bool _isStopped;       ///< whether the event has been stopped.
    Node* _currentTarget;  ///< Current target
    
    friend class EventDispatcher;
};

} // namespace mycocos2d
} // namespace bubblefs

// end of base group
/// @}

#endif // BUBBLEFS_UTILS_COCOS2D_EVENT_H_