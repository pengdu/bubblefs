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

// cocos2d-x/cocos/base/ObjectFactory.h

#ifndef BUBBLEFS_UTILS_COCOS2D_OBJECT_FACTORY_H_
#define BUBBLEFS_UTILS_COCOS2D_OBJECT_FACTORY_H_

#include <functional>
#include <string>
#include <unordered_map>
#include "platform/cocos2d_macros.h"
#include "utils/cocos2d_ref.h"

namespace bubblefs {
namespace mycocos2d {

class ObjectFactory
{
public:
    typedef Ref* (*Instance)(void);
    typedef std::function<Ref* (void)> InstanceFunc;
    struct TInfo
    {
        TInfo(void);
        TInfo(const std::string& type, Instance ins = nullptr);
        TInfo(const std::string& type, InstanceFunc ins = nullptr);
        TInfo(const TInfo &t);
        ~TInfo(void);
        TInfo& operator= (const TInfo &t);
        std::string _class;
        Instance _fun;
        InstanceFunc _func;
    };
    typedef std::unordered_map<std::string, TInfo>  FactoryMap;

    static ObjectFactory* getInstance();
    static void destroyInstance();
    Ref* createObject(const std::string &name);

    void registerType(const TInfo &t);
    void removeAll();

protected:
    ObjectFactory(void);
    virtual ~ObjectFactory(void);
private:
    static ObjectFactory *_sharedFactory;
    FactoryMap _typeMap;
};

} // namespace mycocos2d
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_COCOS2D_OBJECT_FACTORY_H_