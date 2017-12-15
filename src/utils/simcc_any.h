
// simcc/simcc/any.h

#ifndef BUBBLEFS_UTILS_SIMCC_ANY_H_
#define BUBBLEFS_UTILS_SIMCC_ANY_H_

#include <algorithm>
#include <typeinfo>
#include "platform/base_error.h"

namespace bubblefs {
namespace mysimcc {
  
// Variant type that can hold Any other type
class Any {
public:
    Any() : content_(NULL) {}
    ~Any() {
        delete content_;
    }

    template<typename ValueType>
    explicit Any(const ValueType& value)
        : content_(new Holder<ValueType>(value)) {}

    Any(const Any& rhs)
        : content_(rhs.content_ ? rhs.content_->Clone() : NULL) {}

public:
    Any& swap(Any& rhs) {
        std::swap(content_, rhs.content_);
        return *this;
    }

    template<typename ValueType>
    Any& operator=(const ValueType& rhs) {
        Any(rhs).swap(*this);
        return*this;
    }

    Any& operator=(const Any& rhs) {
        Any(rhs).swap(*this);
        return*this;
    }

    bool IsEmpty() const {
        return !content_;
    }

    const std::type_info& GetType() const {
        return content_ ? content_->GetType() : typeid(void);
    }

    template<typename ValueType>
    ValueType operator()() const {
        if (GetType() == typeid(ValueType)) {
            return static_cast<Any::Holder<ValueType>*>(content_)->held_;
        } else {
            return ValueType();
        }
    }
protected:
    class PlaceHolder {
    public:
        virtual ~PlaceHolder() {}
    public:
        virtual const std::type_info& GetType() const = 0;
        virtual PlaceHolder* Clone() const = 0;
    };

    template<typename ValueType>
    class Holder : public PlaceHolder {
    public:
        Holder(const ValueType& value)
            : held_(value) {}

        virtual const std::type_info& GetType() const {
            return typeid(ValueType);
        }

        virtual PlaceHolder* Clone() const {
            return new Holder(held_);
        }

        ValueType held_;
    };

protected:
    PlaceHolder* content_;

    template<typename ValueType>
    friend ValueType* any_cast(Any*);
};

template<typename ValueType>
ValueType* any_cast(Any* any) {
    if (any && any->GetType() == typeid(ValueType)) {
        return &(static_cast<Any::Holder<ValueType>*>(any->content_)->held_);
    }

    return NULL;
}

template<typename ValueType>
const ValueType* any_cast(const Any* any) {
    return any_cast<ValueType>(const_cast<Any*>(any));
}

template<typename ValueType>
ValueType any_cast(const Any& any) {
    const ValueType* result = any_cast<ValueType>(&any);
    assert(result);

    if (!result) {
        return ValueType();
    }

    return *result;
}

} // namespace mysimcc
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SIMCC_ANY_H_