// Copyright (c) 2011, The Toft Authors
// All rights reserved.
/// @file cgi_params.h
/// @brief CGI paramters handling
/// @date  03/30/2011 05:23:27 PM
/// @author CHEN Feng <chen3feng@gmail.com>

// toft/net/uri/query_params.h

#ifndef BUBBLESF_UTILS_TOFT_NET_URI_QUERY_PARAMS_H_
#define BUBBLESF_UTILS_TOFT_NET_URI_QUERY_PARAMS_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "utils/toft_base_string_string_piece.h"

namespace bubblefs {
namespace mytoft {

struct QueryParam
{
public:
    QueryParam() {}
    QueryParam(const std::string& aname, const std::string& avalue)
        : name(aname), value(avalue)
    {
    }
public:
    std::string name;
    std::string value;
};

class QueryParams
{
public:
    bool Parse(const std::string& params);
    bool ParseFromUrl(const std::string& url);
    void AppendToString(std::string* target) const;
    void WriteToString(std::string* target) const;
    std::string ToString() const;

    const QueryParam* Find(const std::string& name) const;
    QueryParam* Find(const std::string& name);

    QueryParam& Get(size_t index);
    const QueryParam& Get(size_t index) const;

    bool GetValue(const std::string& name, std::string* value) const;
    bool GetValue(const std::string& name, int32_t* value) const;

    const std::string& GetOrDefaultValue(
        const std::string& name,
        const std::string& default_value) const;

    size_t Count() const;
    void Clear();
    bool IsEmpty() const;
    void Add(const QueryParam& param);
    void Add(const std::string& name, const std::string& value);
    void Set(const std::string& name, const std::string& value);
    bool Remove(const std::string& name);
private:
    std::vector<QueryParam> m_params;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLESF_UTILS_TOFT_NET_URI_QUERY_PARAMS_H_