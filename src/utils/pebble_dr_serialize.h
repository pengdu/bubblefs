/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

// Pebble/src/framework/dr/serialize.h

#ifndef BUBBLEFS_UTILS_PEBBLE_DR_SERIALIZE_H_
#define BUBBLEFS_UTILS_PEBBLE_DR_SERIALIZE_H_

#include <string>
#include <stdlib.h>
#include "platform/types.h"
#include "utils/pebble_dr_virtual_transport.h"

namespace bubblefs {
namespace mypebble { namespace dr { namespace detail {
class ArrayOutOfBoundsException : public mypebble::TException {
};

class FixBuffer : public mypebble::dr::transport::TVirtualTransport<FixBuffer> {
    public:
        FixBuffer(uint8_t *buf, uint32_t buf_len)
            : m_buf(buf),
              m_buf_bound(buf + buf_len),
              m_buf_pos(buf) {
        }

        uint32_t read(uint8_t* buf, uint32_t len);

        uint32_t readAll(uint8_t* buf, uint32_t len);

        void write(const uint8_t* buf, uint32_t len);

        int32_t used();

    private:
        uint8_t *m_buf;

        uint8_t *m_buf_bound;

        uint8_t *m_buf_pos;
};

class AutoBuffer : public mypebble::dr::transport::TVirtualTransport<AutoBuffer> {
    public:
        explicit AutoBuffer(uint32_t init_size) {
            m_buf = m_buf_pos = reinterpret_cast<uint8_t *>(malloc(init_size));
            m_buf_bound = m_buf + init_size;
        }

        void write(const uint8_t* buf, uint32_t len);

        int32_t used();

        char * str();

        ~AutoBuffer() {
            free(m_buf);
        }

    private:
        uint8_t *m_buf;

        uint8_t *m_buf_bound;

        uint8_t *m_buf_pos;
};
} // namespace detail

/// @brief 字段序列化接口返回码
enum PackError {
    kINSUFFICIENTBUFFER = -1, ///< 给的BUFFER不足打包
    kINVALIDBUFFER = -2, ///< 无法解包
    kINVALIDOBJECT = -3, ///< 对象类型非法
    kINVALIDPARAMETER = -4, ///< 参数非法
    kINVALIDPATH = -5, ///< 找不到更新路径
    kINVALIDFIELD = -6, ///< 所给的字段不匹配
    kUNKNOW = -99 ///< 未知错误
};

/// @brief 内部使用
template<typename TDATA, typename TPROTOCOL>
int Pack(const TDATA *obj, uint8_t *buff, uint32_t buff_len) {
    if (obj == NULL || buff == NULL) return dr::kINVALIDPARAMETER;

    try {
        std::shared_ptr<detail::FixBuffer> f_buff(new detail::FixBuffer(buff, buff_len));
        TPROTOCOL protocol(f_buff);

        obj->write(&protocol);

        return f_buff->used();
    } catch (dr::detail::ArrayOutOfBoundsException) {
        return dr::kINSUFFICIENTBUFFER;
    } catch (...) {
        return dr::kUNKNOW;
    }
}

/// @brief 内部使用
template<typename TDATA, typename TPROTOCOL>
int UnPack(TDATA *obj, const uint8_t *buff, uint32_t buff_len) {
    if (obj == NULL || buff == NULL) return dr::kINVALIDPARAMETER;

    try {
        std::shared_ptr<detail::FixBuffer> f_buff(new detail::FixBuffer(
            const_cast<uint8_t*>(buff), buff_len));
        TPROTOCOL protocol(f_buff);

        obj->read(&protocol);

        return f_buff->used();
    } catch (dr::detail::ArrayOutOfBoundsException) {
        return dr::kINVALIDBUFFER;
    } catch (...) {
        return dr::kUNKNOW;
    }
}

class InvalidParameterException : public mypebble::TException {
};

/// @brief 内部使用
template<typename TDATA, typename TPROTOCOL>
int Pack(const TDATA *obj, std::string *str) { //NOLINT
    if (obj == NULL || str == NULL) return dr::kINVALIDPARAMETER;

    try {
        std::shared_ptr<detail::AutoBuffer> a_buff(new detail::AutoBuffer(256));
        TPROTOCOL protocol(a_buff);

        obj->write(&protocol);

        str->assign(a_buff->str(), a_buff->used());

        return a_buff->used();
    } catch (dr::detail::ArrayOutOfBoundsException) {
        return dr::kINSUFFICIENTBUFFER;
    } catch (...) {
        return dr::kUNKNOW;
    }
}

/// @brief 内部使用
template<typename TDATA, typename TPROTOCOL>
int UnPack(TDATA *obj, const std::string &str) {
    return UnPack<TDATA, TPROTOCOL>(obj,
        reinterpret_cast<const uint8_t*>(str.c_str()), str.size());
}

/// @brief 不依赖生成代码，直接对序列化后(Binary打包)的buffer进行字段更新操作
/// @param path 用id序列表示的要更新字段的路径
/// @param old_data 原始数据
/// @param old_data_len 原始数据的长度
/// @param field (借助反射)打包后的字段
/// @param field_len 打包后的字段长度
/// @param new_data_buf 接收新数据的buff，该buff长度应该足以接收新数据，
///   建议等于(@ref old_data_len + @ref field_len)
/// @param new_data_buf_len 接收新数据的buff长度
/// @return >0表示成功，返回新数据的长度，否则表示失败，具体看@ref PackError
int UpdateField(const std::vector<int16_t> &path, char *old_data, size_t old_data_len,
    char *field, size_t field_len, char *new_data_buf, size_t new_data_buf_len);

} // namespace dr
} // namespace mypebble
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PEBBLE_DR_SERIALIZE_H_