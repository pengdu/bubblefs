// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// baidu/common/include/tprinter.h

#ifndef  BUBBLEFS_UTILS_BDCOM_TPRINTER_H_
#define  BUBBLEFS_UTILS_BDCOM_TPRINTER_H_

#include <stdint.h>
#include <string>
#include <vector>

namespace bubblefs {
namespace mybdcom {

using std::string;

class TPrinter {
public:
    typedef std::vector<string> Line;
    typedef std::vector<Line> Table;

    TPrinter();
    TPrinter(int cols);
    ~TPrinter();

    bool AddRow(const std::vector<string>& cols);

    bool AddRow(int argc, ...);

    bool AddRow(const std::vector<int64_t>& cols);

    void Print(bool has_head = true);

    string ToString(bool has_head = true);

    void Reset();

    void Reset(int cols);

    size_t Rows() { return _table.size(); }

    static string RemoveSubString(const string& input, const string& substr);

private:
    static const uint32_t kMaxColWidth = 50;
    size_t _cols;
    std::vector<int> _col_width;
    Table _table;
};

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_TPRINTER_H_