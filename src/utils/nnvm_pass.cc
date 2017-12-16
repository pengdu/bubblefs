/*!
 *  Copyright (c) 2016 by Contributors
 * \file pass.cc
 * \brief Support for pass registry.
 */

// nnvm/src/core/pass.cc

#include "utils/nnvm_pass.h"
#include <algorithm>

namespace bubblefs {
namespace mydmlc {
// enable registry
DMLC_REGISTRY_ENABLE(mynnvm::PassFunctionReg);
}  // namespace dmlc
}  // namespace bubblefs

namespace bubblefs {
namespace mynnvm {

const PassFunctionReg* FindPassDep(const std::string&attr_name) {
  for (auto* r : mydmlc::Registry<PassFunctionReg>::List()) {
    for (auto& s : r->graph_attr_targets) {
      if (s == attr_name) return r;
    }
  }
  return nullptr;
}

Graph ApplyPasses(Graph g,
                  const std::vector<std::string>& pass) {
  std::vector<const PassFunctionReg*> fpass;
  for (auto& name : pass) {
    auto* reg = mydmlc::Registry<PassFunctionReg>::Find(name);
    PANIC_ENFORCE(reg != nullptr, "Cannot find pass in the registry");
    fpass.push_back(reg);
  }

  for (auto r : fpass) {
    for (auto& dep : r->graph_attr_dependency) {
      if (g.attrs.count(dep) == 0) {
        auto* pass_dep = FindPassDep(dep);
        std::string msg;
        if (pass_dep != nullptr) {
          msg = " The attribute is provided by pass " + pass_dep->name;
        }
        PANIC("Graph attr dependency is required by pass but is not available ");
      }
    }
    g = r->body(std::move(g));
  }

  return g;
}

}  // namespace nnvm
}  // namespace bubblefs