/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/framework/scope.h
// Paddle/paddle/framework/scope.cc

#pragma once

#include <list>
#include <memory>  // for unique_ptr
#include <mutex>   // for call_once
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/paddle_variable.h"
#include "utils/paddle_printf.h"
#include "utils/paddle_threadpool.h"

namespace bubblefs {
namespace mypaddle {
namespace framework {

class Scope;

/**
 * @brief Scope that manage all variables.
 *
 * Scope is an association of a name to Variable. All variables belong to
 * Scope. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}
  ~Scope();

  /// Create a sub-scope. Returns a reference other than a pointer so
  /// to prevent from manual deletion.
  /// Mark it to const because that new kid scope cannot change parent scope.
  Scope& NewScope() const;

  /// Create a variable with given name if it doesn't exist.
  Variable* Var(const std::string& name);

  /// Create a variable with a scope-unique name.
  Variable* Var(std::string* name = nullptr);

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  Variable* FindVar(const std::string& name) const;

  const Scope& parent() const { return *parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  const Scope* FindScope(const Variable* var) const;

  void DeleteScope(Scope* scope);

  /// Drop all kids scopes belonged to this scope.
  void DropKids();

  // enumerate all the variables current contains.
  std::vector<std::string> LocalVarNames() const;

  // Rename variable to a new name
  void Rename(const std::string& origin_name,
              const std::string& new_name) const;

  // Rename variable to a new name and return the new name
  std::string Rename(const std::string& origin_name) const;

 private:
  Variable* FindVarLocally(const std::string& name) const;

  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const* parent) : parent_(parent) {}

  mutable std::unordered_map<std::string, Variable*> vars_;
  mutable std::list<Scope*> kids_;
  Scope const* parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Scope);
};

Scope::~Scope() {
  DropKids();
  for (auto& kv : vars_) {
    VLOG(3) << "Destroy variable " << kv.first;
    delete kv.second;
  }
}

Scope& Scope::NewScope() const {
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable* Scope::Var(const std::string& name) {
  auto* v = FindVarLocally(name);
  if (v != nullptr) return v;
  v = new Variable();
  vars_[name] = v;
  VLOG(3) << "Create variable " << name;
  v->name_ = &(vars_.find(name)->first);
  return v;
}

Variable* Scope::Var(std::string* name) {
  auto var_name = string::Sprintf("%p.%d", this, vars_.size());
  if (name != nullptr) {
    *name = var_name;
  }
  return Var(var_name);
}

Variable* Scope::FindVar(const std::string& name) const {
  auto var = FindVarLocally(name);
  if (var != nullptr) {
    return var;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  for (auto& kv : vars_) {
    if (kv.second == var) {
      return this;
    }
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
}
void Scope::DropKids() {
  for (Scope* s : kids_) delete s;
  kids_.clear();
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::vector<std::string> known_vars;
  known_vars.reserve(this->vars_.size());
  for (auto& p : vars_) {
    known_vars.emplace_back(p.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) {
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  PADDLE_ENFORCE(it != this->kids_.end(), "Cannot find %p as kid scope", scope);
  this->kids_.erase(it);
  // Make delete async.
  Async([scope] { delete scope; });
}

void Scope::Rename(const std::string& origin_name,
                   const std::string& new_name) const {
  auto origin_it = vars_.find(origin_name);
  PADDLE_ENFORCE(origin_it != vars_.end(),
                 "Cannot find original variable with name %s", origin_name);
  auto new_it = vars_.find(new_name);
  PADDLE_ENFORCE(new_it == vars_.end(),
                 "The variable with name %s is already in the scope", new_name);
  vars_[new_name] = origin_it->second;
  vars_.erase(origin_it);
}

std::string Scope::Rename(const std::string& origin_name) const {
  auto var_name = string::Sprintf("%p.%d", this, vars_.size());
  Rename(origin_name, var_name);
  return var_name;
}

Variable* Scope::FindVarLocally(const std::string& name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) return it->second;
  return nullptr;
}

}  // namespace framework
}  // namespace mypaddle
}  // namespace bubblefs