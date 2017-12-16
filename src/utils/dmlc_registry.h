/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.h
 * \brief Registry utility that helps to build registry singletons.
 */

// dmlc-core/include/dmlc/registry.h

#ifndef BUBBLEFS_UTILS_DMLC_REGISTRY_H_
#define BUBBLEFS_UTILS_DMLC_REGISTRY_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "platform/base_error.h"
#include "platform/macros.h"
#include "utils/dmlc_parameter.h"

/*
/// \brief A registry for file system implementations.
///
/// Filenames are specified as an URI, which is of the form
/// [scheme://]<filename>.
/// File system implementations are registered using the REGISTER_FILE_SYSTEM
/// macro, providing the 'scheme' as the key.

class FileSystemRegistry {
 public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry();
  virtual Status Register(const string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const string& scheme) = 0;
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<string>* schemes) = 0;
};

//std::unique_ptr<FileSystemRegistry> file_system_registry_;
 
class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  Status Register(const string& scheme, Factory factory) override;
  FileSystem* Lookup(const string& scheme) override;
  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, std::unique_ptr<FileSystem>> registry_
      GUARDED_BY(mu_);
};

Status FileSystemRegistryImpl::Register(const string& scheme,
                                        FileSystemRegistry::Factory factory) {
  mutex_lock lock(mu_);
  if (!registry_.emplace(string(scheme), std::unique_ptr<FileSystem>(factory()))
           .second) {
    return errors::AlreadyExists("File factory for ", scheme,
                                 " already registered");
  }
  return Status::OK();
}

FileSystem* FileSystemRegistryImpl::Lookup(const string& scheme) {
  mutex_lock lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<string>* schemes) {
  mutex_lock lock(mu_);
  for (const auto& e : registry_) {
    schemes->push_back(e.first);
  }
  return Status::OK();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {}

Status Env::GetFileSystemForFile(const string& fname, FileSystem** result) {
  StringPiece scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(scheme.ToString());
  if (!file_system) {
    return errors::Unimplemented("File system scheme ", scheme,
                                 " not implemented");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<string>* schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const string& scheme,
                               FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, std::move(factory));
}

//////////////////////////////////////////////////////////////////////////

static mutex* get_device_factory_lock() {
  static mutex device_factory_lock;
  return &device_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<DeviceFactory> factory;
  int priority;
};

std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
}

}  // namespace

// static
int32 DeviceFactory::DevicePriority(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter != factories.end()) {
    return iter->second.priority;
  }

  return -1;
}

// static
void DeviceFactory::Register(const string& device_type, DeviceFactory* factory,
                             int priority) {
  mutex_lock l(*get_device_factory_lock());
  std::unique_ptr<DeviceFactory> factory_ptr(factory);
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter == factories.end()) {
    factories[device_type] = {std::move(factory_ptr), priority};
  } else {
    if (iter->second.priority < priority) {
      iter->second = {std::move(factory_ptr), priority};
    } else if (iter->second.priority == priority) {
      LOG(FATAL) << "Duplicate registration of device factory for type "
                 << device_type << " with the same priority " << priority;
    }
  }
}

DeviceFactory* DeviceFactory::GetFactory(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());  // could use reader lock
  auto it = device_factories().find(device_type);
  if (it == device_factories().end()) {
    return nullptr;
  }
  return it->second.factory.get();
}

Status DeviceFactory::AddDevices(const SessionOptions& options,
                                 const string& name_prefix,
                                 std::vector<Device*>* devices) {
  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return Status::OK();
}

Device* DeviceFactory::NewDevice(const string& type,
                                 const SessionOptions& options,
                                 const string& name_prefix) {
  auto device_factory = GetFactory(type);
  if (!device_factory) {
    return nullptr;
  }
  SessionOptions opt = options;
  (*opt.config.mutable_device_count())[type] = 1;
  std::vector<Device*> devices;
  TF_CHECK_OK(device_factory->CreateDevices(opt, name_prefix, &devices));
  CHECK_EQ(devices.size(), size_t{1});
  return devices[0];
}
*/ 

namespace bubblefs {  
namespace mydmlc {
/*!
 * \brief Registry class.
 *  Registry can be used to register global singletons.
 *  The most commonly use case are factory functions.
 *
 * \tparam EntryType Type of Registry entries,
 *     EntryType need to name a name field.
 */
template<typename EntryType>
class Registry {
 public:
  /*! \return list of entries in the registry(excluding alias) */
  inline static const std::vector<const EntryType*>& List() {
    return Get()->const_list_;
  }
  /*! \return list all names registered in the registry, including alias */
  inline static std::vector<std::string> ListAllNames() {
    const std::map<std::string, EntryType*> &fmap = Get()->fmap_;
    typename std::map<std::string, EntryType*>::const_iterator p;
    std::vector<std::string> names;
    for (p = fmap.begin(); p !=fmap.end(); ++p) {
      names.push_back(p->first);
    }
    return names;
  }
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const EntryType *Find(const std::string &name) {
    const std::map<std::string, EntryType*> &fmap = Get()->fmap_;
    typename std::map<std::string, EntryType*>::const_iterator p = fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return NULL;
    }
  }
  /*!
   * \brief Add alias to the key_name
   * \param key_name The original entry key
   * \param alias The alias key.
   */
  inline void AddAlias(const std::string& key_name,
                       const std::string& alias) {
    EntryType* e = fmap_.at(key_name);
    if (fmap_.count(alias)) {
      PANIC_ENFORCE_EQ(e, fmap_.at(alias)); // "Trying to register alias " << alias << " for key " << key_name << " but " << alias << " is already taken";
    } else {
      fmap_[alias] = e;
    }
  }
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER__(const std::string& name) {
    PANIC_ENFORCE_EQ(fmap_.count(name), 0U); // name << " already registered";
    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    const_list_.push_back(e);
    entry_list_.push_back(e);
    return *e;
  }
  /*!
   * \brief Internal function to either register or get registered entry
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER_OR_GET__(const std::string& name) {
    if (fmap_.count(name) == 0) {
      return __REGISTER__(name);
    } else {
      return *fmap_.at(name);
    }
  }
  /*!
   * \brief get a singleton of the Registry.
   *  This function can be defined by DMLC_ENABLE_REGISTRY.
   * \return get a singleton
   */
  static Registry *Get();

 private:
  /*! \brief list of entry types */
  std::vector<EntryType*> entry_list_;
  /*! \brief list of entry types */
  std::vector<const EntryType*> const_list_;
  /*! \brief map of name->function */
  std::map<std::string, EntryType*> fmap_;
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {
    for (size_t i = 0; i < entry_list_.size(); ++i) {
      delete entry_list_[i];
    }
  }
};

/*!
 * \brief Common base class for function registry.
 *
 * \code
 *  // This example demonstrates how to use Registry to create a factory of trees.
 *  struct TreeFactory :
 *      public FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
 *  };
 *
 *  // in a independent cc file
 *  namespace dmlc {
 *  DMLC_REGISTRY_ENABLE(TreeFactory);
 *  }
 *  // register binary tree constructor into the registry.
 *  DMLC_REGISTRY_REGISTER(TreeFactory, TreeFactory, BinaryTree)
 *      .describe("Constructor of BinaryTree")
 *      .set_body([]() { return new BinaryTree(); });
 * \endcode
 *
 * \tparam EntryType The type of subclass that inheritate the base.
 * \tparam FunctionType The function type this registry is registerd.
 */
template<typename EntryType, typename FunctionType>
class FunctionRegEntryBase {
 public:
  /*! \brief name of the entry */
  std::string name;
  /*! \brief description of the entry */
  std::string description;
  /*! \brief additional arguments to the factory function */
  std::vector<ParamFieldInfo> arguments;
  /*! \brief Function body to create ProductType */
  FunctionType body;
  /*! \brief Return type of the function */
  std::string return_type;

  /*!
   * \brief Set the function body.
   * \param body Function body to set.
   * \return reference to self.
   */
  inline EntryType &set_body(FunctionType body) {
    this->body = body;
    return this->self();
  }
  /*!
   * \brief Describe the function.
   * \param description The description of the factory function.
   * \return reference to self.
   */
  inline EntryType &describe(const std::string &description) {
    this->description = description;
    return this->self();
  }
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline EntryType &add_argument(const std::string &name,
                                 const std::string &type,
                                 const std::string &description) {
    ParamFieldInfo info;
    info.name = name;
    info.type = type;
    info.type_info_str = info.type;
    info.description = description;
    arguments.push_back(info);
    return this->self();
  }
  /*!
   * \brief Append list if arguments to the end.
   * \param args Additional list of arguments.
   * \return reference to self.
   */
  inline EntryType &add_arguments(const std::vector<ParamFieldInfo> &args) {
    arguments.insert(arguments.end(), args.begin(), args.end());
    return this->self();
  }
  /*!
  * \brief Set the return type.
  * \param type Return type of the function, could be Symbol or Symbol[]
  * \return reference to self.
  */
  inline EntryType &set_return_type(const std::string &type) {
    return_type = type;
    return this->self();
  }

 protected:
  /*!
   * \return reference of self as derived type
   */
  inline EntryType &self() {
    return *(static_cast<EntryType*>(this));
  }
};

/*!
 * \def DMLC_REGISTRY_ENABLE
 * \brief Macro to enable the registry of EntryType.
 * This macro must be used under namespace dmlc, and only used once in cc file.
 * \param EntryType Type of registry entry
 */
#define DMLC_REGISTRY_ENABLE(EntryType)                                 \
  template<>                                                            \
  Registry<EntryType > *Registry<EntryType >::Get() {                   \
    static Registry<EntryType > inst;                                   \
    return &inst;                                                       \
  }                                                                     \

/*!
 * \brief Generic macro to register an EntryType
 *  There is a complete example in FactoryRegistryEntryBase.
 *
 * \param EntryType The type of registry entry.
 * \param EntryTypeName The typename of EntryType, must do not contain namespace :: .
 * \param Name The name to be registered.
 * \sa FactoryRegistryEntryBase
 */
#define DMLC_REGISTRY_REGISTER(EntryType, EntryTypeName, Name)          \
  static ATTRIBUTE_UNUSED EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ = \
      ::bubblefs::mydmlc::Registry<EntryType>::Get()->__REGISTER__(#Name)           \

/*!
 * \brief (Optional) Declare a file tag to current file that contains object registrations.
 *
 *  This will declare a dummy function that will be called by register file to
 *  incur a link dependency.
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_LINK_TAG
 */
#define DMLC_REGISTRY_FILE_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __() { return 0; }

/*!
 * \brief (Optional) Force link to all the objects registered in file tag.
 *
 *  This macro must be used in the same file as DMLC_REGISTRY_ENABLE and
 *  in the same namespace as DMLC_REGISTRY_FILE_TAG
 *
 *  DMLC_REGISTRY_FILE_TAG and DMLC_REGISTRY_LINK_TAG are optional macros for registration.
 *  They are used to encforce link of certain file into during static linking.
 *
 *  This is mainly used to solve problem during statically link a library which contains backward registration.
 *  Specifically, this avoids the objects in these file tags to be ignored by compiler.
 *
 *  For dynamic linking, this problem won't occur as everything is loaded by default.
 *
 *  Use of this is optional as it will create an error when a file tag do not exist.
 *  An alternative solution is always ask user to enable --whole-archieve during static link.
 *
 * \begincode
 * // in file objective_registry.cc
 * DMLC_REGISTRY_ENABLE(MyObjective);
 * DMLC_REGISTRY_LINK_TAG(regression_op);
 * DMLC_REGISTRY_LINK_TAG(rank_op);
 *
 * // in file regression_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(regression_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, logistic_reg, logistic_reg);
 * // ...
 *
 * // in file rank_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(rank_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, pairwiserank, pairwiserank);
 *
 * \endcode
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_ENABLE, DMLC_REGISTRY_FILE_TAG
 */
#define DMLC_REGISTRY_LINK_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __();                   \
  static int ATTRIBUTE_UNUSED __reg_file_tag_ ## UniqueTag ## __ = \
      __dmlc_registry_file_tag_ ## UniqueTag ## __();
      
} // namespace mydmlc   
} // namesapce bubblefs  

#endif  // BUBBLEFS_UTILS_DMLC_REGISTRY_H_