/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

// caffe/include/caffe/layer_factory.hpp

#ifndef BUBBLEFS_UTILS_CAFFE_CLASS_FACTORY_H_
#define BUBBLEFS_UTILS_CAFFE_CLASS_FACTORY_H_

#include <map>
#include <string>
#include <vector>

namespace bubblefs {
namespace mycaffe {

#define CaffeTypeRegistry(TypeName) \
class TypeName##Registry { \
 public: \
  typedef std::shared_ptr<TypeName> (*Creator)(); \
  typedef std::map<std::string, Creator> CreatorRegistry; \
  static CreatorRegistry& Registry() { \
    static CreatorRegistry* g_registry_ = new CreatorRegistry(); \
    return *g_registry_; \
  } \
  static void AddCreator(const std::string& type, Creator creator) { \
    CreatorRegistry& registry = Registry(); \
    registry[type] = creator; \
  } \
  static std::shared_ptr<TypeName> Create##TypeName(const std::string& type) { \
    CreatorRegistry& registry = Registry(); \
    return registry[type](); \
  } \
  static vector<std::string> TypeName##TypeList() { \
    CreatorRegistry& registry = Registry(); \
    std::vector<std::string> types; \
    for (typename CreatorRegistry::iterator iter = registry.begin(); \
         iter != registry.end(); ++iter) { \
      types.push_back(iter->first); \
    } \
    return types; \
  } \
 private: \
  TypeName##Registry() {} \
  static std::string LayerTypeListString() { \
    std::vector<std::string> types = TypeName##TypeList(); \
    std::string types_str; \
    for (std::vector<std::string>::iterator iter = types.begin(); \
         iter != types.end(); ++iter) { \
      if (iter != types.begin()) { \
        types_str += ", "; \
      } \
      types_str += *iter; \
    } \
    return types_str; \
  } \
}; \

#define CaffeTypeRegistry(TypeName) \
class TypeName##Registerer { \
 public: \
  TypeName##Registerer(const std::string& type, \
                  std::shared_ptr<TypeName> (*creator)()) { \
    TypeName##Registry::AddCreator(type, creator); \
  } \
}; \

}  // namespace mycaffe
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE_CLASS_FACTORY_H_