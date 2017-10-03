#include <stdio.h>
#include <functional>
#include "utils/registry.h"

namespace tree {
  
struct Tree {
  virtual void Print() = 0;
  virtual ~Tree() {}
};

struct BinaryTree : public Tree {
  virtual void Print() {
    printf("I am binary tree\n");
  }
};

struct AVLTree : public Tree {
  virtual void Print() {
    printf("I am AVL tree\n");
  }
};

// registry to get the trees
struct TreeFactory
    : public bubblefs::FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
};

#define REGISTER_TREE(Name)                                             \
  COMMON_REGISTRY_REGISTER(::tree::TreeFactory, TreeFactory, Name)        \
  .set_body([]() { return new Name(); } )
  
}  // namespace tree

namespace bubblefs {
// usually this sits on a seperate file
COMMON_REGISTRY_ENABLE(tree::TreeFactory);
} // namespace bubblefs

namespace tree {
// Register the trees, can be in seperate files
REGISTER_TREE(BinaryTree)
.describe("This is a binary tree.");
REGISTER_TREE(AVLTree);
} // namespace tree

int main(int argc, char *argv[]) {
  // construct a binary tree
  tree::Tree *binary = bubblefs::Registry<tree::TreeFactory>::Find("BinaryTree")->body();
  binary->Print();
  // construct a binary tree
  tree::Tree *avl = bubblefs::Registry<tree::TreeFactory>::Find("AVLTree")->body();
  avl->Print();
  delete binary; delete avl;
  return 0;
}