
// git/path.c

#ifndef BUBBLEFS_UTILS_GIT_TRIE_H_
#define BUBBLEFS_UTILS_GIT_TRIE_H_

namespace bubblefs {
namespace mygit {

/*
 * A compressed trie.  A trie node consists of zero or more characters that
 * are common to all elements with this prefix, optionally followed by some
 * children.  If value is not NULL, the trie node is a terminal node.
 *
 * For example, consider the following set of strings:
 * abc
 * def
 * definite
 * definition
 *
 * The trie would look like:
 * root: len = 0, children a and d non-NULL, value = NULL.
 *    a: len = 2, contents = bc, value = (data for "abc")
 *    d: len = 2, contents = ef, children i non-NULL, value = (data for "def")
 *       i: len = 3, contents = nit, children e and i non-NULL, value = NULL
 *           e: len = 0, children all NULL, value = (data for "definite")
 *           i: len = 2, contents = on, children all NULL,
 *              value = (data for "definition")
 */
struct trie {
        struct trie *children[256];
        int len;
        char *contents;
        void *value;
};

struct trie *make_trie_node(const char *key, void *value);

void *add_to_trie(struct trie *root, const char *key, void *value);

typedef int (*match_fn)(const char *unmatched, void *data, void *baton);

int trie_find(struct trie *root, const char *key, match_fn fn,
              void *baton);
  
} // namespace mygit  
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_GIT_TRIE_H_