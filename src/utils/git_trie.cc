
// git/path.c

#include "utils/git_trie.h"
#include <stdlib.h>
#include <string.h>

namespace bubblefs {
namespace mygit {
  
struct trie *make_trie_node(const char *key, void *value)
{
        struct trie *new_node = (struct trie*)calloc(1, sizeof(*new_node));
        new_node->len = strlen(key);
        if (new_node->len) {
                new_node->contents = (char*)malloc(new_node->len);
                memcpy(new_node->contents, key, new_node->len);
        }
        new_node->value = value;
        return new_node;
}

/*
 * Add a key/value pair to a trie.  The key is assumed to be \0-terminated.
 * If there was an existing value for this key, return it.
 */
void *add_to_trie(struct trie *root, const char *key, void *value)
{
        struct trie *child;
        void *old;
        int i;

        if (!*key) {
                /* we have reached the end of the key */
                old = root->value;
                root->value = value;
                return old;
        }

        for (i = 0; i < root->len; i++) {
                if (root->contents[i] == key[i])
                        continue;

                /*
                 * Split this node: child will contain this node's
                 * existing children.
                 */
                child = (struct trie*)malloc(sizeof(*child));
                memcpy(child->children, root->children, sizeof(root->children));

                child->len = root->len - i - 1;
                if (child->len) {
                        child->contents = strndup(root->contents + i + 1,
                                                  child->len);
                }
                child->value = root->value;
                root->value = NULL;
                root->len = i;

                memset(root->children, 0, sizeof(root->children));
                root->children[(unsigned char)root->contents[i]] = child;

                /* This is the newly-added child. */
                root->children[(unsigned char)key[i]] =
                        make_trie_node(key + i + 1, value);
                return NULL;
        }

        /* We have matched the entire compressed section */
        if (key[i]) {
                child = root->children[(unsigned char)key[root->len]];
                if (child) {
                        return add_to_trie(child, key + root->len + 1, value);
                } else {
                        child = make_trie_node(key + root->len + 1, value);
                        root->children[(unsigned char)key[root->len]] = child;
                        return NULL;
                }
        }

        old = root->value;
        root->value = value;
        return old;
}

/*
 * Search a trie for some key.  Find the longest /-or-\0-terminated
 * prefix of the key for which the trie contains a value.  Call fn
 * with the unmatched portion of the key and the found value, and
 * return its return value.  If there is no such prefix, return -1.
 *
 * The key is partially normalized: consecutive slashes are skipped.
 *
 * For example, consider the trie containing only [refs,
 * refs/worktree] (both with values).
 *
 * | key             | unmatched  | val from node | return value |
 * |-----------------|------------|---------------|--------------|
 * | a               | not called | n/a           | -1           |
 * | refs            | \0         | refs          | as per fn    |
 * | refs/           | /          | refs          | as per fn    |
 * | refs/w          | /w         | refs          | as per fn    |
 * | refs/worktree   | \0         | refs/worktree | as per fn    |
 * | refs/worktree/  | /          | refs/worktree | as per fn    |
 * | refs/worktree/a | /a         | refs/worktree | as per fn    |
 * |-----------------|------------|---------------|--------------|
 *
 */
int trie_find(struct trie *root, const char *key, match_fn fn,
              void *baton)
{
        int i;
        int result;
        struct trie *child;

        if (!*key) {
                /* we have reached the end of the key */
                if (root->value && !root->len)
                        return fn(key, root->value, baton);
                else
                        return -1;
        }

        for (i = 0; i < root->len; i++) {
                /* Partial path normalization: skip consecutive slashes. */
                if (key[i] == '/' && key[i+1] == '/') {
                        key++;
                        continue;
                }
                if (root->contents[i] != key[i])
                        return -1;
        }

        /* Matched the entire compressed section */
        key += i;
        if (!*key)
                /* End of key */
                return fn(key, root->value, baton);

        /* Partial path normalization: skip consecutive slashes */
        while (key[0] == '/' && key[1] == '/')
                key++;

        child = root->children[(unsigned char)*key];
        if (child)
                result = trie_find(child, key + 1, fn, baton);
        else
                result = -1;

        if (result >= 0 || (*key != '/' && *key != 0))
                return result;
        if (root->value)
                return fn(key, root->value, baton);
        else
                return -1;
}

/*
struct common_dir {
        // Not considered garbage for report_linked_checkout_garbage
        unsigned ignore_garbage:1;
        unsigned is_dir:1;
        // Not common even though its parent is
        unsigned exclude:1;
        const char *dirname;
};

static struct common_dir common_list[] = {
        { 0, 1, 0, "branches" },
        { 0, 1, 0, "hooks" },
        { 0, 1, 0, "info" },
        { 0, 0, 1, "info/sparse-checkout" },
        { 1, 1, 0, "logs" },
        { 1, 1, 1, "logs/HEAD" },
        { 0, 1, 1, "logs/refs/bisect" },
        { 0, 1, 0, "lost-found" },
        { 0, 1, 0, "objects" },
        { 0, 1, 0, "refs" },
        { 0, 1, 1, "refs/bisect" },
        { 0, 1, 0, "remotes" },
        { 0, 1, 0, "worktrees" },
        { 0, 1, 0, "rr-cache" },
        { 0, 1, 0, "svn" },
        { 0, 0, 0, "config" },
        { 1, 0, 0, "gc.pid" },
        { 0, 0, 0, "packed-refs" },
        { 0, 0, 0, "shallow" },
        { 0, 0, 0, NULL }
};

static struct trie common_trie;
static int common_trie_done_setup;

static void init_common_trie(void)
{
        struct common_dir *p;

        if (common_trie_done_setup)
                return;

        for (p = common_list; p->dirname; p++)
                add_to_trie(&common_trie, p->dirname, p);

        common_trie_done_setup = 1;
}
*/
  
} // namespace mygit
} // namespace bubblefs