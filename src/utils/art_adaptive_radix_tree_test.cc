
#include "utils/art_adaptive_radix_tree.h"
#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define fail_unless(x) assert(x)

namespace bubblefs {
namespace art {
  
typedef struct {
    int count;
    int max_count;
    const char **expected;
} prefix_data;

static int test_prefix_cb(void *data, const unsigned char *k, uint32_t k_len, void *val) {
    prefix_data *p = (prefix_data*)data;
    fail_unless(p->count < p->max_count);
    fail_unless(memcmp(k, p->expected[p->count], k_len) == 0);
    p->count++;
    return 0;
}

void test_art_iter_prefix()
{
    art_tree t;
    int res = art_tree_init(&t);
    fail_unless(res == 0);

    const char *s = "api.foo.bar";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    s = "api.foo.baz";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    s = "api.foe.fum";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    s = "abc.123.456";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    s = "api.foo";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    s = "api";
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, NULL));

    // Iterate over api
    const char *expected[] = {"api", "api.foe.fum", "api.foo", "api.foo.bar", "api.foo.baz"};
    prefix_data p = { 0, 5, expected };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"api", 3, test_prefix_cb, &p));
    fail_unless(p.count == p.max_count);

    // Iterate over 'a'
    const char *expected2[] = {"abc.123.456", "api", "api.foe.fum", "api.foo", "api.foo.bar", "api.foo.baz"};
    prefix_data p2 = { 0, 6, expected2 };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"a", 1, test_prefix_cb, &p2));
    fail_unless(p2.count == p2.max_count);

    // Check a failed iteration
    prefix_data p3 = { 0, 0, NULL };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"b", 1, test_prefix_cb, &p3));
    fail_unless(p3.count == 0);

    // Iterate over api.
    const char *expected4[] = {"api.foe.fum", "api.foo", "api.foo.bar", "api.foo.baz"};
    prefix_data p4 = { 0, 4, expected4 };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"api.", 4, test_prefix_cb, &p4));
    fail_unless(p4.count == p4.max_count);

    // Iterate over api.foo.ba
    const char *expected5[] = {"api.foo.bar"};
    prefix_data p5 = { 0, 1, expected5 };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"api.foo.bar", 11, test_prefix_cb, &p5));
    fail_unless(p5.count == p5.max_count);

    // Check a failed iteration on api.end
    prefix_data p6 = { 0, 0, NULL };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"api.end", 7, test_prefix_cb, &p6));
    fail_unless(p6.count == 0);

    // Iterate over empty prefix
    prefix_data p7 = { 0, 6, expected2 };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"", 0, test_prefix_cb, &p7));
    fail_unless(p7.count == p7.max_count);

    res = art_tree_destroy(&t);
    fail_unless(res == 0);
}

void test_art_long_prefix()
{
    art_tree t;
    int res = art_tree_init(&t);
    fail_unless(res == 0);

    uintptr_t v;
    const char *s;

    s = "this:key:has:a:long:prefix:3";
    v = 3;
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, (void*)v));

    s = "this:key:has:a:long:common:prefix:2";
    v = 2;
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, (void*)v));

    s = "this:key:has:a:long:common:prefix:1";
    v = 1;
    fail_unless(NULL == art_insert(&t, (unsigned char*)s, strlen(s)+1, (void*)v));

    // Search for the keys
    s = "this:key:has:a:long:common:prefix:1";
    fail_unless(1 == (uintptr_t)art_search(&t, (unsigned char*)s, strlen(s)+1));

    s = "this:key:has:a:long:common:prefix:2";
    fail_unless(2 == (uintptr_t)art_search(&t, (unsigned char*)s, strlen(s)+1));

    s = "this:key:has:a:long:prefix:3";
    fail_unless(3 == (uintptr_t)art_search(&t, (unsigned char*)s, strlen(s)+1));


    const char *expected[] = {
        "this:key:has:a:long:common:prefix:1",
        "this:key:has:a:long:common:prefix:2",
        "this:key:has:a:long:prefix:3",
    };
    prefix_data p = { 0, 3, expected };
    fail_unless(!art_iter_prefix(&t, (unsigned char*)"this:key:has", 12, test_prefix_cb, &p));
    fail_unless(p.count == p.max_count);

    res = art_tree_destroy(&t);
    fail_unless(res == 0);
}

void test_art_max_prefix_len_scan_prefix()
{
    art_tree t;
    int res = art_tree_init(&t);
    fail_unless(res == 0);

    const char* key1 = "foobarbaz1-test1-foo";
    fail_unless(NULL == art_insert(&t, (const unsigned char*)key1, strlen(key1)+1, NULL));

    const char *key2 = "foobarbaz1-test1-bar";
    fail_unless(NULL == art_insert(&t, (const unsigned char*)key2, strlen(key2)+1, NULL));

    const char *key3 = "foobarbaz1-test2-foo";
    fail_unless(NULL == art_insert(&t, (const unsigned char*)key3, strlen(key3)+1, NULL));

    fail_unless(art_size(&t) == 3);

    // Iterate over api
    const char *expected[] = {key2, key1};
    prefix_data p = { 0, 2, expected };
    const char *prefix = "foobarbaz1-test1";
    fail_unless(!art_iter_prefix(&t, (unsigned char*)prefix, strlen(prefix), test_prefix_cb, &p));
    fail_unless(p.count == p.max_count);

    res = art_tree_destroy(&t);
    fail_unless(res == 0);
}

} // namespace art
} // namespace bubblefs