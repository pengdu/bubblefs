/*
 * GIT - The information manager from hell
 *
 * Copyright (C) Linus Torvalds, 2005
 */

// git/git-compat-util.h

#ifndef BUBBLEFS_PLATFORM_GIT_USAGE_H_
#define BUBBLEFS_PLATFORM_GIT_USAGE_H_

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "platform/macros.h"

/*
 *
 * `die`, `usage`, `error`, and `warning` report errors of various
kinds.

- `die` is for fatal application errors.  It prints a message to
  the user and exits with status 128.

- `usage` is for errors in command line usage.  After printing its
  message, it exits with status 129.  (See also `usage_with_options`
  in the link:api-parse-options.html[parse-options API].)

- `error` is for non-fatal library errors.  It prints a message
  to the user and returns -1 for convenience in signaling the error
  to the caller.

- `warning` is for reporting situations that probably should not
  occur but which the user (and Git) can continue to work around
  without running into too many problems.  Like `error`, it
  returns -1 after reporting the situation to the caller.

Customizable error handlers
---------------------------

The default behavior of `die` and `error` is to write a message to
stderr and then exit or return as appropriate.  This behavior can be
overridden using `set_die_routine` and `set_error_routine`.  For
example, "git daemon" uses set_die_routine to write the reason `die`
was called to syslog before exiting.

Library errors
--------------

Functions return a negative integer on error.  Details beyond that
vary from function to function:

- Some functions return -1 for all errors.  Others return a more
  specific value depending on how the caller might want to react
  to the error.

- Some functions report the error to stderr with `error`,
  while others leave that for the caller to do.

- errno is not meaningful on return from most functions (except
  for thin wrappers for system calls).

Check the function's API documentation to be sure.

Caller-handled errors
---------------------

An increasing number of functions take a parameter 'struct strbuf *err'.
On error, such functions append a message about what went wrong to the
'err' strbuf.  The message is meant to be complete enough to be passed
to `die` or `error` as-is.  For example:

        if (ref_transaction_commit(transaction, &err))
                die("%s", err.buf);

The 'err' parameter will be untouched if no error occurred, so multiple
function calls can be chained:

        t = ref_transaction_begin(&err);
        if (!t ||
            ref_transaction_update(t, "HEAD", ..., &err) ||
            ret_transaction_commit(t, &err))
                die("%s", err.buf);

The 'err' parameter must be a pointer to a valid strbuf.  To silence
a message, pass a strbuf that is explicitly ignored:

        if (thing_that_can_fail_in_an_ignorable_way(..., &err))
                strbuf_reset(&err);
*/

namespace bubblefs {
namespace mygit {
 
extern void set_panic_routine(NORETURN_PTR void (*routine)(const char *err, va_list params));
extern void set_error_routine(void (*routine)(const char *err, va_list params));
extern void (*get_error_routine(void))(const char *err, va_list params);
extern void set_warn_routine(void (*routine)(const char *warn, va_list params));
extern void (*get_warn_routine(void))(const char *warn, va_list params);
extern void set_panic_is_recursing_routine(int (*routine)(void));

/* General helper functions */
extern void vreportf(const char *prefix, const char *err, va_list params);
extern NORETURN void usage(const char *err);
extern NORETURN void usagef(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern NORETURN void panic(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern NORETURN void panic_errno(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern int error(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern int error_errno(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern void warning(const char *err, ...) __attribute__((format (printf, 1, 2)));
extern void warning_errno(const char *err, ...) __attribute__((format (printf, 1, 2)));

}  // namespace mygit
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_GIT_USAGE_H_