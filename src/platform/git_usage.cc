/*
 * GIT - The information manager from hell
 *
 * Copyright (C) Linus Torvalds, 2005
 */

#include "platform/git_usage.h"

namespace bubblefs {
namespace mygit {

void vreportf(const char *prefix, const char *err, va_list params)
{
        char msg[4096];
        char *p;

        vsnprintf(msg, sizeof(msg), err, params);
        for (p = msg; *p; p++) {
                if (iscntrl(*p) && *p != '\t' && *p != '\n')
                        *p = '?';
        }
        fprintf(stderr, "%s%s\n", prefix, msg);
}

static NORETURN void usage_builtin(const char *err, va_list params)
{
        vreportf("usage: ", err, params);
        exit(129);
}

static NORETURN void panic_builtin(const char *err, va_list params)
{
        vreportf("fatal: ", err, params);
        exit(128);
}

static void error_builtin(const char *err, va_list params)
{
        vreportf("error: ", err, params);
}

static void warn_builtin(const char *warn, va_list params)
{
        vreportf("warning: ", warn, params);
}

static int panic_is_recursing_builtin(void)
{
        static int dying;
        /*
         * Just an arbitrary number X where "a < x < b" where "a" is
         * "maximum number of pthreads we'll ever plausibly spawn" and
         * "b" is "something less than Inf", since the point is to
         * prevent infinite recursion.
         */
        static const int recursion_limit = 1024;

        dying++;
        if (dying > recursion_limit) {
                return 1;
        } else if (dying == 2) {
                warning("panic() called many times. Recursion error or racy threaded death!");
                return 0;
        } else {
                return 0;
        }
}

/* If we are in a dlopen()ed .so write to a global variable would segfault
 * (ugh), so keep things static. */
static NORETURN_PTR void (*usage_routine)(const char *err, va_list params) = usage_builtin;
static NORETURN_PTR void (*panic_routine)(const char *err, va_list params) = panic_builtin;
static void (*error_routine)(const char *err, va_list params) = error_builtin;
static void (*warn_routine)(const char *err, va_list params) = warn_builtin;
static int (*panic_is_recursing)(void) = panic_is_recursing_builtin;

void set_panic_routine(NORETURN_PTR void (*routine)(const char *err, va_list params))
{
        panic_routine = routine;
}

void set_error_routine(void (*routine)(const char *err, va_list params))
{
        error_routine = routine;
}

void (*get_error_routine(void))(const char *err, va_list params)
{
        return error_routine;
}

void set_warn_routine(void (*routine)(const char *warn, va_list params))
{
        warn_routine = routine;
}

void (*get_warn_routine(void))(const char *warn, va_list params)
{
        return warn_routine;
}

void set_panic_is_recursing_routine(int (*routine)(void))
{
        panic_is_recursing = routine;
}

void NORETURN usagef(const char *err, ...)
{
        va_list params;

        va_start(params, err);
        usage_routine(err, params);
        va_end(params);
}

void NORETURN usage(const char *err)
{
        usagef("%s", err);
}

void NORETURN panic(const char *err, ...)
{
        va_list params;

        if (panic_is_recursing()) {
                fputs("fatal: recursion detected in panic handler\n", stderr);
                exit(128);
        }

        va_start(params, err);
        panic_routine(err, params);
        va_end(params);
}

static const char *fmt_with_err(char *buf, int n, const char *fmt)
{
        char str_error[256], *err;
        size_t i, j;

        err = strerror(errno);
        for (i = j = 0; err[i] && j < sizeof(str_error) - 1; ) {
                if ((str_error[j++] = err[i++]) != '%')
                        continue;
                if (j < sizeof(str_error) - 1) {
                        str_error[j++] = '%';
                } else {
                        /* No room to double the '%', so we overwrite it with
                         * '\0' below */
                        j--;
                        break;
                }
        }
        str_error[j] = 0;
        snprintf(buf, n, "%s: %s", fmt, str_error);
        return buf;
}

void NORETURN panic_errno(const char *fmt, ...)
{
        char buf[1024];
        va_list params;

        if (panic_is_recursing()) {
                fputs("fatal: recursion detected in panic_errno handler\n",
                        stderr);
                exit(128);
        }

        va_start(params, fmt);
        panic_routine(fmt_with_err(buf, sizeof(buf), fmt), params);
        va_end(params);
}

#undef error_errno
int error_errno(const char *fmt, ...)
{
        char buf[1024];
        va_list params;

        va_start(params, fmt);
        error_routine(fmt_with_err(buf, sizeof(buf), fmt), params);
        va_end(params);
        return -1;
}

#undef error
int error(const char *err, ...)
{
        va_list params;

        va_start(params, err);
        error_routine(err, params);
        va_end(params);
        return -1;
}

void warning_errno(const char *warn, ...)
{
        char buf[1024];
        va_list params;

        va_start(params, warn);
        warn_routine(fmt_with_err(buf, sizeof(buf), warn), params);
        va_end(params);
}

void warning(const char *warn, ...)
{
        va_list params;

        va_start(params, warn);
        warn_routine(warn, params);
        va_end(params);
}

static NORETURN void BUG_vfl(const char *file, int line, const char *fmt, va_list params)
{
        char prefix[256];

        /* truncation via snprintf is OK here */
        if (file)
                snprintf(prefix, sizeof(prefix), "BUG: %s:%d: ", file, line);
        else
                snprintf(prefix, sizeof(prefix), "BUG: ");

        vreportf(prefix, fmt, params);
        abort();
}

NORETURN void BUG_fl(const char *file, int line, const char *fmt, ...)
{
        va_list ap;
        va_start(ap, fmt);
        BUG_vfl(file, line, fmt, ap);
        va_end(ap);
}

NORETURN void BUG(const char *fmt, ...)
{
        va_list ap;
        va_start(ap, fmt);
        BUG_vfl(NULL, 0, fmt, ap);
        va_end(ap);
}

}  // namespace mygit
}  // namespace bubblefs