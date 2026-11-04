/*
 * Let CRFsuite open files whose names are not representable in the ANSI
 * code page on Windows.
 *
 * Python passes file names to CRFsuite as UTF-8 (c_string_encoding=utf-8),
 * but on Windows the C runtime's fopen() interprets narrow file names in the
 * ANSI code page, so non-ASCII names end up mangled (e.g. "modèl" becomes
 * "modÃ¨l"). setup.py force-includes this header when compiling with MSVC;
 * in CRFsuite's C sources it redirects fopen() to a wrapper that converts
 * the UTF-8 name to UTF-16 and calls _wfopen(). C++ sources are unaffected.
 */
#ifndef PYCRFSUITE_WIN32_UTF8_FOPEN_H
#define PYCRFSUITE_WIN32_UTF8_FOPEN_H

#if defined(_WIN32) && !defined(__cplusplus)

#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

static wchar_t *pycrfsuite_utf8_to_wide(const char *s)
{
    wchar_t *w;
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) {
        return NULL;
    }
    w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (w != NULL && MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) <= 0) {
        free(w);
        w = NULL;
    }
    return w;
}

static FILE *pycrfsuite_fopen_utf8(const char *name, const char *mode)
{
    FILE *fp = NULL;
    wchar_t *wname = pycrfsuite_utf8_to_wide(name);
    wchar_t *wmode = pycrfsuite_utf8_to_wide(mode);
    if (wname != NULL && wmode != NULL) {
        fp = _wfopen(wname, wmode);
    } else if (wname == NULL && wmode != NULL) {
        /* Not valid UTF-8: keep the previous behaviour. */
        fp = fopen(name, mode);
    }
    free(wname);
    free(wmode);
    return fp;
}

/* Defined after the wrapper, so the wrapper itself still calls the CRT. */
#define fopen pycrfsuite_fopen_utf8

#endif /* _WIN32 && !__cplusplus */

#endif /* PYCRFSUITE_WIN32_UTF8_FOPEN_H */
