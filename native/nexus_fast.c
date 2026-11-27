/*
 * nexus_fast.c -- compiled C99 hot-path helpers for Nexus.
 *
 * Two byte-oriented primitives that are called on every rendered line:
 *
 *   nexus_strip_ansi()       -- strip ANSI CSI escapes from a byte buffer
 *   nexus_find_tail_offset() -- find the byte offset of the trailing N lines
 *
 * Both are pure POSIX/libc: no allocation, no globals, no external deps.
 * The Python side (nexus/native_bridge.py) is the only caller and always
 * supplies explicit lengths, so embedded NUL bytes are handled correctly.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <sys/types.h>

#define NEXUS_CHUNK 65536

static int nexus_is_csi_param(unsigned char c)
{
    return (c >= '0' && c <= '9') || c == ';';
}

static int nexus_is_ascii_alpha(unsigned char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

/*
 * Strip escape sequences that match the Python regex
 *
 *     \x1b\[[0-9;]*[A-Za-z]
 *
 * Semantics are copied from re.sub(): a candidate is only removed when the
 * ESC, the '[' and the trailing ASCII letter are all present. Anything
 * incomplete (bare ESC, ESC at EOF, "ESC[31" at EOF, "ESC[12x") is copied
 * literally and scanning continues at the ESC + 1 position, exactly like the
 * regex engine which cannot start a new match inside a failed match.
 *
 * Returns the total stripped length (the length the result would have if the
 * destination were large enough). When dst is non-NULL and dst_cap > 0, at
 * most dst_cap - 1 payload bytes are copied and dst is NUL terminated, so the
 * function is always a safe bounded write. Callers that need the whole string
 * pass dst_cap >= returned_length + 1.
 */
size_t nexus_strip_ansi(const char *src, size_t src_len, char *dst, size_t dst_cap)
{
    size_t i = 0;
    size_t written = 0;
    size_t limit;

    if (src == NULL) {
        return 0;
    }

    /* Reserve one byte for the NUL terminator, if there is room for it. */
    limit = (dst != NULL && dst_cap > 0) ? dst_cap - 1 : 0;

    while (i < src_len) {
        unsigned char c = (unsigned char)src[i];

        if (c == 0x1b && i + 1 < src_len && src[i + 1] == '[') {
            size_t j = i + 2;

            while (j < src_len && nexus_is_csi_param((unsigned char)src[j])) {
                j += 1;
            }
            if (j < src_len && nexus_is_ascii_alpha((unsigned char)src[j])) {
                i = j + 1; /* drop the whole CSI sequence */
                continue;
            }
        }

        if (written < limit) {
            dst[written] = src[i];
        }
        written += 1;
        i += 1;
    }

    if (dst != NULL && dst_cap > 0) {
        dst[written < limit ? written : limit] = '\0';
    }

    return written;
}

/*
 * Return the byte offset such that file[offset:] holds the last `keep_lines`
 * lines, or 0 when the file has fewer lines than requested.
 *
 * `keep_lines == 0` returns the file size (0 for an empty file). A single
 * trailing newline is ignored before counting, which matches Python's
 * splitlines() view of a newline-terminated file.
 *
 * Returns -1 when path is NULL, cannot be opened, or cannot be seeked.
 */
long long nexus_find_tail_offset(const char *path, size_t keep_lines)
{
    FILE *fp;
    off_t end;
    off_t effective_end;
    off_t chunk_end;
    size_t newlines = 0;
    char buf[NEXUS_CHUNK];
    int found = 0;
    long long result = 0;
    unsigned char last = 0;

    if (path == NULL) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    if (fseeko(fp, (off_t)0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    end = ftello(fp);
    if (end < 0) {
        fclose(fp);
        return -1;
    }

    if (keep_lines == 0) {
        fclose(fp);
        return (long long)end;
    }
    if (end == 0) {
        fclose(fp);
        return 0;
    }

    /* Ignore exactly one trailing newline. */
    effective_end = end;
    if (fseeko(fp, end - 1, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    if (fread(&last, 1, 1, fp) == 1 && last == '\n') {
        effective_end = end - 1;
    }
    if (effective_end == 0) {
        fclose(fp);
        return 0;
    }

    chunk_end = effective_end;
    while (chunk_end > 0 && !found) {
        off_t chunk_start = chunk_end - (off_t)NEXUS_CHUNK;
        size_t want;
        size_t got = 0;

        if (chunk_start < 0) {
            chunk_start = 0;
        }
        want = (size_t)(chunk_end - chunk_start);

        if (fseeko(fp, chunk_start, SEEK_SET) != 0) {
            break;
        }
        while (got < want) {
            size_t n = fread(buf + got, 1, want - got, fp);

            if (n == 0) {
                break;
            }
            got += n;
        }
        if (got == 0) {
            break;
        }

        while (got > 0) {
            got -= 1;
            if ((unsigned char)buf[got] == '\n') {
                newlines += 1;
                if (newlines == keep_lines) {
                    /* chunk_start + got is the newline; +1 starts the line. */
                    result = (long long)(chunk_start + (off_t)got + 1);
                    found = 1;
                    break;
                }
            }
        }

        chunk_end = chunk_start;
    }

    fclose(fp);
    return found ? result : 0;
}
