#include "plt_rtsp.h"

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_CLIENTS 4
#define REQ_MAX     2048
#define RTP_MTU     1400          /* payload bytes per RTP packet */
#define PT_H264     96
#define CH_RTP      0             /* interleaved channel for RTP  */

enum { ST_INIT = 0, ST_READY, ST_PLAYING };

typedef struct {
    int      fd;
    int      state;
    char     req[REQ_MAX];
    int      reqlen;
    uint16_t seq;
    uint32_t ssrc;
} client_t;

struct plt_rtsp {
    int      listen_fd;
    int      width, height;
    client_t cl[MAX_CLIENTS];
    uint8_t  sps[64], pps[64];
    uint32_t sps_len, pps_len;
};

/* --- base64, for sprop-parameter-sets ------------------------------------- */
static void b64(const uint8_t *in, uint32_t n, char *out, size_t outsz)
{
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (uint32_t i = 0; i < n && o + 5 < outsz; i += 3) {
        const uint32_t a = in[i];
        const uint32_t b = i + 1 < n ? in[i + 1] : 0;
        const uint32_t c = i + 2 < n ? in[i + 2] : 0;
        out[o++] = T[a >> 2];
        out[o++] = T[((a & 3) << 4) | (b >> 4)];
        out[o++] = i + 1 < n ? T[((b & 15) << 2) | (c >> 6)] : '=';
        out[o++] = i + 2 < n ? T[c & 63] : '=';
    }
    out[o] = 0;
}

static void set_nonblock(int fd)
{
    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

/* Write the whole buffer, tolerating short writes.  A wedged client is dropped
 * rather than allowed to stall the capture loop. */
static int write_all(int fd, const uint8_t *p, size_t n)
{
    /* The socket is non-blocking so that a client which stops reading cannot
     * stall the capture loop.  Spin only briefly on a full send buffer, then
     * give up and let the caller drop the client -- a viewer that cannot keep
     * up is not a reason to drop frames for everyone else. */
    int stalls = 0;
    while (n) {
        const ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w > 0) { p += w; n -= (size_t)w; stalls = 0; continue; }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (++stalls > 2000) return -1;
            usleep(200);
            continue;
        }
        return -1;
    }
    return 0;
}

plt_rtsp_t *plt_rtsp_start(int port, int width, int height, char *err, size_t errlen)
{
    struct sockaddr_in a;
    plt_rtsp_t *s = calloc(1, sizeof *s);
    int on = 1;

    if (!s) { if (err) snprintf(err, errlen, "out of memory"); return NULL; }
    s->width = width; s->height = height;
    for (int i = 0; i < MAX_CLIENTS; i++) s->cl[i].fd = -1;

    s->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s->listen_fd < 0) { if (err) snprintf(err, errlen, "socket: %s", strerror(errno)); free(s); return NULL; }
    setsockopt(s->listen_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);

    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(s->listen_fd, (struct sockaddr *)&a, sizeof a) ||
        listen(s->listen_fd, 4)) {
        if (err) snprintf(err, errlen, "bind/listen port %d: %s", port, strerror(errno));
        close(s->listen_fd); free(s); return NULL;
    }
    set_nonblock(s->listen_fd);
    return s;
}

int plt_rtsp_ready(const plt_rtsp_t *s) { return s && s->sps_len && s->pps_len; }

static void drop(client_t *c)
{
    if (c->fd >= 0) close(c->fd);
    c->fd = -1; c->state = ST_INIT; c->reqlen = 0;
}

/* --- the four methods we answer ------------------------------------------- */

static int hdr_cseq(const char *req)
{
    const char *p = strstr(req, "CSeq:");
    if (!p) p = strstr(req, "Cseq:");
    return p ? atoi(p + 5) : 0;
}

static void reply(client_t *c, int cseq, const char *status, const char *extra)
{
    char buf[1024];
    const int n = snprintf(buf, sizeof buf,
                           "RTSP/1.0 %s\r\nCSeq: %d\r\nServer: pluto\r\n%s\r\n",
                           status, cseq, extra ? extra : "");
    write_all(c->fd, (const uint8_t *)buf, (size_t)n);
}

/* The Request-URI out of the request line, so DESCRIBE can echo it back as a
 * Content-Base.  Without one a client has to guess what `a=control:track1` is
 * relative to, and they do not all guess the same way. */
static void request_uri(const char *req, char *out, size_t n)
{
    const char *p = strchr(req, ' ');
    size_t i = 0;
    out[0] = 0;
    if (!p) return;
    for (p++; *p && *p != ' ' && *p != '\r' && i + 1 < n; p++) out[i++] = *p;
    out[i] = 0;
    if (i && out[i - 1] != '/' && i + 1 < n) { out[i++] = '/'; out[i] = 0; }
}

static void do_describe(plt_rtsp_t *s, client_t *c, int cseq, const char *req)
{
    char sdp[768], hdr[320], s64[128], p64[128], uri[160];

    if (!plt_rtsp_ready(s)) { reply(c, cseq, "503 Service Unavailable", ""); return; }
    request_uri(req, uri, sizeof uri);
    b64(s->sps, s->sps_len, s64, sizeof s64);
    b64(s->pps, s->pps_len, p64, sizeof p64);

    /* profile-level-id is the three bytes after the SPS NAL header; a client
     * needs it to pick a decoder before any picture has arrived. */
    const int n = snprintf(sdp, sizeof sdp,
        "v=0\r\n"
        "o=- 0 0 IN IP4 0.0.0.0\r\n"
        "s=pluto\r\n"
        /* RFC 4566 wants a connection line even when the transport is
         * interleaved and the address is therefore never used. */
        "c=IN IP4 0.0.0.0\r\n"
        "t=0 0\r\n"
        "a=tool:pluto\r\n"
        "m=video 0 RTP/AVP %d\r\n"
        "a=rtpmap:%d H264/90000\r\n"
        "a=fmtp:%d packetization-mode=1;profile-level-id=%02X%02X%02X;"
        "sprop-parameter-sets=%s,%s\r\n"
        "a=control:track1\r\n",
        PT_H264, PT_H264, PT_H264,
        s->sps_len > 1 ? s->sps[1] : 0x42, s->sps_len > 2 ? s->sps[2] : 0,
        s->sps_len > 3 ? s->sps[3] : 0x1E, s64, p64);

    snprintf(hdr, sizeof hdr,
             "Content-Type: application/sdp\r\nContent-Base: %s\r\n"
             "Content-Length: %d\r\n", uri[0] ? uri : "rtsp://0.0.0.0/", n);
    reply(c, cseq, "200 OK", hdr);
    write_all(c->fd, (const uint8_t *)sdp, (size_t)n);
}

static void do_setup(client_t *c, int cseq, const char *req)
{
    /* Interleaved only -- see plt_rtsp.h.  A UDP request is refused with 461,
     * which is the code that makes a client retry over TCP rather than fail. */
    if (!strstr(req, "interleaved") && !strstr(req, "RTP/AVP/TCP")) {
        reply(c, cseq, "461 Unsupported Transport", "");
        return;
    }
    reply(c, cseq, "200 OK",
          "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\nSession: 12345678\r\n");
    c->state = ST_READY;
}

static void handle(plt_rtsp_t *s, client_t *c)
{
    const int cseq = hdr_cseq(c->req);

    if      (!strncmp(c->req, "OPTIONS", 7))
        reply(c, cseq, "200 OK",
              "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER\r\n");
    else if (!strncmp(c->req, "DESCRIBE", 8)) do_describe(s, c, cseq, c->req);
    else if (!strncmp(c->req, "SETUP", 5))    do_setup(c, cseq, c->req);
    else if (!strncmp(c->req, "PLAY", 4)) {
        reply(c, cseq, "200 OK", "Session: 12345678\r\nRange: npt=0.000-\r\n");
        c->state = ST_PLAYING;
    }
    else if (!strncmp(c->req, "GET_PARAMETER", 13) ||
             !strncmp(c->req, "SET_PARAMETER", 13))
        reply(c, cseq, "200 OK", "Session: 12345678\r\n");   /* keepalive */
    else if (!strncmp(c->req, "TEARDOWN", 8)) { reply(c, cseq, "200 OK", ""); drop(c); }
    else reply(c, cseq, "501 Not Implemented", "");
}

int plt_rtsp_poll(plt_rtsp_t *s)
{
    int playing = 0, fd;

    if (!s) return 0;
    while ((fd = accept(s->listen_fd, NULL, NULL)) >= 0) {
        int slot = -1, on = 1;
        for (int i = 0; i < MAX_CLIENTS; i++) if (s->cl[i].fd < 0) { slot = i; break; }
        if (slot < 0) { close(fd); continue; }
        set_nonblock(fd);
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
        s->cl[slot].fd = fd; s->cl[slot].state = ST_INIT; s->cl[slot].reqlen = 0;
        s->cl[slot].seq = 0; s->cl[slot].ssrc = 0xAC1DBEEFu ^ (uint32_t)slot;
    }

    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c = &s->cl[i];
        if (c->fd < 0) continue;
        for (;;) {
            const ssize_t r = recv(c->fd, c->req + c->reqlen,
                                   (size_t)(REQ_MAX - 1 - c->reqlen), 0);
            if (r > 0) {
                c->reqlen += (int)r;
                c->req[c->reqlen] = 0;
                /* One request per blank line; RTSP requests here carry no body. */
                char *end;
                while ((end = strstr(c->req, "\r\n\r\n")) != NULL) {
                    const int used = (int)(end - c->req) + 4;
                    const char save = c->req[used];
                    c->req[used] = 0;
                    handle(s, c);
                    /* TEARDOWN drops the client inside handle(), which zeroes
                     * reqlen -- consuming the request afterwards would compute
                     * a negative length as size_t and walk off the heap. */
                    if (c->fd < 0) break;
                    c->req[used] = save;
                    memmove(c->req, c->req + used, (size_t)(c->reqlen - used) + 1);
                    c->reqlen -= used;
                }
                if (c->fd < 0) break;
                if (c->reqlen >= REQ_MAX - 1) { drop(c); break; }
                continue;
            }
            if (r == 0) { drop(c); break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            drop(c); break;
        }
        if (c->fd >= 0 && c->state == ST_PLAYING) playing++;
    }
    return playing;
}

/* --- RTP ------------------------------------------------------------------ */

/* One interleaved RTP packet: the 4-byte framing, the 12-byte RTP header, then
 * the payload the caller has already shaped (single NAL or FU-A fragment). */
static int send_rtp(client_t *c, const uint8_t *payload, uint32_t n,
                    uint32_t ts, int marker)
{
    uint8_t h[16];
    h[0] = '$'; h[1] = CH_RTP;
    h[2] = (uint8_t)((12 + n) >> 8); h[3] = (uint8_t)(12 + n);
    h[4] = 0x80; h[5] = (uint8_t)(PT_H264 | (marker ? 0x80 : 0));
    h[6] = (uint8_t)(c->seq >> 8); h[7] = (uint8_t)c->seq;
    h[8]  = (uint8_t)(ts >> 24); h[9]  = (uint8_t)(ts >> 16);
    h[10] = (uint8_t)(ts >> 8);  h[11] = (uint8_t)ts;
    h[12] = (uint8_t)(c->ssrc >> 24); h[13] = (uint8_t)(c->ssrc >> 16);
    h[14] = (uint8_t)(c->ssrc >> 8);  h[15] = (uint8_t)c->ssrc;
    c->seq++;
    if (write_all(c->fd, h, sizeof h)) return -1;
    return write_all(c->fd, payload, n);
}

/* RFC 6184: small NALs go whole, large ones are split FU-A. */
static void send_nal(plt_rtsp_t *s, const uint8_t *nal, uint32_t len, uint32_t ts)
{
    for (int i = 0; i < MAX_CLIENTS; i++) {
        client_t *c = &s->cl[i];
        if (c->fd < 0 || c->state != ST_PLAYING) continue;

        if (len <= RTP_MTU) {
            if (send_rtp(c, nal, len, ts, 1)) drop(c);
            continue;
        }
        const uint8_t ind = (uint8_t)((nal[0] & 0xE0) | 28);   /* FU-A */
        const uint8_t typ = (uint8_t)(nal[0] & 0x1F);
        uint32_t off = 1, left = len - 1;
        int first = 1, bad = 0;
        while (left && !bad) {
            const uint32_t take = left > RTP_MTU - 2 ? RTP_MTU - 2 : left;
            uint8_t pkt[RTP_MTU];
            pkt[0] = ind;
            pkt[1] = (uint8_t)(typ | (first ? 0x80 : 0) | (take == left ? 0x40 : 0));
            memcpy(pkt + 2, nal + off, take);
            bad = send_rtp(c, pkt, take + 2, ts, take == left) != 0;
            off += take; left -= take; first = 0;
        }
        if (bad) drop(c);
    }
}

void plt_rtsp_send_annexb(plt_rtsp_t *s, const uint8_t *buf, uint32_t len, uint32_t ts90k)
{
    uint32_t i = 0;

    if (!s || !buf || len < 4) return;
    while (i + 3 < len) {
        uint32_t sc, start, end;
        /* find a start code */
        if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1)            sc = 3;
        else if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 0 && buf[i + 3] == 1) sc = 4;
        else { i++; continue; }

        start = i + sc;
        for (end = start; end + 3 < len; end++)
            if (buf[end] == 0 && buf[end + 1] == 0 &&
                (buf[end + 2] == 1 || (buf[end + 2] == 0 && buf[end + 3] == 1))) break;
        if (end + 3 >= len) end = len;
        if (end > start) {
            const uint8_t type = (uint8_t)(buf[start] & 0x1F);
            if (type == 7 && end - start <= sizeof s->sps) {
                s->sps_len = end - start; memcpy(s->sps, buf + start, s->sps_len);
            } else if (type == 8 && end - start <= sizeof s->pps) {
                s->pps_len = end - start; memcpy(s->pps, buf + start, s->pps_len);
            }
            send_nal(s, buf + start, end - start, ts90k);
        }
        i = end;
    }
}

void plt_rtsp_stop(plt_rtsp_t *s)
{
    if (!s) return;
    for (int i = 0; i < MAX_CLIENTS; i++) drop(&s->cl[i]);
    if (s->listen_fd >= 0) close(s->listen_fd);
    free(s);
}
