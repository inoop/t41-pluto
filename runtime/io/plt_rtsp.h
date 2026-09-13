/* A small RTSP server: enough of RFC 2326 to put H.264 in front of VLC.
 *
 * Part of `io/` -- the outside world.  It includes nothing from the runtime
 * stack and knows nothing about the network that produced the picture; it is
 * handed NAL units and a timestamp.
 *
 * RTP is carried INTERLEAVED over the RTSP TCP connection (RFC 2326 s10.12),
 * not over UDP.  That is a deliberate simplification and a practical one: one
 * socket instead of four, no port negotiation, no firewall or NAT trouble, and
 * no packet loss to conceal on a camera whose whole job is to be looked at over
 * a LAN.  A client that asks for UDP is answered 461 and retries over TCP,
 * which both VLC and ffplay do.
 */
#ifndef PLT_IO_RTSP_H
#define PLT_IO_RTSP_H

#include <stddef.h>
#include <stdint.h>

typedef struct plt_rtsp plt_rtsp_t;

/* Start listening.  Returns NULL on failure with `err` set.  Nothing is sent
 * until a client has completed SETUP and PLAY. */
plt_rtsp_t *plt_rtsp_start(int port, int width, int height, char *err, size_t errlen);

/* Accept new clients and answer pending requests.  Never blocks; call it once
 * per frame.  Returns the number of clients currently playing. */
int plt_rtsp_poll(plt_rtsp_t *s);

/* Hand over one Annex-B buffer -- start codes and all, as the encoder produced
 * it.  It is split into NAL units, SPS/PPS are remembered for the next
 * DESCRIBE, and the rest is packetized to every playing client.  `ts90k` is the
 * frame's presentation time in the RTP 90 kHz clock. */
void plt_rtsp_send_annexb(plt_rtsp_t *s, const uint8_t *buf, uint32_t len, uint32_t ts90k);

/* Nonzero once an SPS and a PPS have been seen, i.e. DESCRIBE can be answered
 * with a complete SDP.  Worth waiting for before advertising the URL. */
int plt_rtsp_ready(const plt_rtsp_t *s);

void plt_rtsp_stop(plt_rtsp_t *s);

#endif /* PLT_IO_RTSP_H */
