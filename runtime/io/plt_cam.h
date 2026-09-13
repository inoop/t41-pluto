/* The camera: NV12 frames from the ISP, small enough to work with.
 *
 * This is `io/` -- the outside world.  It includes NOTHING from the runtime
 * stack (hal/, core/, exec/, plt_engine), and only tools/ includes it, so the
 * layering of docs/ARCHITECTURE.md stays linear and hal/ stays innocent of
 * cameras.  It is also the only part of pluto that touches vendor code: the
 * Ingenic IMP SDK, behind these four functions.  Nothing in the neural path
 * knows it exists.
 *
 * The ISP scales in hardware, so a frame arrives already small and the CPU
 * never resizes anything.  It does NOT crop: this camera's ISP wedges -- and
 * on one configuration panicked the board -- when a crop and a scale are asked
 * for together, so we capture the whole field of view at an aspect-preserving
 * size and hand out the centred rectangle for the caller to read out of it.
 * That rectangle is exactly what a centre-square hardware crop would have
 * given, and reading it costs nothing: it is where the preprocessing loop
 * starts, not a copy.
 */
#ifndef PLT_IO_CAM_H
#define PLT_IO_CAM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int      chn;                 /* framesource channel                     */
    int      cap_w, cap_h;        /* what the ISP scales to = the whole FOV  */
    int      out_w, out_h;        /* the centred rectangle you asked for     */
    int      crop_x, crop_y;      /* where it sits in the captured frame     */
    int      sensor_w, sensor_h;  /* queried, not assumed                    */
    int      fps;                 /* fps_num / fps_den, for printing         */
    int      fps_num, fps_den;    /* the sensor reports a packed rational    */
    uint32_t stride;              /* Y-plane stride, derived from a frame    */
    void    *frame;               /* IMPFrameInfo * of the frame being held  */
    int      opened, streaming;
    /* The IMPSensorInfo we opened with: IMP_ISP_DelSensor() wants the same one
     * back at teardown, and io/ does not expose libimp's types.  Sized well
     * past the real struct (plt_cam.c static-asserts it) and declared as
     * long long so it is aligned for one, which keeps <imp/...> out of here. */
    unsigned long long sensor[32];

    /* The stream channel, when one was asked for: a SECOND framesource channel
     * bound straight to the hardware H.264 encoder.  The ISP feeds the encoder
     * itself, so streaming costs the CPU nothing but the fetching of finished
     * NAL units -- the network's channel above is untouched by it. */
    int stream_chn, enc_grp, enc_chn;
    int stream_on, recving;
    void *stream_held;            /* IMPEncoderStream * currently held */
    int osd_grp, osd_on, n_rgn;
    int rgn[8];                   /* IMPRgnHandle, one per drawable box */
} plt_cam_t;

/* Boxes are drawn by the ISP's OSD block, which sits between the framesource
 * and the encoder: the CPU never touches a pixel of the stream. */
#define PLT_CAM_MAX_BOXES 8

/* A box in STREAM-channel pixels -- the network sees a crop of that view, so a
 * detection has to be offset by cam->crop_x/crop_y before it lands here. */
typedef struct { int x0, y0, x1, y1; } plt_cam_box_t;

/* How the sensor's field of view is made to fit the network's input.
 *
 *   CROP       the ISP scales uniformly until the input fits INSIDE the frame,
 *              and the middle is read out.  Most magnification, but on a 16:9
 *              sensor feeding a square input it throws away ~46% of the width.
 *   LETTERBOX  the ISP scales uniformly until the whole frame fits INSIDE the
 *              input, and the remainder is padded.  This is what YOLOX's own
 *              preprocessing does (compile/pack_input.py) and therefore what
 *              the model was quantized against.
 *   STRETCH    the ISP scales each axis independently to the input size.  No
 *              padding and no lost field of view, but the aspect ratio is not
 *              preserved -- a 16:9 sensor into a square input makes everything
 *              1.78x too tall, which the model has never seen.
 *
 * None of them changes how much work the network does: it runs its full input
 * size either way.  CROP buys magnification, nothing else.
 */
typedef enum {
    PLT_CAM_FIT_CROP = 0,
    PLT_CAM_FIT_LETTERBOX,
    PLT_CAM_FIT_STRETCH
} plt_cam_fit_t;

/* Ask plt_cam_open() for a stream channel as well.  NULL means none. */
typedef struct {
    int kbps;                     /* target bitrate; 0 picks a sane default */
    int fps;                      /* encoder frame rate; 0 follows the sensor */
} plt_cam_stream_cfg_t;

/* One NAL-carrying buffer, Annex-B framed as the encoder produced it. */
typedef struct { const uint8_t *data; uint32_t len; } plt_cam_pack_t;

/* Bring up sensor + ISP + one framesource channel.  `out_w` x `out_h` is the
 * region you want; the channel is configured for the smallest aspect-preserving
 * capture that contains it, and cam->crop_x/crop_y say where it landed.
 * 0 on success; on failure `err` says which call failed and with what code. */
int plt_cam_open(plt_cam_t *cam, int out_w, int out_h, plt_cam_fit_t fit,
                 const plt_cam_stream_cfg_t *stream, char *err, size_t errlen);

/* Block for the next frame.  `*nv12` points into the video-buffer pool and is
 * valid until plt_cam_release().  Copy what you need, then release promptly --
 * holding it starves the pipeline. */
int plt_cam_frame(plt_cam_t *cam, const uint8_t **nv12, char *err, size_t errlen);

void plt_cam_release(plt_cam_t *cam);

/* Fetch one ENCODED frame if the hardware has finished it.  Returns 1 with the
 * packs filled in, 0 if nothing is ready yet, -1 on error.  Never blocks.
 *
 * The encoder runs at the sensor's rate while the detector runs slower, so call
 * this in a loop until it returns 0 -- otherwise finished frames pile up and
 * the view drifts behind the detections. */
int  plt_cam_stream_frame(plt_cam_t *cam, plt_cam_pack_t *packs, int max, int *n,
                          int64_t *ts_us, char *err, size_t errlen);
void plt_cam_stream_release(plt_cam_t *cam);

/* Replace the overlay with these boxes; `n` of 0 clears it.  Boxes past
 * PLT_CAM_MAX_BOXES are dropped. */
int plt_cam_boxes(plt_cam_t *cam, const plt_cam_box_t *b, int n);

/* Strict LIFO teardown.  Safe to call on a partly-opened camera, which is what
 * makes it usable from a signal handler's shutdown path. */
void plt_cam_close(plt_cam_t *cam);

#endif /* PLT_IO_CAM_H */
