/* Turning a camera frame into what the network wants.
 *
 * Part of `io/` -- it takes the quantization as plain scalars, never a model
 * struct, which is what keeps this a leaf with no dependency on the runtime.
 */
#ifndef PLT_IO_PRE_H
#define PLT_IO_PRE_H

#include <stdint.h>

/* A rectangle inside an NV12 frame.
 *
 * The ISP hands us the whole field of view (see plt_cam.h: it will not crop and
 * scale at the same time), so every consumer here reads a sub-rectangle.  Both
 * planes are addressed off `nv12`, which is why the full captured height has to
 * be here too -- the UV plane starts at nv12 + stride*cap_h, not stride*h.
 *
 * `x0` and `y0` must be even: chroma is subsampled 2x2, and an odd offset lands
 * on the V byte of a UV pair and swaps the colours of the entire image.
 */
typedef struct {
    const uint8_t *nv12;      /* start of the Y plane, as the ISP gave it */
    uint32_t       stride;    /* Y-plane stride, not necessarily cap_w    */
    int            cap_h;     /* full captured height, to find the UV plane */
    int            x0, y0;    /* top-left of the region we want           */
    int            w, h;      /* its size                                 */
} plt_pre_view_t;

/* The whole input transform is a byte table.
 *
 * The network wants x = (p/255)*2 - 1 quantized as x_q = rint(x/scale) + zp.
 * Both steps are fixed functions of one 8-bit value, so precompute all 256 of
 * them: a per-pixel float divide plus a round costs ~45 ms a frame, a lookup
 * costs ~1.  `rint` is round-half-to-even, which is what numpy's rint does, so
 * this reproduces compile/nnmath.py:quantize() exactly -- and it has to, or the
 * device stops matching the simulator.
 */
void plt_pre_build_lut(int8_t lut[256], float scale, int zero_point);

/* A LUT for a detector that takes RAW pixel values (no (p/255)*2-1 mapping),
 * i.e. x = p quantized as x_q = rint(p/scale) + zp.  YOLOX preprocessing. */
void plt_pre_build_lut_linear(int8_t lut[256], float scale, int zero_point);

/* NV12 region -> planar int8 CHW, via the LUT. */
void plt_pre_view_int8(const plt_pre_view_t *v, const int8_t lut[256], int8_t *chw);

/* Same, but channel order B,G,R -- what a cv2-trained detector (YOLOX) expects. */
void plt_pre_view_int8_bgr(const plt_pre_view_t *v, const int8_t lut[256], int8_t *chw);

/* NV12 region -> the network's NDHWC32 input tensor, in ONE pass.
 *
 * The two-step route -- NV12 to planar CHW, then plt_ndhwc32_from_planar() --
 * costs far more than the conversion itself: the second leg walks
 * pad_h*pad_w*32 byte positions (5.5 M for a 416x416 input) to place three
 * useful values per pixel, writing a zero to the other twenty-nine and testing
 * three bounds conditions for each.
 *
 * For a 3-channel input none of that is needed.  Channels 0,1,2 of a pixel are
 * bytes 0,1,2 of its 32-byte cell -- contiguous -- so the whole transform is
 * one store per pixel, straight into the tensor.  The unused lanes are read
 * only by the stem's zero weights.
 *
 * `lut` maps a pixel byte to the FED byte `q - zero_point` directly, so no
 * per-pixel arithmetic is left after the table.  `bgr` picks the channel order.
 *
 * The view may be SHORTER than the tensor (letterboxing): rows `v->h` up to
 * `dst_h` are filled with `pad`, which is the fed byte for YOLOX's 114 grey.
 */
void plt_pre_build_lut_fed(uint8_t lut[256], float scale, int zero_point, int linear);
void plt_pre_view_ndhwc32(const plt_pre_view_t *v, const uint8_t lut[256],
                          volatile uint8_t *dst, uint32_t row_bytes, int bgr,
                          int dst_h, uint8_t pad);

/* plt_pre_view_ndhwc32 for a PACKED input (plt_engine_t.input_compact): three
 * bytes a pixel, `row_bytes` = width * 3. */
void plt_pre_view_rgb_packed(const plt_pre_view_t *v, const uint8_t lut[256],
                             volatile uint8_t *dst, uint32_t row_bytes, int bgr,
                             int dst_h, uint8_t pad);

/* NV12 region -> packed RGB888, for eyeballing what the camera actually sees. */
void plt_pre_view_rgb(const plt_pre_view_t *v, uint8_t *rgb);

#endif /* PLT_IO_PRE_H */
