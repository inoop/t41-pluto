/* YOLOX decode + NMS: the 9 head conv outputs -> detection boxes.
 *
 * The compiled graph stops at the raw reg/obj/cls conv outputs (see the YOLOX
 * import cut); this turns them into boxes.  Pure scalar C, no HAL and no device
 * dependency -- it runs the same on the host (tests/detect_host_test.c) and in
 * the camera tool.  Inputs are planar int8 [C,H,W] per tensor with the tensor's
 * (scale, zero_point); boxes come back in letterboxed-input pixels.
 */
#ifndef PLT_IO_DETECT_H
#define PLT_IO_DETECT_H

#include <stdint.h>

#define PLT_DET_SCALES 3

typedef struct { float x1, y1, x2, y2, score; int cls; } plt_det_t;

typedef struct {
    int   grid;                 /* H == W for this scale */
    int   stride;               /* 8, 16, 32 */
    const int8_t *reg, *obj, *cls;      /* planar [C,H,W] int8 */
    float reg_s, obj_s, cls_s;          /* dequant scales */
    int   reg_zp, obj_zp, cls_zp;       /* dequant zero points */
} plt_det_scale_t;

/* Decode all scales, run per-class NMS, write up to `max_out` detections sorted
 * by score.  Returns the count written. */
int plt_yolox_decode(const plt_det_scale_t scales[PLT_DET_SCALES], int num_classes,
                     float conf_thr, float nms_thr, plt_det_t *out, int max_out);

#endif /* PLT_IO_DETECT_H */
