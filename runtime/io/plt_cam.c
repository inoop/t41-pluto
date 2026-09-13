#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "plt_cam.h"

#include <imp/imp_common.h>
#include <imp/imp_encoder.h>
#include <imp/imp_osd.h>
#include <imp/imp_framesource.h>
#include <imp/imp_isp.h>
#include <imp/imp_system.h>

/* This camera: a GalaxyCore GC4023 on i2c 0x29, per thingino's board config for
 * the Noorio T110 (noorio_t110_t41nq_gc4023_aic8800d). */
#define CAM_SENSOR "gc4023"
#define CAM_I2C    0x29
#define CAM_CHN    0

/* libimp logs its own errors to a "log server" (IMP_LOG_OUT_SERVER) that does
 * not run on this camera, so every failure inside it is silent by default.
 * The executable's symbols win over a shared library's, so defining
 * imp_log_fun here interposes it and puts libimp's own diagnosis on stdout --
 * which is the only reason the version mismatch below was findable at all. */
void imp_log_fun(int le, int op, int out, const char *tag, const char *file,
                 int line, const char *func, const char *fmt, ...);
void imp_log_fun(int le, int op, int out, const char *tag, const char *file,
                 int line, const char *func, const char *fmt, ...)
{
    va_list ap;
    (void)op; (void)out; (void)file;
    if (le < 6) return;                     /* errors only, unless debugging */
    fprintf(stderr, "[imp %s %s:%d] ", tag ? tag : "?", func ? func : "?", line);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* plt_cam.h stores the sensor info opaquely; make sure it actually fits. */
typedef char plt_cam_sensor_fits[sizeof(IMPSensorInfo) <=
                                 sizeof(((plt_cam_t *)0)->sensor) ? 1 : -1];

#define FAIL(call, rc) do {                                                    \
    if (err) snprintf(err, errlen, "%s failed (%d)", (call), (int)(rc));        \
    return -1;                                                                  \
} while (0)

/* Each step is checked on its own.  Chaining them with || -- as the reference
 * implementations do -- means a failure tells you nothing about which of nine
 * calls broke, over a serial console, on a camera you cannot attach a debugger
 * to. */
int plt_cam_open(plt_cam_t *cam, int out_w, int out_h, plt_cam_fit_t fit,
                 const plt_cam_stream_cfg_t *stream, char *err, size_t errlen)
{
    IMPSensorInfo sensor;
    IMPFSChnAttr  chn;
    IMPISPSENSORAttr sattr;
    int rc;

    memset(cam, 0, sizeof *cam);
    /* With a stream, channel 0 is the one bound to the encoder and the network
     * reads channel 1.  That is the arrangement proven on this board
     * proven on this board); without a stream nothing changes. */
    cam->chn        = stream ? 1 : CAM_CHN;
    cam->stream_chn = CAM_CHN;
    cam->enc_grp    = 0;
    cam->enc_chn    = 0;
    cam->osd_grp    = 0;
    cam->out_w = out_w;
    cam->out_h = out_h;

    memset(&sensor, 0, sizeof sensor);
    snprintf(sensor.name, sizeof sensor.name, "%s", CAM_SENSOR);
    sensor.cbus_type = TX_SENSOR_CONTROL_INTERFACE_I2C;
    snprintf(sensor.i2c.type, sizeof sensor.i2c.type, "%s", CAM_SENSOR);
    sensor.i2c.addr           = CAM_I2C;
    sensor.i2c.i2c_adapter_id = 0;

    /* Order matters and is not the obvious one: the sensor comes up BEFORE
     * IMP_System_Init(). */
    if ((rc = IMP_ISP_Open())) {
        /* The usual cause on this camera, and it is not fixable from in here:
         * the shipped /usr/lib/libimp.so is SDK T41:20211215a while the driver
         * /etc/init.d/rcS loads is T41:20230620a, and IMP_ISP_Open refuses the
         * mismatch.
         *
         * The stock app does not hit it because it STATICALLY LINKS its own
         * libimp, built 20230620a, which matches the driver exactly -- /opt/
         * lecam/LeCam imports no IMP_* symbol dynamically and never loads
         * /usr/lib/libimp.so, which is an unused leftover.  (libdrivers.so, its
         * only vendor .so besides libvenus, is the NNA memory helper -- ddr_*,
         * __aie_* -- and has nothing to do with the ISP.)
         *
         * So there is no stock camera stack to fall back on.  The fix is to
         * link a libimp that DOES match: SDK 1.2.x reports 20230620a and runs
         * against the stock driver untouched (tools/fetch_sdk.py fetches 1.2.6).
         * The SDK ChangeLog states the rule plainly: "when you update the ISP
         * driver, you should also update the SDK synchronously." */
        if (err) snprintf(err, errlen,
                          "IMP_ISP_Open failed (%d) -- if the log above reports an SDK/driver "
                          "version mismatch, the libimp being loaded does not match the "
                          "tx-isp driver in flash; build against SDK 1.2.6 "
                          "(python3 tools/fetch_sdk.py) and make sure /run/pluto/lib is "
                          "populated (make deploy-lib)", rc);
        return -1;
    }
    cam->opened = 1;
    memcpy(&cam->sensor, &sensor, sizeof sensor);   /* DelSensor wants it back */
    if ((rc = IMP_ISP_AddSensor(IMPVI_MAIN, &sensor)))      FAIL("IMP_ISP_AddSensor", rc);
    if ((rc = IMP_ISP_EnableSensor(IMPVI_MAIN, &sensor)))   FAIL("IMP_ISP_EnableSensor", rc);
    if ((rc = IMP_System_Init()))                           FAIL("IMP_System_Init", rc);
    if ((rc = IMP_ISP_EnableTuning()))                      FAIL("IMP_ISP_EnableTuning", rc);

    /* Ask the sensor how big it is rather than hardcoding a resolution. */
    memset(&sattr, 0, sizeof sattr);
    if ((rc = IMP_ISP_Tuning_GetSensorAttr(IMPVI_MAIN, &sattr)))
        FAIL("IMP_ISP_Tuning_GetSensorAttr", rc);
    cam->sensor_w = (int)sattr.width;
    cam->sensor_h = (int)sattr.height;
    /* `fps` is a packed rational, numerator in the high half: 0x001E0001 = 30/1. */
    cam->fps_num  = (int)(sattr.fps >> 16);
    cam->fps_den  = (int)(sattr.fps & 0xFFFFu);
    if (cam->fps_den <= 0) cam->fps_den = 1;
    if (cam->fps_num <= 0) { cam->fps_num = 15; cam->fps_den = 1; }
    cam->fps      = cam->fps_num / cam->fps_den;
    if (cam->sensor_w <= 0 || cam->sensor_h <= 0) {
        if (err) snprintf(err, errlen, "sensor reports %dx%d", cam->sensor_w, cam->sensor_h);
        return -1;
    }

    /* The capture size: the whole sensor, scaled down uniformly until the
     * region we want fits inside it.  We deliberately do NOT ask the ISP to
     * crop.  Every crop+scale combination tried on this camera either returned
     * no frames at all (with "isp overflow !!!" and "irq-status unusual" from
     * the driver) or, for a 896->224 centre crop, panicked the board into a
     * reboot.  Scale-only works at every size tried, from 1280x720 down to
     * 224x224, so we take the whole frame and let the caller read the middle
     * of it -- which is the same pixels, for the price of an array offset.
     *
     * picWidth must be a multiple of 16 ("chnAttr->picWidth should be align to
     * 16", and CreateChn refuses otherwise), but we round UP to 32 instead.
     * The buffer pool pads the Y-plane stride to a multiple of 32 whatever we
     * ask for -- a 400-wide channel allocates 416*224*1.5 bytes -- and
     * IMPFrameInfo does not report the padding, so a 32-aligned width is the
     * one case where the stride is unambiguously the width.  (Getting this
     * wrong is not subtle: the frame comes out sheared into diagonal stripes.)
     * Rounding up also guarantees the wanted region still fits. */
    {
        const double sx = (double)cam->sensor_w / out_w;
        const double sy = (double)cam->sensor_h / out_h;
        int cw, ch;

        if (fit == PLT_CAM_FIT_STRETCH) {
            /* Each axis independently: the ISP's scaler takes an out width and
             * an out height and does not insist they agree. */
            cw = out_w; ch = out_h;
        } else {
            /* CROP takes the GENTLER downscale so the frame still contains the
             * input; LETTERBOX takes the harsher one so the whole frame fits
             * inside it.  That single choice is the whole difference. */
            const double s = (fit == PLT_CAM_FIT_CROP) ? (sx < sy ? sx : sy)
                                                       : (sx > sy ? sx : sy);
            cw = (int)((cam->sensor_w / s) + 0.5);
            ch = (int)((cam->sensor_h / s) + 0.5);
        }

        cw = (cw + 31) & ~31;
        ch = (ch +  1) & ~1;
        if (fit == PLT_CAM_FIT_CROP) {
            if (cw < out_w) cw = (out_w + 31) & ~31;
            if (ch < out_h) ch = (out_h +  1) & ~1;
        } else {
            /* rounding the width up to 32 must not push it past the input */
            while (cw > out_w && cw > 32) cw -= 32;
            if (ch > out_h) ch = out_h & ~1;
        }

        cam->cap_w  = cw;
        cam->cap_h  = ch;
        /* Chroma is subsampled 2x2, so an odd offset would swap the U and V
         * of every pixel in the crop. */
        /* Only CROP reads out of the middle; the other two fill the input from
         * (0,0), so there is nothing to offset by -- and the box mapping for
         * the overlay is the same expression in all three cases. */
        cam->crop_x = cw > out_w ? (((cw - out_w) / 2) & ~1) : 0;
        cam->crop_y = ch > out_h ? (((ch - out_h) / 2) & ~1) : 0;
    }

    /* Zero the struct and assign by name: the T41 IMPFSChnAttr has a leading
     * i2dattr and trailing fcrop/mirr_enable that the T20 one does not, so a
     * positional initialiser would silently mis-align. */
    memset(&chn, 0, sizeof chn);
    chn.picWidth  = cam->cap_w;
    chn.picHeight = cam->cap_h;
    chn.pixFmt    = PIX_FMT_NV12;
    chn.crop.enable      = 0;
    chn.scaler.enable    = 1;
    chn.scaler.outwidth  = cam->cap_w;
    chn.scaler.outheight = cam->cap_h;
    chn.outFrmRateNum = cam->fps_num;
    chn.outFrmRateDen = cam->fps_den;
    chn.nrVBs         = 3;
    chn.type          = FS_PHY_CHANNEL;

    if (stream) {
        /* The stream channel first, and bound BEFORE anything is enabled: the
         * bind is what makes the ISP hand finished frames to the encoder
         * without the CPU carrying them. */
        IMPEncoderChnAttr enc;
        IMPCell fs_cell  = { DEV_ID_FS,  cam->stream_chn, 0 };
        IMPCell enc_cell = { DEV_ID_ENC, cam->enc_grp,    0 };
        const int kbps = stream->kbps > 0 ? stream->kbps : 2000;
        const int efps = stream->fps  > 0 ? stream->fps  : cam->fps_num / cam->fps_den;
        IMPFSChnAttr sc = chn;
        sc.outFrmRateNum = efps;
        sc.outFrmRateDen = 1;

        if ((rc = IMP_FrameSource_CreateChn(cam->stream_chn, &sc)))  FAIL("IMP_FrameSource_CreateChn(stream)", rc);
        if ((rc = IMP_FrameSource_SetChnAttr(cam->stream_chn, &sc))) FAIL("IMP_FrameSource_SetChnAttr(stream)", rc);

        memset(&enc, 0, sizeof enc);
        if ((rc = IMP_Encoder_SetDefaultParam(&enc, IMP_ENC_PROFILE_AVC_MAIN,
                                              IMP_ENC_RC_MODE_CBR,
                                              (uint16_t)cam->cap_w, (uint16_t)cam->cap_h,
                                              (uint32_t)efps, 1, (uint32_t)(efps * 2), 2,
                                              -1, (uint32_t)kbps)))
            FAIL("IMP_Encoder_SetDefaultParam", rc);
        if ((rc = IMP_Encoder_CreateGroup(cam->enc_grp)))              FAIL("IMP_Encoder_CreateGroup", rc);
        if ((rc = IMP_Encoder_CreateChn(cam->enc_chn, &enc)))          FAIL("IMP_Encoder_CreateChn", rc);
        if ((rc = IMP_Encoder_RegisterChn(cam->enc_grp, cam->enc_chn))) FAIL("IMP_Encoder_RegisterChn", rc);
        /* The OSD block goes BETWEEN the framesource and the encoder, so the
         * boxes are composited in hardware on the way past.  One region per
         * drawable box, created hidden and moved as detections come and go --
         * creating and destroying regions per frame would be far more work. */
        {
            IMPCell osd_cell = { DEV_ID_OSD, cam->osd_grp, 0 };
            IMPOSDGrpRgnAttr ga;
            IMPOSDRgnAttr ra;

            if ((rc = IMP_OSD_CreateGroup(cam->osd_grp))) FAIL("IMP_OSD_CreateGroup", rc);
            cam->osd_on = 1;
            for (int i = 0; i < PLT_CAM_MAX_BOXES; i++) {
                memset(&ra, 0, sizeof ra);
                ra.type = OSD_REG_RECT;
                ra.rect.p0.x = 0; ra.rect.p0.y = 0;
                ra.rect.p1.x = 15; ra.rect.p1.y = 15;
                ra.fmt = PIX_FMT_MONOWHITE;
                ra.data.lineRectData.color     = OSD_GREEN;
                ra.data.lineRectData.linewidth = 3;
                cam->rgn[i] = IMP_OSD_CreateRgn(&ra);
                if (cam->rgn[i] < 0) FAIL("IMP_OSD_CreateRgn", cam->rgn[i]);
                memset(&ga, 0, sizeof ga);
                ga.show = 0; ga.layer = i; ga.scalex = 1.0f; ga.scaley = 1.0f;
                if ((rc = IMP_OSD_RegisterRgn(cam->rgn[i], cam->osd_grp, &ga)))
                    FAIL("IMP_OSD_RegisterRgn", rc);
                IMP_OSD_SetRgnAttr(cam->rgn[i], &ra);
                IMP_OSD_ShowRgn(cam->rgn[i], cam->osd_grp, 0);
                cam->n_rgn++;
            }
            if ((rc = IMP_System_Bind(&fs_cell, &osd_cell)))   FAIL("IMP_System_Bind(fs->osd)", rc);
            if ((rc = IMP_System_Bind(&osd_cell, &enc_cell)))  FAIL("IMP_System_Bind(osd->enc)", rc);
            if ((rc = IMP_OSD_Start(cam->osd_grp)))            FAIL("IMP_OSD_Start", rc);
        }
        cam->stream_on = 1;
    }

    if ((rc = IMP_FrameSource_CreateChn(cam->chn, &chn)))  FAIL("IMP_FrameSource_CreateChn", rc);
    if ((rc = IMP_FrameSource_SetChnAttr(cam->chn, &chn))) FAIL("IMP_FrameSource_SetChnAttr", rc);
    if (cam->stream_on) {
        if ((rc = IMP_FrameSource_EnableChn(cam->stream_chn))) FAIL("IMP_FrameSource_EnableChn(stream)", rc);
        if ((rc = IMP_Encoder_StartRecvPic(cam->enc_chn)))     FAIL("IMP_Encoder_StartRecvPic", rc);
        cam->recving = 1;
    }
    if ((rc = IMP_FrameSource_EnableChn(cam->chn)))        FAIL("IMP_FrameSource_EnableChn", rc);
    cam->streaming = 1;
    /* Mandatory: the default frame depth is 0, which means GetFrame never
     * returns anything.  May only be set once per channel. */
    if ((rc = IMP_FrameSource_SetFrameDepth(cam->chn, 2)))
        FAIL("IMP_FrameSource_SetFrameDepth", rc);

    return 0;
}

int plt_cam_frame(plt_cam_t *cam, const uint8_t **nv12, char *err, size_t errlen)
{
    IMPFrameInfo *f = NULL;
    int rc;

    if ((rc = IMP_FrameSource_GetFrame(cam->chn, &f)) || !f)
        FAIL("IMP_FrameSource_GetFrame", rc);

    /* The stride is the capture width, which plt_cam_open() made a multiple of
     * 32 precisely so that it is.  It cannot be derived from the frame: f->size
     * is width*height*3/2, the UNPADDED size, so on a channel whose stride was
     * padded the arithmetic would confidently return the wrong answer. */
    cam->stride = (uint32_t)cam->cap_w;

    cam->frame = f;
    *nv12 = (const uint8_t *)(uintptr_t)f->virAddr;
    return 0;
}

int plt_cam_boxes(plt_cam_t *cam, const plt_cam_box_t *b, int n)
{
    if (!cam->osd_on) return 0;
    if (n > cam->n_rgn) n = cam->n_rgn;

    for (int i = 0; i < cam->n_rgn; i++) {
        if (i >= n) { IMP_OSD_ShowRgn(cam->rgn[i], cam->osd_grp, 0); continue; }
        IMPOSDRgnAttr ra;
        int x0 = b[i].x0, y0 = b[i].y0, x1 = b[i].x1, y1 = b[i].y1;
        /* Clamp into the frame: the OSD rejects a region that runs off the
         * edge, and a detection on the border routinely does. */
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > cam->cap_w - 1) x1 = cam->cap_w - 1;
        if (y1 > cam->cap_h - 1) y1 = cam->cap_h - 1;
        if (x1 - x0 < 8 || y1 - y0 < 8) { IMP_OSD_ShowRgn(cam->rgn[i], cam->osd_grp, 0); continue; }

        /* Scale the line to the box.  A fixed 3px border on a 12px-wide
         * detection meets itself in the middle and draws a solid green block,
         * which reads as a bug rather than as a small object. */
        const int side = (x1 - x0) < (y1 - y0) ? (x1 - x0) : (y1 - y0);
        memset(&ra, 0, sizeof ra);
        ra.type = OSD_REG_RECT;
        ra.rect.p0.x = x0; ra.rect.p0.y = y0;
        ra.rect.p1.x = x1; ra.rect.p1.y = y1;
        ra.fmt = PIX_FMT_MONOWHITE;
        ra.data.lineRectData.color     = OSD_GREEN;
        ra.data.lineRectData.linewidth = side >= 48 ? 3u : (side >= 24 ? 2u : 1u);
        IMP_OSD_SetRgnAttr(cam->rgn[i], &ra);
        IMP_OSD_ShowRgn(cam->rgn[i], cam->osd_grp, 1);
    }
    return 0;
}

int plt_cam_stream_frame(plt_cam_t *cam, plt_cam_pack_t *packs, int max, int *n,
                         int64_t *ts_us, char *err, size_t errlen)
{
    IMPEncoderStream *st;
    int rc;

    *n = 0;
    if (!cam->stream_on || !cam->recving) return 0;
    if (cam->stream_held) plt_cam_stream_release(cam);

    /* The detector's loop must never wait on the encoder, but a ZERO timeout
     * here means "wait forever", not "do not wait" -- it hangs the frame loop
     * until the watchdog alarm fires.  One millisecond is the smallest honest
     * way to say "only if it is already finished". */
    if (IMP_Encoder_PollingStream(cam->enc_chn, 1)) return 0;

    st = calloc(1, sizeof *st);
    if (!st) FAIL("out of memory for a stream handle", 0);
    if ((rc = IMP_Encoder_GetStream(cam->enc_chn, st, 1))) {
        free(st);
        if (rc == -1) return 0;                 /* nothing ready after all */
        FAIL("IMP_Encoder_GetStream", rc);
    }
    cam->stream_held = st;

    for (unsigned i = 0; i < st->packCount && *n < max; i++) {
        if (!st->pack[i].length) continue;
        packs[*n].data = (const uint8_t *)(uintptr_t)(st->virAddr + st->pack[i].offset);
        packs[*n].len  = st->pack[i].length;
        (*n)++;
    }
    if (ts_us) *ts_us = st->packCount ? st->pack[0].timestamp : 0;
    return 1;
}

void plt_cam_stream_release(plt_cam_t *cam)
{
    if (!cam->stream_held) return;
    IMP_Encoder_ReleaseStream(cam->enc_chn, (IMPEncoderStream *)cam->stream_held);
    free(cam->stream_held);
    cam->stream_held = NULL;
}

void plt_cam_release(plt_cam_t *cam)
{
    if (cam->frame) {
        IMP_FrameSource_ReleaseFrame(cam->chn, (IMPFrameInfo *)cam->frame);
        cam->frame = NULL;
    }
}

void plt_cam_close(plt_cam_t *cam)
{
    plt_cam_release(cam);
    plt_cam_stream_release(cam);
    if (cam->recving) { IMP_Encoder_StopRecvPic(cam->enc_chn); cam->recving = 0; }
    if (cam->stream_on) {
        IMPCell fs_cell  = { DEV_ID_FS,  cam->stream_chn, 0 };
        IMPCell enc_cell = { DEV_ID_ENC, cam->enc_grp,    0 };
        IMPCell osd_cell = { DEV_ID_OSD, cam->osd_grp,    0 };
        IMP_FrameSource_DisableChn(cam->stream_chn);
        if (cam->osd_on) {
            IMP_OSD_Stop(cam->osd_grp);
            for (int i = 0; i < cam->n_rgn; i++) {
                IMP_OSD_ShowRgn(cam->rgn[i], cam->osd_grp, 0);
                IMP_OSD_UnRegisterRgn(cam->rgn[i], cam->osd_grp);
                IMP_OSD_DestroyRgn(cam->rgn[i]);
            }
            IMP_System_UnBind(&fs_cell, &osd_cell);
            IMP_System_UnBind(&osd_cell, &enc_cell);
            IMP_OSD_DestroyGroup(cam->osd_grp);
            cam->osd_on = 0;
        } else
        IMP_System_UnBind(&fs_cell, &enc_cell);
        IMP_Encoder_UnRegisterChn(cam->enc_chn);
        IMP_Encoder_DestroyChn(cam->enc_chn);
        IMP_Encoder_DestroyGroup(cam->enc_grp);
        IMP_FrameSource_DestroyChn(cam->stream_chn);
        cam->stream_on = 0;
    }
    if (cam->streaming) {
        IMP_FrameSource_DisableChn(cam->chn);
        IMP_FrameSource_DestroyChn(cam->chn);
        cam->streaming = 0;
    }
    if (cam->opened) {
        /* The exact reverse of the bring-up, per the SDK's own
         * sample_system_exit(): skipping DelSensor or DisableTuning leaves the
         * ISP half-torn-down and IMP_ISP_Close() complains that "sensor[0] has
         * been deleted", which then makes the NEXT process's Open unreliable.
         * IMP_ISP_DelSensor wants the same IMPSensorInfo it was given, so we
         * keep it in the handle rather than rebuilding it here. */
        IMP_System_Exit();
        IMP_ISP_DisableSensor(IMPVI_MAIN);
        IMP_ISP_DelSensor(IMPVI_MAIN, (IMPSensorInfo *)&cam->sensor);
        IMP_ISP_DisableTuning();
        IMP_ISP_Close();
        cam->opened = 0;
    }
}
