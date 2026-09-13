/* plt — the pluto command line.
 *
 *   plt dump <model.pluto>          print the container's header, tensors and layers
 *   plt run  <model.pluto> [opts]   run the graph, from a file or from the camera
 *
 * One source, two builds.  Under -DPLT_HOST it compiles with plain gcc and links
 * only core/plt_model.c, which is what `make roundtrip` needs: `plt dump` must
 * print plt_model_dump()'s bytes and NOTHING else on stdout, because the test
 * diffs them against compile/demo_model.py.  Usage and load errors go to stderr
 * for exactly that reason.
 *
 * The device build adds `run`, whose stdout is also a contract:
 * tests/model_end_to_end.sh scrapes `RESULT run rc=`, the per-layer report lines
 * and `RESULT run layers=N ok=1`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/plt_model.h"

#ifndef PLT_HOST
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include "hal/plt_isa_mxu.h"
#include "hal/plt_mxu3.h"
#include "hal/plt_dev.h"

#include "core/plt_layout.h"
#include "io/plt_cam.h"
#include "io/plt_coco_labels.h"
#include "io/plt_rtsp.h"
#include "io/plt_labels.h"
#include "io/plt_jpeg.h"
#include "io/plt_pre.h"
#include "io/plt_detect.h"
#include "plt_engine.h"
#endif

static int usage(void)
{
    fprintf(stderr,
        "usage: plt <command> [args]\n"
        "\n"
        "  dump <model.pluto>            print the header, tensor and layer tables\n"
#ifndef PLT_HOST
        "  run  <model.pluto> [options]  run the graph\n"
        "\n"
        "    input\n"
        "       --input FILE      raw bytes for the graph's input tensor\n"
        "       --camera          read live frames from the camera instead\n"
        "       --frames N        stop after N frames (0 = until Ctrl-C)\n"
        "       --warmup N        frames to discard while auto-exposure settles\n"
        "       --fit MODE        how the field of view meets the input tensor:\n"
        "                         letterbox (default) whole frame, aspect kept, padded\n"
        "                         stretch             whole frame, aspect NOT kept\n"
        "                         crop                centre only, most magnification\n"
        "\n"
        "    what to do with the output\n"
        "       --yolox           decode the 9 head outputs as YOLOX detections\n"
        "       --patchcore       score a PatchCore model (models/patchcore): anomaly\n"
        "                         score = largest patch nearest-neighbour distance\n"
        "       --batch-hwc3 FILE run every H x W x 3 int8 frame in FILE in turn\n"
        "       --map FILE        with --patchcore --batch-hwc3: write each patch map\n"
        "       --topk N          classifier: how many classes to print (default 5)\n"
        "       --out FILE        write the last layer's output tensor\n"
        "\n"
        "    live view (implies --camera)\n"
        "       --rtsp [PORT]     serve H.264 over RTSP, default port 8554.\n"
        "                         With --yolox the detections are drawn on it.\n"
        "                         Play it with: ffplay -rtsp_transport tcp \\\n"
        "                                         rtsp://<camera>:8554/\n"
        "       --bitrate KBPS    encoder target bitrate (default 2000)\n"
        "\n"
        "    diagnostics\n"
        "       --checksums       print an FNV-1a of every layer's output\n"
        "       --profile         print per-layer microseconds\n"
        "       --repeat N        run a file input N times warm; report the median,\n"
        "                         then N more unprofiled (WALL) and check every warm\n"
        "                         run leaves the same activations\n"
        "       --no-views        no concat views (producers writing into a concat)\n"
        "       --verbose         print the memory plan's views\n"
        "       --vsum            per-tensor checksums (forces --no-reuse; 31 MB,\n"
        "                         which this board does not have -- expect an OOM)\n"
        "       --from N --to N   run only layers [N, N]\n"
        "       --no-reuse        one activation buffer per layer (needed by --from/--to)\n"
        "       --dump-rgb FILE   write one frame as raw RGB888 and exit\n"
        "       --dump-jpeg FILE  write one frame as a JPEG and exit\n"
        "\n"
        "  membench [small]              memory-access cost on this CPU, no array\n"
        "                                (the full bench needs ~38 MB: never run two)\n"
        "  e.g.  plt run yolox_nano.pluto --camera --yolox --rtsp --frames 0\n"
#endif
    );
    return 2;
}

/* --- dump ----------------------------------------------------------------- */

static int cmd_dump(int argc, char **argv)
{
    plt_model_t m;
    int rc;

    if (argc < 1) return usage();
    rc = plt_model_load(argv[0], &m);
    if (rc != 0) {
        fprintf(stderr, "%s: load failed (rc=%d)\n", argv[0], rc);
        return 1;
    }
    plt_model_dump(&m, stdout);
    plt_model_free(&m);
    return 0;
}

/* --- run ------------------------------------------------------------------ */

#ifndef PLT_HOST

static int read_file(const char *path, void **data, uint32_t *bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(p); return -1; }
    fclose(f);
    *data = p; *bytes = (uint32_t)n;
    return 0;
}

static int run_yolox(plt_ctx_t *ctx, plt_engine_t *eng, int num_classes,
                     plt_det_t *dets, int max_dets);

/* Ctrl-C must not skip the camera teardown, or the ISP is left streaming and
 * the next run cannot open it. */
static volatile sig_atomic_t s_stop;
static void on_signal(int sig) { (void)sig; s_stop = 1; }

/* The graph emits dequantized logits -- there is no softmax in the model -- so
 * turn them into probabilities here.  Subtract the max first: exp() of a raw
 * logit overflows long before the ratios do. */
static void softmax(const float *logits, int n, float *prob)
{
    float mx = logits[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) { prob[i] = expf(logits[i] - mx); sum += prob[i]; }
    for (int i = 0; i < n; i++) prob[i] /= sum;
}

/* `frame` < 0 suppresses the timing line, for the one-shot --input path where
 * there is no frame number and the millisecond count is meaningless. */
static void print_topk(const float *prob, int n, int k, int frame, double ms)
{
    if (frame >= 0)
        printf("frame %d  %.1f ms  %.1f fps\n", frame, ms, ms > 0 ? 1000.0 / ms : 0.0);
    for (int r = 0; r < k; r++) {
        int best = -1;
        for (int i = 0; i < n; i++)
            if (prob[i] >= 0.0f && (best < 0 || prob[i] > prob[best])) best = i;
        if (best < 0) break;
        printf("  %5.1f%%  %s\n", 100.0 * prob[best],
               best < PLT_LABEL_COUNT ? PLT_LABELS[best] : "?");
        ((float *)prob)[best] = -1.0f;      /* consumed */
    }
    fflush(stdout);
}

/* The live loop.
 *
 * The ISP hands us the whole field of view already scaled down (it will not
 * crop and scale at once -- see plt_cam.h), so the CPU reads the centred
 * network-sized rectangle out of it, converting colour and quantizing as it
 * goes: one table lookup per channel, no resampling.  The frame is released
 * before inference, not after: holding it for the ~40 ms of the graph would
 * starve the pipeline. */
static int run_camera(plt_cam_t *cam, plt_ctx_t *ctx, plt_engine_t *eng, int topk,
                      int frames, int warmup, const char *dumprgb,
                      const char *dumpjpg, int yolox, int rtsp_port)
{
    const plt_tensor_t *in = &eng->node[0].in[0];
    const int w = in->shape.w, h = in->shape.h, c = in->shape.c;
    const plt_tensor_t *out = &eng->node[eng->n - 1].out;
    const int nclass = out->shape.c;
    char err[128] = {0};
    plt_pre_view_t view;
    int8_t lut[256];
    int8_t *chw = NULL;
    float *prob = NULL;
    int rc = 1;

    if (topk < 1) topk = 1;
    if (topk > nclass) topk = nclass;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);       /* a client that hangs up must not kill us */

    plt_rtsp_t *rtsp = NULL;
    if (rtsp_port) {
        char rerr[128] = {0};
        rtsp = plt_rtsp_start(rtsp_port, cam->cap_w, cam->cap_h, rerr, sizeof rerr);
        if (!rtsp) { printf("RESULT run rc=rtsp %s\n", rerr); goto done; }
        printf("rtsp: listening on port %d -- rtsp://<camera>:%d/  (TCP transport)\n",
               rtsp_port, rtsp_port);
    }

    /* The ISP gives us the whole field of view; the network gets the middle
     * of it.  Everything but nv12 is fixed for the life of the channel. */
    memset(&view, 0, sizeof view);
    view.x0 = cam->crop_x;
    view.y0 = cam->crop_y;
    /* Whichever is smaller: a cropped capture is bigger than the input and we
     * read the middle of it; a letterboxed one is smaller and we pad. */
    view.w  = cam->cap_w < w ? cam->cap_w : w;
    view.h  = cam->cap_h < h ? cam->cap_h : h;

    if (yolox) plt_pre_build_lut_linear(lut, in->q.scale, in->q.zero_point);
    else       plt_pre_build_lut(lut, in->q.scale, in->q.zero_point);
    /* A 3-channel input goes straight into the tensor -- see plt_pre.h. */
    uint8_t fed[256];
    const int one_pass = (c == 3);
    plt_pre_build_lut_fed(fed, in->q.scale, in->q.zero_point, yolox);
    /* YOLOX letterboxes with 114 grey; a classifier's LUT is not linear in the
     * pixel value, so ask the table rather than assuming. */
    const uint8_t padfed = fed[114];
    chw  = malloc((size_t)c * w * h);
    prob = malloc((size_t)nclass * sizeof *prob);
    if (!chw || (!yolox && !prob)) { printf("RESULT run rc=mem\n"); goto done; }

    /* Auto-exposure needs a few seconds of frames before it is worth looking at
     * what the camera thinks it sees. */
    for (int i = 0; i < warmup && !s_stop; i++) {
        const uint8_t *nv12;
        alarm(30);
        if (plt_cam_frame(cam, &nv12, err, sizeof err)) break;
        plt_cam_release(cam);
    }

    for (int n = 0; !s_stop && (frames <= 0 || n < frames); n++) {
        const uint8_t *nv12;
        uint32_t t0, tp, t1;

        alarm(30);
        if (plt_cam_frame(cam, &nv12, err, sizeof err)) {
            printf("RESULT run rc=frame %s\n", err);
            goto shutdown;
        }
        if (n == 0)
            printf("first frame: capture %dx%d stride %u -> %dx%d of a %dx%d input at +%d+%d%s\n",
                   cam->cap_w, cam->cap_h, cam->stride, view.w, view.h, w, h,
                   cam->crop_x, cam->crop_y,
                   view.h < h ? " (rest padded)" : "");

        view.nv12   = nv12;
        view.stride = cam->stride;
        view.cap_h  = cam->cap_h;

        t0 = plt_now_us();
        if (dumpjpg) {
            /* The same view the network reads, so this answers "what did it
             * actually see" rather than "what is the camera pointed at". */
            const int ok = plt_jpeg_write(dumpjpg, &view, 85) == 0;
            printf(ok ? "wrote %s (%dx%d JPEG)\n" : "could not write %s (%dx%d)\n",
                   dumpjpg, w, h);
            plt_cam_release(cam);
            rc = ok ? 0 : 1;
            goto shutdown;
        }
        if (dumprgb) {
            uint8_t *rgb = malloc((size_t)w * h * 3);
            FILE *f = rgb ? fopen(dumprgb, "wb") : NULL;
            if (f) {
                plt_pre_view_rgb(&view, rgb);
                fwrite(rgb, 1, (size_t)w * h * 3, f);
                fclose(f);
                printf("wrote %s (%dx%d RGB888)\n", dumprgb, w, h);
            }
            free(rgb);
            plt_cam_release(cam);
            rc = 0;
            goto shutdown;
        }
        if (one_pass && eng->input_compact) {
            plt_pre_view_rgb_packed(&view, fed, plt_ctx_ptr(ctx, eng->input),
                                    (uint32_t)w * 3u, yolox, h, padfed);
            plt_cam_release(cam);
        } else if (one_pass) {
            plt_pre_view_ndhwc32(&view, fed, plt_ctx_ptr(ctx, eng->input),
                                 plt_ndhwc32_row_bytes(w), yolox, h, padfed);
            plt_cam_release(cam);
        } else {
            if (yolox) plt_pre_view_int8_bgr(&view, lut, chw);
            else       plt_pre_view_int8(&view, lut, chw);
            plt_cam_release(cam);
            plt_ndhwc32_from_planar(plt_ctx_ptr(ctx, eng->input), chw, c, h, w,
                                    in->q.zero_point);
        }
        tp = plt_now_us();
        if (plt_engine_run(eng, -1, -1)) {
            printf("RESULT run rc=run %s\n", ctx->err);
            goto shutdown;
        }
        t1 = plt_now_us();

        if (rtsp) {
            /* Accept clients, then hand over every frame the encoder has
             * finished.  The encoder runs at its own rate, so drain until it
             * says none -- otherwise the view lags further behind each frame. */
            plt_rtsp_poll(rtsp);
            for (;;) {
                plt_cam_pack_t pk[8];
                int np = 0; int64_t pts = 0;
                char serr[128] = {0};
                const int got = plt_cam_stream_frame(cam, pk, 8, &np, &pts, serr, sizeof serr);
                if (got <= 0) break;
                for (int k = 0; k < np; k++)
                    plt_rtsp_send_annexb(rtsp, pk[k].data, pk[k].len,
                                         (uint32_t)((pts * 9) / 100));   /* us -> 90 kHz */
                plt_cam_stream_release(cam);
            }
        }

        if (yolox) {
            printf("frame %d  %.1f ms  %.1f fps  (pre %.1f + net %.1f)\n", n,
                   (t1 - t0) / 1000.0, (t1 > t0) ? 1000000.0 / (t1 - t0) : 0.0,
                   (tp - t0) / 1000.0, (t1 - tp) / 1000.0);
            plt_det_t dets[PLT_CAM_MAX_BOXES];
            const int nd = run_yolox(ctx, eng, 80, dets, PLT_CAM_MAX_BOXES);
            /* The network reads a crop of the streamed view, so a detection's
             * coordinates have to be shifted by where that crop sits before the
             * OSD can draw it over the right pixels. */
            plt_cam_box_t bx[PLT_CAM_MAX_BOXES];
            for (int i = 0; i < nd; i++) {
                bx[i].x0 = (int)dets[i].x1 + cam->crop_x;
                bx[i].y0 = (int)dets[i].y1 + cam->crop_y;
                bx[i].x1 = (int)dets[i].x2 + cam->crop_x;
                bx[i].y1 = (int)dets[i].y2 + cam->crop_y;
            }
            plt_cam_boxes(cam, bx, nd);
            fflush(stdout);
        } else {
            softmax((const float *)plt_ctx_ptr(ctx, out->mem), nclass, prob);
            print_topk(prob, nclass, topk, n, (t1 - t0) / 1000.0);
        }
    }
    rc = 0;

shutdown:
    alarm(0);
done:
    plt_rtsp_stop(rtsp);
    free(chw);
    free(prob);
    if (rc == 0) printf("RESULT run camera ok=1\n");
    return rc;
}

/* Unpack an NDHWC32 head tensor to planar int8 q = byte ^ 0x80 (the byte is
 * q + 128).  Caller frees. */
static int8_t *unpack_q(plt_ctx_t *ctx, const plt_tensor_t *t)
{
    const int C = t->shape.c, H = t->shape.h, W = t->shape.w;
    const uint32_t row = plt_ndhwc32_row_bytes(W), plane = plt_ndhwc32_plane_bytes(H, W);
    const volatile uint8_t *p = plt_ctx_ptr(ctx, t->mem);
    int8_t *out = malloc((size_t)C * H * W);
    if (!p || !out) { free(out); return NULL; }
    for (int c = 0; c < C; c++) {
        const int g = c / 32, k = c % 32;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                out[(c * H + y) * W + x] = (int8_t)(p[g * plane + y * row + x * 32 + k] ^ 0x80u);
    }
    return out;
}

/* PatchCore scoring (models/patchcore).
 *
 * The model's last layer is the memory bank as a 1x1 convolution: for every
 * patch feature q and bank vector b_k it emits ||b_k||^2 - 2 q.b_k.  Its INPUT is
 * the patch features themselves.  So the squared nearest-neighbour distance of a
 * patch is  ||q||^2 + min_k(output_k) , computed here from those two tensors,
 * and the image's anomaly score is the largest patch distance.
 *
 * The minimum is taken on the stored bytes: b = q + 128 is monotone in the real
 * value, so no dequantization happens until the one byte that wins.  Writes the
 * patch map (row-major, float32) to `map` when it is non-NULL. */
static float run_patchcore(plt_ctx_t *ctx, const plt_engine_t *eng, float *map,
                           int *arg_y, int *arg_x)
{
    const plt_node_t *nd = &eng->node[eng->n - 1];
    const plt_tensor_t *dt = &nd->out, *ft = &nd->in[0];
    const int K = dt->shape.c, D = ft->shape.c, H = dt->shape.h, W = dt->shape.w;
    const uint8_t *dp = (const uint8_t *)(uintptr_t)plt_ctx_ptr(ctx, dt->mem);
    const uint8_t *fp = (const uint8_t *)(uintptr_t)plt_ctx_ptr(ctx, ft->mem);
    const uint32_t row = plt_ndhwc32_row_bytes(W), plane = plt_ndhwc32_plane_bytes(H, W);
    const int npx = H * W;
    float best = -1.0f;
    *arg_y = *arg_x = 0;
    if (!dp || !fp || npx > 4096) return -1.0f;

    /* Plane by plane, pixel by pixel, lane by lane: every read walks forward
     * through one group plane, which is what this CPU's memory wants. */
    uint8_t mn[4096];
    int64_t q2[4096];
    memset(q2, 0, (size_t)npx * sizeof q2[0]);

    /* The minimum over the bank, as a running MAXIMUM of inverted bytes kept in
     * one plane-sized buffer: max(~b) = ~min(b), and MXU3 has an unsigned byte
     * max but this runtime has not needed a min.  64 bytes a step across every
     * group plane, then each pixel's 32 lanes reduced once.  A bank whose
     * channel count is not whole groups pads its last group's lanes with the
     * array's zero-weight output, which would win the minimum -- so only whole
     * groups take the vector path. */
    static uint8_t acc[64 * 64 * 32 + 64] __attribute__((aligned(64)));
    static const uint8_t ones[64] __attribute__((aligned(64))) = { [0 ... 63] = 0xFF };
    if (plane > sizeof acc - 64 || K % 32) return -1.0f;
    memset(acc, 0, plane);
    PLT_VLD64(1, ones);
    for (int g = 0; g < K / 32; g++) {
        register const uint8_t *src __asm__("t0") = dp + (uint32_t)g * plane;
        register uint8_t       *dst __asm__("t1") = acc;
        register uint32_t       n   __asm__("t2") = (plane + 63) / 64;
        __asm__ __volatile__(".set push\n\t.set noreorder\n\t.set noat\n\t"
            "1:\n\t"
            PLT_M3_LAO(0, 0, 8, 0) PLT_M3_LAO(0, 1, 8, 1)
            PLT_M3_LAO(2, 0, 9, 0) PLT_M3_LAO(2, 1, 9, 1)
            PLT_M3_OP(PLT_M3_XORV, 0, 1, 0)
            PLT_M3_OP(PLT_M3_MAXUB, 0, 2, 2)
            PLT_M3_SAO(2, 0, 9, 0) PLT_M3_SAO(2, 1, 9, 1)
            "addiu %[s], %[s], 64\n\t" "addiu %[d], %[d], 64\n\t"
            "addiu %[n], %[n], -1\n\t" "bnez %[n], 1b\n\t" "nop\n\t"
            ".set pop\n\t"
            : [s] "+r"(src), [d] "+r"(dst), [n] "+r"(n) :: "memory");
    }
    __asm__ __volatile__("sync" ::: "memory");
    for (int y = 0, i = 0; y < H; y++) {
        const uint8_t *r = acc + (uint32_t)y * row;
        for (int x = 0; x < W; x++, i++, r += 32) {
            uint8_t m = 0;
            for (int l = 0; l < 32; l++) m = r[l] > m ? r[l] : m;
            mn[i] = (uint8_t)~m;
        }
    }
    const int fzp = ft->q.zero_point;
    for (int g = 0; g * 32 < D; g++) {
        const int lanes = D - g * 32 < 32 ? D - g * 32 : 32;
        const uint8_t *pl = fp + (uint32_t)g * plane;
        for (int y = 0, i = 0; y < H; y++) {
            const uint8_t *r = pl + (uint32_t)y * row;
            for (int x = 0; x < W; x++, i++, r += 32) {
                int32_t acc = 0;
                for (int l = 0; l < lanes; l++) {
                    const int v = (int8_t)(r[l] ^ 0x80u) - fzp;
                    acc += v * v;
                }
                q2[i] += acc;
            }
        }
    }
    const double fs2 = (double)ft->q.scale * ft->q.scale;
    for (int i = 0; i < npx; i++) {
        const double dmin = ((int8_t)(mn[i] ^ 0x80u) - dt->q.zero_point) * (double)dt->q.scale;
        double sc = dmin + (double)q2[i] * fs2;
        if (sc < 0) sc = 0;
        if (map) map[i] = (float)sc;
        if ((float)sc > best) { best = (float)sc; *arg_y = i / W; *arg_x = i % W; }
    }
    return best;
}

/* YOLOX decode: the head outputs are the tensors no layer consumes.  Classify
 * each by channel count (reg=4/obj=1/cls=rest) and grid, decode, print. */
static int run_yolox(plt_ctx_t *ctx, plt_engine_t *eng, int num_classes,
                     plt_det_t *dets, int max_dets)
{
    const plt_tensor_rec_t *ir = plt_model_tensor(&eng->model, eng->model.header->input_ids[0]);
    const int in_h = ir ? ir->shape[2] : 416;

    plt_det_scale_t sc[PLT_DET_SCALES];
    memset(sc, 0, sizeof sc);
    for (int i = 0; i < eng->n; i++) {
        const uint32_t oid = eng->node[i].rec->output_id;
        int consumed = 0;
        for (int j = 0; j < eng->n && !consumed; j++)
            for (int k = 0; k < PLT_MAX_IO; k++)
                if (eng->node[j].rec->input_ids[k] == oid) { consumed = 1; break; }
        if (consumed) continue;

        const plt_tensor_t *t = &eng->node[i].out;
        const int grid = t->shape.h, stride = grid ? in_h / grid : 0;
        int s = stride == 8 ? 0 : stride == 16 ? 1 : stride == 32 ? 2 : -1;
        if (s < 0) continue;
        sc[s].grid = grid; sc[s].stride = stride;
        int8_t *q = unpack_q(ctx, t);
        if (t->shape.c == 4)      { sc[s].reg = q; sc[s].reg_s = t->q.scale; sc[s].reg_zp = t->q.zero_point; }
        else if (t->shape.c == 1) { sc[s].obj = q; sc[s].obj_s = t->q.scale; sc[s].obj_zp = t->q.zero_point; }
        else                      { sc[s].cls = q; sc[s].cls_s = t->q.scale; sc[s].cls_zp = t->q.zero_point; }
    }

    plt_det_t out[256];
    const int n = plt_yolox_decode(sc, num_classes, 0.30f, 0.45f, out, 256);
    for (int i = 0; i < n; i++)
        printf("DET %-14s %.4f  %.1f %.1f %.1f %.1f\n",
               plt_coco_label(out[i].cls), out[i].score,
               out[i].x1, out[i].y1, out[i].x2, out[i].y2);
    printf("RESULT yolox dets=%d\n", n);
    for (int s = 0; s < PLT_DET_SCALES; s++) {
        free((void *)sc[s].reg); free((void *)sc[s].obj); free((void *)sc[s].cls);
    }
    for (int i = 0; dets && i < n && i < max_dets; i++) dets[i] = out[i];
    return n < max_dets ? n : max_dets;
}

/* Debug: FNV-1a over a layer's LOGICAL (valid-channel) data, so a host compare
 * needs no knowledge of the array's padding bytes. */
static void run_vsum(plt_ctx_t *ctx, plt_engine_t *eng)
{
    for (int i = 0; i < eng->n; i++) {
        const plt_tensor_t *t = &eng->node[i].out;
        uint64_t h = 0xcbf29ce484222325ULL;
        if (t->layout == PLT_FMT_NDHWC32) {
            int8_t *q = unpack_q(ctx, t);
            const long n = (long)t->shape.c * t->shape.h * t->shape.w;
            for (long j = 0; j < n; j++) { h ^= (uint8_t)q[j]; h *= 0x100000001b3ULL; }
            free(q);
        } else {
            const volatile uint8_t *p = plt_ctx_ptr(ctx, t->mem);
            const long n = (long)t->shape.c * (t->dtype == PLT_DT_FP32 ? 4 : 1);
            for (long j = 0; j < n; j++) { h ^= p[j]; h *= 0x100000001b3ULL; }
        }
        printf("VSUM %d %016llx %s c=%d h=%d w=%d\n", i, (unsigned long long)h,
               eng->kernel[i]->name, t->shape.c, t->shape.h, t->shape.w);
    }
}


/* plt membench: how much a 64-byte vector load costs as a function of where it
 * reads, so feed layouts are chosen from the cache this CPU actually has rather
 * than from a guess.  Loads only -- nothing reaches the array. */
static uint64_t mb_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

/* plt membench small: one sequential-stream test in 6 MB, small enough that two
 * can run at once on the 64 MB board (the full bench takes ~38 MB). */
static int membench_small(void)
{
    enum { SB = 6 << 20 };
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, SB)) return 1;
    memset(buf, 1, SB);
    for (int rep = 0; rep < 5; rep++) {
        uint64_t t0 = mb_now_ns();
        uint32_t loads = 0;
        for (int pass = 0; pass < 4; pass++)
            for (uint32_t o = 0; o + 256 <= (uint32_t)SB; o += 256, loads += 4) {
                const uint8_t *p = buf + o;
                PLT_VLD64(0, p); PLT_VLD64(1, p + 64); PLT_VLD64(2, p + 128); PLT_VLD64(3, p + 192);
                PLT_VST64(0, (uint8_t *)(uintptr_t)p);
            }
        printf("MBS seq load+store %5.1f ns per 64B load\n", (double)(mb_now_ns() - t0) / loads);
    }
    free(buf);
    return 0;
}

static int cmd_membench(int argc, char **argv)
{
    if (argc > 0 && !strcmp(argv[0], "small")) return membench_small();
    enum { BUF = 16 << 20, NOPS = 400000 };
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, BUF)) return 1;
    for (uint32_t i = 0; i < BUF; i++) buf[i] = (uint8_t)(i * 2654435761u >> 24);
    uint32_t *off = malloc(NOPS * sizeof *off);
    if (!off) return 1;
    uint32_t x = 12345;
    for (uint32_t ws = 4096; ws <= (uint32_t)BUF; ws <<= 1) {
        for (int i = 0; i < NOPS; i++) {
            x = x * 1664525u + 1013904223u;
            off[i] = (uint32_t)(((uint64_t)x * (ws / 64)) >> 32) * 64u;
        }
        uint64_t best = ~0ull;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = mb_now_ns();
            for (int i = 0; i < NOPS; i++) PLT_VLD64(0, buf + off[i]);
            const uint64_t d = mb_now_ns() - t0;
            if (d < best) best = d;
        }
        printf("MB random ws=%6u KB  %6.1f ns/load\n", ws >> 10, (double)best / NOPS);
    }
    /* S interleaved forward streams of 64-byte loads, 4 loads per stream per
     * round, spaced `gap` apart: the shape of a window feed. */
    static const int streams[] = { 1, 2, 4, 5, 8, 10, 16, 20, 40, 80 };
    static const uint32_t gaps[] = { 1024, 20000, 200000 };
    for (unsigned gi = 0; gi < sizeof gaps / sizeof gaps[0]; gi++)
    for (unsigned si = 0; si < sizeof streams / sizeof streams[0]; si++) {
        const int S = streams[si];
        const uint32_t gap = gaps[gi] & ~63u;
        if ((uint64_t)gap * S + 64 * (uint64_t)NOPS / S > BUF) continue;
        const int rounds = NOPS / (4 * S);
        uint64_t best = ~0ull;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = mb_now_ns();
            for (int r = 0; r < rounds; r++)
                for (int s = 0; s < S; s++) {
                    const uint8_t *p = buf + (uint32_t)s * gap + (uint32_t)r * 256u;
                    PLT_VLD64(0, p); PLT_VLD64(1, p + 64); PLT_VLD64(2, p + 128); PLT_VLD64(3, p + 192);
                }
            const uint64_t d = mb_now_ns() - t0;
            if (d < best) best = d;
        }
        printf("MB streams=%2d gap=%6u  %6.1f ns/load\n", S, gap, (double)best / (rounds * 4 * S));
    }
    /* Software prefetch: 4 streams, 4 loads per stream per round, with and
     * without a `pref` `ahead` bytes in front of each load group. */
    {
        static const int aheads[] = { 0, 256, 512, 1024, 2048, 4096 };
        for (unsigned ai = 0; ai < sizeof aheads / sizeof aheads[0]; ai++)
        for (int S = 2; S <= 8; S *= 2) {
            const int ahead = aheads[ai];
            const uint32_t gap = 1500000u;
            const int rounds = ((int)BUF - (int)gap * S - 65536) / 256;
            if (rounds <= 1000) continue;
            uint64_t best = ~0ull;
            /* fresh region per repetition: walk a different offset so the
             * data is not already cached from the previous repetition */
            for (int rep = 0; rep < 2; rep++) {
                /* evict by touching a large unrelated region */
                volatile uint32_t sink = 0;
                for (uint32_t i = 0; i < BUF; i += 4096) sink += buf[(i * 7919u) % BUF];
                const uint64_t t0 = mb_now_ns();
                const int R = rounds > 20000 ? 20000 : rounds;
                for (int r = 0; r < R; r++)
                    for (int st = 0; st < S; st++) {
                        const uint8_t *p = buf + (uint32_t)st * gap + (uint32_t)r * 256u + (uint32_t)rep * 16384u;
                        if (ahead) __builtin_prefetch(p + ahead);
                        PLT_VLD64(0, p); PLT_VLD64(1, p + 64); PLT_VLD64(2, p + 128); PLT_VLD64(3, p + 192);
                    }
                const uint64_t d = (mb_now_ns() - t0) / (uint64_t)(R * 4 * S);
                if (d < best) best = d;
            }
            printf("MB pref ahead=%4d streams=%d  %5.1f ns/load\n", ahead, S, (double)best);
        }
    }
    /* Stores into cold memory, with and without a PREF hint on the target:
     * hint 1 store, 5 store_streamed, 30 PrepareForStore (zero the cache line
     * without reading it, where a core implements it). */
    {
        enum { SB = 12 << 20 };
        uint8_t *cold = NULL;
        if (posix_memalign((void **)&cold, 4096, SB) == 0) {
            memset(cold, 1, SB);
            static const int hints[] = { -1, 1, 5, 30 };
            static const uint8_t src64[64] __attribute__((aligned(64))) = { 7 };
            PLT_VLD64(0, src64);
            for (unsigned h = 0; h < 4; h++)
            for (int ahead = 0; ahead <= 256; ahead += 128) {
                if (hints[h] < 0 && ahead) continue;
                uint64_t best = ~0ull;
                for (int rep = 0; rep < 2; rep++) {
                    /* evict: walk the 16 MB bench buffer */
                    volatile uint32_t sink = 0;
                    for (uint32_t i = 0; i < BUF; i += 32) sink += buf[i];
                    const uint32_t n = 60000;
                    uint8_t *d = cold + (uint32_t)rep * 5000000u;
                    uint64_t t0 = mb_now_ns();
                    for (uint32_t i = 0; i < n; i++, d += 64) {
                        register uint8_t *pb __asm__("t0") = d + ahead;
                        if (hints[h] == 1)  __asm__ __volatile__("pref 1, 0($8)" :: "r"(pb) : "memory");
                        if (hints[h] == 5)  __asm__ __volatile__("pref 5, 0($8)" :: "r"(pb) : "memory");
                        if (hints[h] == 30)
                            __asm__ __volatile__("pref 30, 0($8)\n\tpref 30, 32($8)" :: "r"(pb) : "memory");
                        PLT_VST64(0, d);
                    }
                    uint64_t dd = (mb_now_ns() - t0) / n;
                    if (dd < best) best = dd;
                }
                printf("MB store hint=%2d ahead=%3d  %4llu ns/store\n", hints[h], ahead, (unsigned long long)best);
            }
            /* is hint 30 really zeroing without read?  check a line's contents */
            memset(cold, 0xAB, 256);
            { register uint8_t *pc __asm__("t0") = cold;
              __asm__ __volatile__("pref 30, 0($8)" :: "r"(pc) : "memory"); }
            printf("MB pref30 effect: byte0=%02x byte31=%02x byte32=%02x\n", cold[0], cold[31], cold[32]);
            free(cold);
        }
    }
    /* The 1x1 feed pattern with no array: D planes of an 80x80 NDHWC32
     * tensor, per tile 4 loads per group (row 0 and row 1, two 64-byte pixels
     * each), optionally followed by the drain's 4 stores per output group. */
    {
        const int D = 4, H = 80, W = 80;
        const uint32_t row = (uint32_t)W * 32u, plane = (uint32_t)H * row;
        uint8_t *in = NULL, *out = NULL, *other = NULL;
        if (!posix_memalign((void **)&in, 4096, (size_t)D * plane) &&
            !posix_memalign((void **)&out, 4096, (size_t)D * plane) &&
            !posix_memalign((void **)&other, 4096, 8u << 20)) {
            memset(in, 3, (size_t)D * plane); memset(out, 4, (size_t)D * plane);
            for (int mode = 0; mode < 9; mode++) {
                uint64_t best = ~0ull;
                for (int rep = 0; rep < 3; rep++) {
                    memset(other, rep, 8u << 20);           /* evict */
                    if (mode == 5) memset(out, 5, (size_t)D * plane);   /* warm destination */
                    uint64_t t0 = mb_now_ns();
                    for (int rp = 0; rp < H / 2; rp++)
                        for (int cb = 0; cb < W / 4; cb++) {
                            for (int g = 0; g < D; g++) {
                                const uint8_t *q = in + (uint32_t)g * plane + (uint32_t)(2 * rp) * row + (uint32_t)cb * 128u;
                                if (mode == 3) {   /* pair layout: 4 streams */
                                    const uint32_t pst = (uint32_t)D * 64u, rowP = (uint32_t)(W / 2) * pst;
                                    q = in + (uint32_t)(2 * rp) * rowP + (uint32_t)(2 * cb) * pst + (uint32_t)g * 64u;
                                    PLT_VLD64(0, q); PLT_VLD64(1, q + pst); PLT_VLD64(2, q + rowP); PLT_VLD64(3, q + rowP + pst);
                                } else {
                                    PLT_VLD64(0, q); PLT_VLD64(1, q + 64); PLT_VLD64(2, q + row); PLT_VLD64(3, q + row + 64);
                                }
                            }
                            if (mode >= 1)
                                for (int g = 0; g < D; g++) {
                                    uint8_t *q = out + (uint32_t)g * plane + (uint32_t)(2 * rp) * row + (uint32_t)cb * 128u;
                                    if (mode == 2) q = other + 64;   /* hot sink */
                                    if (mode == 4) q = in + (uint32_t)g * plane + (uint32_t)(2 * rp) * row + (uint32_t)cb * 128u;
                                    if (mode == 6 || mode == 7) {   /* walk a 16 KB ring */
                                        const uint32_t k = ((uint32_t)cb * 4u + (uint32_t)g) * 256u % 16128u;
                                        q = other + 256 + k; PLT_VST64(0, q); PLT_VST64(1, q + 64); PLT_VST64(2, q + 128); PLT_VST64(3, q + 192);
                                        continue;
                                    }
                                    if (mode == 8) {   /* stores one row pair behind, sequential per group */
                                        q = out + (uint32_t)g * plane + (uint32_t)(2 * rp) * row + (uint32_t)cb * 128u;
                                    }
                                    PLT_VST64(0, q); PLT_VST64(1, q + 64); PLT_VST64(2, q + row); PLT_VST64(3, q + row + 64);
                                }
                            if (mode == 7 && cb == W / 4 - 1) {
                                /* flush the ring's row pair to the output, in bulk */
                                for (int g = 0; g < D; g++) {
                                    plt_copy_fast(out + (uint32_t)g * plane + (uint32_t)(2 * rp) * row, other + 256, (size_t)row);
                                    plt_copy_fast(out + (uint32_t)g * plane + (uint32_t)(2 * rp + 1) * row, other + 256, (size_t)row);
                                }
                            }
                        }
                    uint64_t d = mb_now_ns() - t0;
                    if (d < best) best = d;
                }
                const double loads = (double)(H / 2) * (W / 4) * D * 4;
                printf("MB pwpattern mode=%d (%s)  %6.1f ns per load\n", mode,
                       mode == 0 ? "loads only" : mode == 1 ? "loads+stores" : mode == 2 ? "loads+sink" : mode == 3 ? "pair layout loads" : mode == 4 ? "in place" : mode == 5 ? "warm dest" : mode == 6 ? "ring 16K stores" : mode == 7 ? "ring + bulk flush" : "stores trail loads",
                       (double)best / loads);
            }
        }
        free(in); free(out); free(other);
    }
    /* The `sync` barrier every NNA instruction carries, against a nop. */
    {
        uint64_t bs = ~0ull, bn = ~0ull;
        for (int rep = 0; rep < 3; rep++) {
            uint64_t t0 = mb_now_ns();
            for (int i = 0; i < NOPS; i++) __asm__ __volatile__("sync" ::: "memory");
            uint64_t d = mb_now_ns() - t0; if (d < bs) bs = d;
            t0 = mb_now_ns();
            for (int i = 0; i < NOPS; i++) __asm__ __volatile__("nop" ::: "memory");
            d = mb_now_ns() - t0; if (d < bn) bn = d;
        }
        printf("MB sync %6.2f ns  nop-loop %6.2f ns\n", (double)bs / NOPS, (double)bn / NOPS);
    }
    /* Store cost: 64-byte stores walking forward (the drain's shape). */
    {
        uint64_t best = ~0ull;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = mb_now_ns();
            for (int i = 0; i < NOPS; i++) PLT_VST64(0, buf + (uint32_t)i * 64u % (BUF - 64));
            const uint64_t d = mb_now_ns() - t0;
            if (d < best) best = d;
        }
        printf("MB seq store  %6.1f ns/store\n", (double)best / NOPS);
    }
    free(off); free(buf);
    return 0;
}

static int cmd_run(int argc, char **argv)
{
    const char *model = NULL, *input = NULL, *outpath = NULL, *dumprgb = NULL;
    const char *dumpjpg = NULL;
    const char *batch = NULL, *mappath = NULL;
    int patchcore = 0;
    int first = -1, last = -1, flags = 0;
    int camera = 0, topk = 5, frames = 0, warmup = 75, yolox = 0, vsum = 0, repeat = 1;
    int rtsp_port = 0, rtsp_kbps = 2000;
    /* Letterbox by default: it is what compile/pack_input.py does, and so what
     * the model was quantized against.  See plt_cam.h. */
    plt_cam_fit_t fit = PLT_CAM_FIT_LETTERBOX;
    /* Only an explicit --topk prints labels on the --input path: that path's
     * stdout is the contract tests/model_end_to_end.sh scrapes, and it has
     * never carried anything but checksums. */
    int topk_given = 0;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "--input")     && i + 1 < argc) input   = argv[++i];
        else if (!strcmp(argv[i], "--out")       && i + 1 < argc) outpath = argv[++i];
        else if (!strcmp(argv[i], "--from")      && i + 1 < argc) first   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--to")        && i + 1 < argc) last    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--checksums")) flags |= PLT_F_CHECKSUM;
        else if (!strcmp(argv[i], "--profile"))   flags |= PLT_F_PROFILE;
        else if (!strcmp(argv[i], "--no-reuse"))  flags |= PLT_F_NOREUSE;
        else if (!strcmp(argv[i], "--no-views"))  flags |= PLT_F_NOVIEWS;
        else if (!strcmp(argv[i], "--verbose"))   flags |= PLT_F_VERBOSE;
        else if (!strcmp(argv[i], "--camera"))     camera  = 1;
        else if (!strcmp(argv[i], "--topk")    && i + 1 < argc) { topk = atoi(argv[++i]); topk_given = 1; }
        else if (!strcmp(argv[i], "--frames")  && i + 1 < argc) frames  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat")  && i + 1 < argc) repeat  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i + 1 < argc) warmup  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-rgb") && i + 1 < argc) dumprgb = argv[++i];
        else if (!strcmp(argv[i], "--dump-jpeg") && i + 1 < argc) dumpjpg = argv[++i];
        else if (!strcmp(argv[i], "--yolox"))      yolox = 1;
        else if (!strcmp(argv[i], "--patchcore"))  patchcore = 1;
        else if (!strcmp(argv[i], "--batch-hwc3") && i + 1 < argc) batch = argv[++i];
        else if (!strcmp(argv[i], "--map") && i + 1 < argc) mappath = argv[++i];
        else if (!strcmp(argv[i], "--rtsp")) {
            rtsp_port = 8554;                      /* optional port argument */
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                rtsp_port = atoi(argv[++i]);
            camera = 1;
        }
        else if (!strcmp(argv[i], "--bitrate") && i + 1 < argc) rtsp_kbps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fit") && i + 1 < argc) {
            const char *m = argv[++i];
            if      (!strcmp(m, "crop"))      fit = PLT_CAM_FIT_CROP;
            else if (!strcmp(m, "letterbox")) fit = PLT_CAM_FIT_LETTERBOX;
            else if (!strcmp(m, "stretch"))   fit = PLT_CAM_FIT_STRETCH;
            else { printf("plt run: --fit wants crop, letterbox or stretch\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--vsum"))       { vsum = 1; flags |= PLT_F_NOREUSE; }
        else if (argv[i][0] != '-' && !model)     model = argv[i];
        else { printf("plt run: unknown argument %s\n", argv[i]); return 1; }
    }
    if (!model) return usage();

    /* A detector's graph branches, and its 9 head outputs must all survive to
     * the decode -- the two rotating activation buffers assume a chain, so a
     * detector run needs its own buffer per tensor. */
    /* Liveness placement for every graph: a chain places exactly as the
     * rotating pair did, and only liveness knows about views. */
    if (!(flags & PLT_F_NOREUSE)) flags |= PLT_F_LIVENESS;

    if (dumprgb || dumpjpg) camera = 1;

    /* A wedged array must not hold the serial console hostage.  The camera loop
     * re-arms this per frame instead, since it is meant to run indefinitely. */
    if (!camera) alarm(600);

    /* The camera comes up BEFORE the array.
     *
     * pluto's context opens /dev/soc-nna and mmaps /dev/mem, and the vendor
     * stack brings up its own view of the same SoC.  Taking the ISP first is
     * both the vendor's order and the one that leaves libimp a clean machine to
     * initialise. */
    plt_cam_t cam;
    memset(&cam, 0, sizeof cam);
    if (camera) {
        char cerr[128] = {0};
        int cw = 0, ch = 0;
        plt_model_t probe;
        if (plt_model_load(model, &probe) == 0) {
            const plt_tensor_rec_t *t = plt_model_tensor(&probe, probe.header->input_ids[0]);
            if (t && t->ndims >= 4) { ch = t->shape[2]; cw = t->shape[3]; }
            plt_model_free(&probe);
        }
        if (cw <= 0 || ch <= 0) { printf("RESULT run rc=camera cannot size the input\n"); return 1; }
        plt_cam_stream_cfg_t scfg = { rtsp_kbps, 0 };
        if (plt_cam_open(&cam, cw, ch, fit, rtsp_port ? &scfg : NULL, cerr, sizeof cerr)) {
            printf("RESULT run rc=camera %s\n", cerr);
            printf("(the ISP is exclusive -- stop any process already streaming)\n");
            return 1;
        }
        printf("camera: sensor %dx%d @%d fps, ISP scaling to %dx%d, "
               "network reads %dx%d at +%d+%d\n",
               cam.sensor_w, cam.sensor_h, cam.fps, cam.cap_w, cam.cap_h,
               cw, ch, cam.crop_x, cam.crop_y);
        fflush(stdout);
    }

    plt_ctx_t ctx;
    if (plt_ctx_open(&ctx)) {
        printf("RESULT run rc=open %s\n", ctx.err); plt_cam_close(&cam); return 1;
    }
    ctx.flags = flags;   /* set before the load: it decides the memory layout */

    static plt_engine_t eng;
    if (plt_engine_load(&ctx, model, &eng)) {
        printf("RESULT run rc=load %s\n", ctx.err); return 1;
    }
    printf("loaded %s: %d layers, %u KB of nmem\n", model, eng.n, ctx.nmem.used / 1024u);

    if (input) {
        void *data; uint32_t bytes;
        if (read_file(input, &data, &bytes)) { printf("RESULT run rc=input\n"); return 1; }
        if (plt_engine_input(&eng, data, bytes)) {
            printf("RESULT run rc=input %s\n", ctx.err); return 1;
        }
        free(data);
    }

    if (camera) {
        const int rc = run_camera(&cam, &ctx, &eng, topk, frames, warmup, dumprgb,
                                  dumpjpg, yolox, rtsp_port);
        plt_cam_close(&cam);
        plt_engine_free(&eng);
        plt_ctx_close(&ctx);
        return rc;
    }

    /* --repeat N: N warm runs in one process, so a measurement is not a cold
     * page cache and a process start.  Each run's phases are reported, then the
     * median by total -- single cold runs wander by several ms, which is larger
     * than most of the optimizations this exists to measure. */
    /* --batch-hwc3 FILE: a run of H x W x 3 frames of quantized int8 pixels,
     * each expanded into the NDHWC32 input and run in turn.  3 bytes a pixel
     * instead of 32 is what makes a whole test set cheap to ship to the board. */
    if (batch) {
        const plt_tensor_rec_t *ir = plt_model_tensor(&eng.model, eng.model.header->input_ids[0]);
        const int H = ir->shape[2], W = ir->shape[3], zp = ir->zero_point;
        const uint32_t fb = (uint32_t)H * W * 3u;
        void *data; uint32_t bytes;
        if (ir->shape[1] != 3 || read_file(batch, &data, &bytes) || bytes % fb) {
            printf("RESULT run rc=batch (want a 3-channel model and whole %ux%ux3 frames)\n", H, W);
            return 1;
        }
        const uint32_t cells = plt_ndhwc32_bytes(3, H, W), row = plt_ndhwc32_row_bytes(W);
        uint8_t *cell = calloc(cells, 1);
        FILE *mf = mappath ? fopen(mappath, "wb") : NULL;
        const plt_tensor_t *dt = &eng.node[eng.n - 1].out;
        float *map = malloc((size_t)dt->shape.h * dt->shape.w * sizeof *map);
        if (!cell || !map) { printf("RESULT run rc=mem\n"); return 1; }
        for (uint32_t f = 0; f < bytes / fb; f++) {
            const uint8_t *px = (const uint8_t *)data + f * fb;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 3; c++)       /* the two's-complement byte of q - zp */
                        cell[(uint32_t)y * row + (uint32_t)x * 32u + c] =
                            (uint8_t)((int8_t)px[((uint32_t)y * W + x) * 3u + c] - zp);
            if (plt_engine_input(&eng, cell, cells)) { printf("RESULT run rc=input %s\n", ctx.err); return 1; }
            const uint32_t t0 = plt_now_us();
            if (plt_engine_run(&eng, -1, -1)) { printf("RESULT run rc=run %s\n", ctx.err); return 1; }
            const uint32_t t1 = plt_now_us();
            if (patchcore) {
                int ay, ax;
                const float sc = run_patchcore(&ctx, &eng, map, &ay, &ax);
                const uint32_t t2 = plt_now_us();
                printf("PATCHCORE frame=%u score=%.4f at=%d,%d net_us=%u tail_us=%u\n",
                       f, sc, ay, ax, t1 - t0, t2 - t1);
                if (mf) fwrite(map, sizeof *map, (size_t)dt->shape.h * dt->shape.w, mf);
            } else {
                printf("FRAME %u net_us=%u\n", f, t1 - t0);
            }
        }
        if (mf) fclose(mf);
        free(map); free(cell); free(data);
        printf("RESULT run frames=%u ok=1\n", bytes / fb);
        plt_engine_free(&eng);
        plt_ctx_close(&ctx);
        return 0;
    }

    if (repeat > 1) {
        uint32_t tot[64], rasm[64], arr[64], tl[64];
        if (repeat > 64) repeat = 64;
        ctx.flags |= PLT_F_PROFILE;
        for (int r = 0; r < repeat; r++) {
            memset(&ctx.prof, 0, sizeof ctx.prof);
            if (plt_engine_run(&eng, first, last)) { printf("RESULT run rc=run %s\n", ctx.err); return 1; }
            uint32_t t = 0;
            for (int i = 0; i < eng.n; i++) t += eng.us[i];
            tot[r] = t; rasm[r] = ctx.prof.assemble_us; arr[r] = ctx.prof.array_us; tl[r] = ctx.prof.tail_us;
            /* Determinism: every warm run must leave the activation memory
             * exactly as the one before did.  Hashed after the timing, so it costs nothing
             * that is measured. */
            uint64_t state = 1469598103934665603ull;
            for (int i = 0; i < eng.n; i++) state = (state ^ plt_engine_checksum(&eng, i)) * 1099511628211ull;
            /* Against run 1, not run 0: the first run leaves never-written
             * padding lanes (a focus or concat output's unused channels) as
             * whatever the fresh allocation held. */
            static uint64_t state1;
            if (r == 1) state1 = state;
            printf("RUN %d total=%u asm=%u arr=%u tail=%u us%s\n", r, t, rasm[r], arr[r], tl[r],
                   r < 2 || state == state1 ? "" : "  STATE-DIFFERS");
        }
        /* Then the same number of runs with profiling OFF, timed only around
         * the whole run: per-layer and per-phase timing is several clock reads
         * a layer, and this is what a caller that is not profiling sees. */
        {
            const int saved = ctx.flags;
            uint32_t wall[64];
            ctx.flags &= ~PLT_F_PROFILE;
            for (int r = 0; r < repeat; r++) {
                const uint64_t t0 = mb_now_ns();
                if (plt_engine_run(&eng, first, last)) { printf("RESULT run rc=run %s\n", ctx.err); return 1; }
                wall[r] = (uint32_t)((mb_now_ns() - t0) / 1000u);
            }
            ctx.flags = saved;
            for (int a = 1; a < repeat; a++)
                for (int b2 = a; b2 > 0 && wall[b2] < wall[b2 - 1]; b2--) {
                    uint32_t x = wall[b2]; wall[b2] = wall[b2 - 1]; wall[b2 - 1] = x;
                }
            printf("WALL n=%d min=%u median=%u max=%u us (profiling off)\n",
                   repeat, wall[0], wall[repeat / 2], wall[repeat - 1]);
        }
        int idx[64];
        for (int r = 0; r < repeat; r++) idx[r] = r;
        for (int a = 1; a < repeat; a++)
            for (int b = a; b > 0 && tot[idx[b]] < tot[idx[b-1]]; b--) { int x = idx[b]; idx[b] = idx[b-1]; idx[b-1] = x; }
        const int m = idx[repeat / 2];
        printf("REPEAT n=%d min=%u median=%u max=%u us  median-run asm=%u arr=%u tail=%u\n",
               repeat, tot[idx[0]], tot[m], tot[idx[repeat-1]], rasm[m], arr[m], tl[m]);
    } else
    if (plt_engine_run(&eng, first, last)) { printf("RESULT run rc=run %s\n", ctx.err); return 1; }
    if (flags & ~PLT_F_NOREUSE) plt_engine_report(&eng, stdout);
    if (yolox) run_yolox(&ctx, &eng, 80, NULL, 0);
    if (patchcore) {
        int ay, ax;
        printf("PATCHCORE score=%.4f at=%d,%d\n", run_patchcore(&ctx, &eng, NULL, &ay, &ax), ay, ax);
    }
    if (vsum) run_vsum(&ctx, &eng);

    if (topk_given) {
        const plt_tensor_t *out = &eng.node[eng.n - 1].out;
        const int nclass = out->shape.c;
        float *prob = malloc((size_t)nclass * sizeof *prob);
        if (prob) {
            softmax((const float *)plt_ctx_ptr(&ctx, out->mem), nclass, prob);
            print_topk(prob, nclass, topk < nclass ? topk : nclass, -1, 0.0);
            free(prob);
        }
    }

    if (outpath) {
        const int lastrun = (last < 0 || last >= eng.n) ? eng.n - 1 : last;
        const plt_mem_t m = eng.node[lastrun].out.mem;
        const volatile uint8_t *p = plt_ctx_ptr(&ctx, m);
        FILE *f = fopen(outpath, "wb");
        if (!f) { printf("RESULT run rc=out\n"); return 1; }
        for (uint32_t i = 0; i < m.bytes; i++) fputc(p[i], f);
        fclose(f);
    }

    printf("RESULT run layers=%d ok=1\n", eng.n);
    plt_engine_free(&eng);
    plt_ctx_close(&ctx);
    return 0;
}

#endif /* !PLT_HOST */

int main(int argc, char **argv)
{
    /* A camera run is watched live, usually through a pipe -- block buffering
     * would hold every frame line back and, on a crash, lose them entirely. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) return usage();
    if (!strcmp(argv[1], "dump")) return cmd_dump(argc - 2, argv + 2);
#ifndef PLT_HOST
    if (!strcmp(argv[1], "run"))  return cmd_run(argc - 2, argv + 2);
    if (!strcmp(argv[1], "membench")) return cmd_membench(argc - 2, argv + 2);
#else
    if (!strcmp(argv[1], "run")) {
        fprintf(stderr, "plt: this is the host build; `run` needs the device build\n");
        return 2;
    }
#endif
    fprintf(stderr, "plt: unknown command '%s'\n", argv[1]);
    return usage();
}
