#include <math.h>

#include "exec/plt_fc_mxu.h"

void plt_fc_int8(const int8_t *in, int Cin, const int8_t *w, const int32_t *bias,
                 const float *w_scale, int Cout, float in_scale, int in_zp,
                 float out_scale, int out_zp, float *logits)
{
    for (int co = 0; co < Cout; co++) {
        const int8_t *wr = w + (long)co * Cin;
        long acc = 0, wsum = 0;
        for (int ci = 0; ci < Cin; ci++) { acc += (long)in[ci] * wr[ci]; wsum += wr[ci]; }
        long ACC = acc - (long)in_zp * wsum + (long)bias[co];
        double real = (double)in_scale * (double)w_scale[co] * (double)ACC;
        double q = rint(real / (double)out_scale) + out_zp;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        logits[co] = (float)((q - out_zp) * (double)out_scale);
    }
}

/* --- the executor ---------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>

#include "hal/plt_dev.h"
#include "hal/plt_isa_mxu.h"

/* The requant region of a fully-connected layer is int32 bias[Cout] followed by
 * fp32 w_scale[Cout] -- the weights are per-output-channel symmetric, so the
 * scale cannot be folded into a single multiplier the way a conv's can. */
static int fc_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    const int cin = node->in[0].shape.c, cout = node->out.shape.c;

    if (node->rec->op != PLT_OP_FC)
        return plt_ctx_fail(ctx, "fc_mxu: layer %u is not a fully-connected", node->rec->id);
    if (node->weights.bytes < (uint32_t)cin * cout)
        return plt_ctx_fail(ctx, "fc_mxu: layer %u weights are %u B, need %d",
                            node->rec->id, node->weights.bytes, cin * cout);
    if (node->table.bytes < (uint32_t)cout * 8u)
        return plt_ctx_fail(ctx, "fc_mxu: layer %u table is %u B, need %d",
                            node->rec->id, node->table.bytes, cout * 8);

    plan->scratch_bytes = 0;
    plan->out_bytes     = (uint32_t)cout * 4u;   /* fp32 logits */
    plan->out_pad_h     = 1;
    plan->out_pad_w     = 1;
    return 0;
}

static int fc_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    const plt_tensor_t *in = &node->in[0];
    const int cin = in->shape.c, cout = node->out.shape.c;

    const volatile uint8_t *src = plt_ctx_ptr(ctx, in->mem);
    const volatile uint8_t *wsrc = plt_ctx_ptr(ctx, node->weights);
    const volatile uint8_t *tsrc = plt_ctx_ptr(ctx, node->table);
    volatile uint8_t       *dst = plt_ctx_ptr(ctx, node->out.mem);
    if (!src || !wsrc || !tsrc || !dst)
        return plt_ctx_fail(ctx, "fc_mxu: layer %u operand unmapped", node->rec->id);

    const uint32_t t0 = plt_prof_now(ctx);

    if (cin % 64 == 0) {
        /* The dot products on MXU3, straight out of the model image and the
         * input tensor -- both ordinary cached memory.  This used to malloc and
         * copy all Cin*Cout weight bytes first (a megabyte for MobileNetV1, and
         * fresh pages to fault in on every frame), plus a second MAC per chunk
         * for sum(w), which is a property of the model: it is computed once
         * and kept.  Same order of operations as plt_fc_int8() above, which
         * stays as the path for an input that is not whole 64-byte chunks. */
        static const volatile uint8_t *key;
        static int32_t *wsum;
        static int key_cin, key_cout;
        if (key != wsrc || key_cin != cin || key_cout != cout || !wsum) {
            free(wsum);
            wsum = malloc((size_t)cout * sizeof *wsum);
            if (!wsum) return plt_ctx_fail(ctx, "fc_mxu: out of memory");
            for (int co = 0; co < cout; co++) {
                int32_t acc = 0;
                for (int ci = 0; ci < cin; ci++)
                    acc += (int8_t)wsrc[(uint32_t)co * (uint32_t)cin + (uint32_t)ci];
                wsum[co] = acc;
            }
            key = wsrc; key_cin = cin; key_cout = cout;
        }

        static int32_t sums[16] __attribute__((aligned(64)));
        const int chunks = cin / 64;
        const uint8_t *x = (const uint8_t *)(uintptr_t)src;
        const uint8_t *wr = (const uint8_t *)(uintptr_t)wsrc;
        const double in_scale = in->q.scale, out_scale = node->out.q.scale;
        const long in_zp = in->q.zero_point;
        const double out_zp = node->out.q.zero_point;

        for (int co = 0; co < cout; co++, wr += cin) {
            PLT_MXU_SUMZ(0);
            for (int c = 0; c < chunks; c++) {
                PLT_VLD64(4, x  + c * 64);
                PLT_VLD64(5, wr + c * 64);
                PLT_MXU_MACSSB16(0, 0, 4, 5);   /* word 0: sum(in * w) */
            }
            PLT_MXU_MFSUM(6, 0);
            PLT_VST64(6, sums);

            int32_t bias;
            float wscale;
            for (int k = 0; k < 4; k++) {
                ((uint8_t *)&bias)[k]   = tsrc[(uint32_t)co * 4u + (uint32_t)k];
                ((uint8_t *)&wscale)[k] = tsrc[(uint32_t)(cout + co) * 4u + (uint32_t)k];
            }
            /* The same tail as the scalar reference, in the same order. */
            const long ACC = (long)sums[0] - in_zp * (long)wsum[co] + bias;
            const double real = in_scale * (double)wscale * (double)ACC;
            double q = rint(real / out_scale) + out_zp;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            float lg = (float)((q - out_zp) * out_scale);
            uint32_t bits;
            memcpy(&bits, &lg, sizeof bits);
            for (int k = 0; k < 4; k++) dst[co * 4 + k] = (uint8_t)(bits >> (8 * k));
        }
        plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
        return 0;
    }

    int8_t  *x      = malloc((size_t)cin);
    int8_t  *w      = malloc((size_t)cin * cout);
    int32_t *bias   = calloc((size_t)cout, sizeof(int32_t));
    float   *wscale = calloc((size_t)cout, sizeof(float));
    float   *logits = malloc((size_t)cout * sizeof(float));
    if (!x || !w || !bias || !wscale || !logits) {
        free(x); free(w); free(bias); free(wscale); free(logits);
        return plt_ctx_fail(ctx, "fc_mxu: out of memory for %d -> %d", cin, cout);
    }

    plt_copy_fast(x, src,  (size_t)cin);
    plt_copy_fast(w, wsrc, (size_t)cin * cout);
    plt_copy_fast(bias,   tsrc,                          (size_t)cout * 4);
    plt_copy_fast(wscale, tsrc + (uint32_t)cout * 4,     (size_t)cout * 4);

    plt_fc_int8(x, cin, w, bias, wscale, cout, in->q.scale, in->q.zero_point,
                node->out.q.scale, node->out.q.zero_point, logits);

    for (int i = 0; i < cout; i++) {
        uint32_t bits;
        memcpy(&bits, &logits[i], sizeof bits);
        for (int k = 0; k < 4; k++) dst[i * 4 + k] = (uint8_t)(bits >> (8 * k));
    }

    free(x); free(w); free(bias); free(wscale); free(logits);
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_fc_mxu = {
    PLT_OP_FC, PLT_EX_MXU_FC, "fc_mxu", fc_plan, fc_run
};
