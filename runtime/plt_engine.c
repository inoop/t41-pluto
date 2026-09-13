#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <string.h>

#include "plt_engine.h"
#include "core/plt_layout.h"
#include "exec/plt_requant.h"
#include "exec/plt_conv_dw.h"
#include "hal/plt_isa_nna.h"

static uint32_t now_us(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (uint32_t)(t.tv_sec * 1000000u + t.tv_usec);
}

static float f32_of(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* Fill in a runtime tensor from its on-disk record.  A .pluto shape is stored
 * NCHW-ish as [N, C, H, W] with trailing dimensions zero for a vector. */
static void tensor_of(plt_tensor_t *t, const plt_tensor_rec_t *rec)
{
    memset(t, 0, sizeof *t);
    t->shape.n = rec->ndims > 0 ? rec->shape[0] : 1;
    t->shape.c = rec->ndims > 1 ? rec->shape[1] : 1;
    t->shape.h = rec->ndims > 2 ? rec->shape[2] : 1;
    t->shape.w = rec->ndims > 3 ? rec->shape[3] : 1;
    t->dtype   = rec->dtype;
    t->layout  = rec->format;
    t->q.scale = f32_of(rec->scale_bits);
    t->q.zero_point = rec->zero_point;
    t->name    = rec->name;
}

/* Point at a constant region of the blob.
 *
 * No copy.  The loaded model image is ordinary cached memory already, and the
 * CPU is what pushes every weight byte into the array, so copying the blob into
 * a mapped window would only make each of those pushes an uncached load --
 * 194 ns against 2.8 ns, measured -- and copying it again inside host
 * memory would just double the runtime's footprint.  So the host arena IS the
 * model image, and a constant's offset is its offset in the file. */
static int stage_blob(plt_ctx_t *ctx, const plt_model_t *m, uint32_t off, uint32_t size,
                      plt_mem_t *out)
{
    if (off == PLT_UNUSED || size == 0) { *out = plt_mem_none(); return 0; }
    if (!plt_model_blob(m, off, size))
        return plt_ctx_fail(ctx, "blob range %u+%u is outside the file", off, size);

    out->space = PLT_SPACE_MODEL;
    out->off   = m->header->blob_off + off;
    out->bytes = size;
    return 0;
}

int plt_engine_load(plt_ctx_t *ctx, const char *model_path, plt_engine_t *eng)
{
    memset(eng, 0, sizeof *eng);
    eng->ctx = ctx;

    if (plt_model_load(model_path, &eng->model) != 0)
        return plt_ctx_fail(ctx, "cannot load %s", model_path);

    const plt_model_t *m = &eng->model;
    eng->n = (int)m->header->num_layers;
    if (eng->n > PLT_MAX_LAYERS)
        return plt_ctx_fail(ctx, "%d layers, this build holds %d", eng->n, PLT_MAX_LAYERS);

    /* 1. Resolve every node and find the executor that claims it. */
    for (int i = 0; i < eng->n; i++) {
        const plt_layer_rec_t *r = &m->layers[i];
        plt_node_t *nd = &eng->node[i];

        nd->rec  = r;
        nd->n_in = 0;
        for (int j = 0; j < PLT_MAX_IO; j++) {
            if (r->input_ids[j] == PLT_UNUSED) continue;
            const plt_tensor_rec_t *tr = plt_model_tensor(m, r->input_ids[j]);
            if (!tr) return plt_ctx_fail(ctx, "layer %u names tensor %u, which does not exist",
                                         r->id, r->input_ids[j]);
            tensor_of(&nd->in[nd->n_in++], tr);
        }
        const plt_tensor_rec_t *outr = plt_model_tensor(m, r->output_id);
        if (!outr) return plt_ctx_fail(ctx, "layer %u has no output tensor", r->id);
        tensor_of(&nd->out, outr);

        eng->kernel[i] = plt_kernel_find(r->op, r->exec);
        if (!eng->kernel[i])
            return plt_ctx_fail(ctx, "layer %u wants op %u executor %u, which this build "
                                "does not have", r->id, r->op, r->exec);
    }

    /* 2. Stage the constants, then plan every node.  Constants first, because
     *    a plan may depend on the sizes the model declares for them.
     *
     *    The constants stay where the loader put them -- see stage_blob(). */
    ctx->model_base = eng->model.data;
    for (int i = 0; i < eng->n; i++) {
        const plt_layer_rec_t *r = &m->layers[i];
        if (stage_blob(ctx, m, r->weight_off, r->weight_size,
                       &eng->node[i].weights) != 0) return -1;
        if (stage_blob(ctx, m, r->reqtbl_off, r->reqtbl_size,
                       &eng->node[i].table) != 0) return -1;
    }
    for (int i = 0; i < eng->n; i++)
        if (eng->kernel[i]->plan(ctx, &eng->node[i], &eng->plan[i]) != 0) return -1;

    /* One shared scratch buffer, sized to the largest request.  Cached memory,
     * deliberately: see the note in core/plt_ctx.h. */
    {
        uint32_t most = 0;
        for (int i = 0; i < eng->n; i++)
            if (eng->plan[i].scratch_bytes > most) most = eng->plan[i].scratch_bytes;
        if (most && plt_ctx_reserve_scratch(ctx, most) != 0) return -1;
    }

    /* 3. Place the activations.  Three strategies:
     *
     *    default (a chain) : two rotating buffers -- layer i's output is read
     *                        only by i+1, so parity is enough.
     *    PLT_F_NOREUSE     : one buffer per tensor -- what --from/--to and a
     *                        per-layer post-mortem need; ~4x the memory.
     *    PLT_F_LIVENESS    : reuse a buffer once its tensor's last consumer has
     *                        run -- a branchy graph (a detector) is correct AND
     *                        small, unlike the rotating scheme which would
     *                        clobber a skip connection. */
    uint32_t in_bytes = 0;
    eng->input_compact = 0;
    {
        const uint32_t in_id = m->header->input_ids[0];
        const plt_tensor_rec_t *ir = plt_model_tensor(m, in_id);
        if (ir) {
            plt_tensor_t t;
            tensor_of(&t, ir);
            in_bytes = t.layout == PLT_FMT_NDHWC32
                     ? plt_ndhwc32_bytes(t.shape.c, t.shape.h, t.shape.w)
                     : (uint32_t)t.shape.c * t.shape.h * t.shape.w;

            /* An RGB image read only by a space-to-depth layer is kept PACKED.
             * As NDHWC32 cells a 640x640 image is 13 MB, of which the focus
             * layer reads 3 bytes in every 32 -- one cache line per cell, 1.6
             * million of them a frame, which made focus a 9 ms layer bound by
             * memory bandwidth alone.  Packed it is 1.2 MB.  A classifier's
             * stem feeds the array 64-byte cells, so it keeps the cells. */
            int readers = 0, focus_only = t.layout == PLT_FMT_NDHWC32 && t.shape.c == 3;
            for (int j = 0; j < eng->n; j++)
                for (int k = 0; k < PLT_MAX_IO; k++)
                    if (eng->node[j].rec->input_ids[k] == in_id) {
                        readers++;
                        if (eng->node[j].rec->op != PLT_OP_FOCUS) focus_only = 0;
                    }
            if (readers && focus_only) {
                eng->input_compact      = 1;
                eng->input_cells_bytes  = in_bytes;
                eng->input_packed_bytes = (uint32_t)t.shape.h * t.shape.w * 3u;
                in_bytes = eng->input_packed_bytes + 64u;   /* slack for lane loads */
            }
        }
    }

    /* 16-byte cells for a focus output.  YOLOX's space-to-depth yields 12
     * channels, stored as 32-lane cells that are 20/32 padding; the conv that
     * reads it only ever pushes whole cells to the array, whose padding lanes
     * meet zero weights.  If the focus writes 16-byte cells and that conv
     * assembles its pushes from them (plt_conv_k3.c, K3_LOAD16), the tensor --
     * written once and read twice over -- is half the bytes.  Only when every
     * reader is such a conv, the input is the packed image and the focus has
     * no requant. */
    for (int i = 0; i < eng->n; i++) {
        plt_node_t *f = &eng->node[i];
        if (f->rec->op != PLT_OP_FOCUS || f->out.shape.c > 16 || !eng->input_compact) continue;
        const volatile uint8_t *ftbl = plt_ctx_ptr(ctx, f->table);
        if (!ftbl) continue;
        plt_requant_t rq;
        plt_requant_of(ftbl, 1, 0, &rq);
        if (rq.kind != PLT_RQ_IDENTITY) continue;
        int readers = 0, ok = 1;
        for (int j = 0; j < eng->n; j++)
            for (int k = 0; k < PLT_MAX_IO; k++)
                if (eng->node[j].rec->input_ids[k] == f->rec->output_id) {
                    readers++;
                    if (eng->node[j].rec->exec != PLT_EX_NNA_K3 || eng->node[j].n_in != 1
                        || eng->node[j].in[0].shape.c > 16) ok = 0;
                }
        if (!ok || !readers) continue;
        f->out.layout = PLT_FMT_NDHWC16;
        eng->plan[i].out_bytes = (uint32_t)plt_pad_h(f->out.shape.h)
                               * (uint32_t)plt_pad_w(f->out.shape.w) * 16u;
        for (int j = 0; j < eng->n; j++)
            if (eng->node[j].rec->input_ids[0] == f->rec->output_id)
                eng->node[j].in[0].layout = PLT_FMT_NDHWC16;
    }

    /* 16-byte cells between NNA layers, on the same reasoning: a tensor of at
     * most 16 channels is half padding in 32-byte cells, and every byte of it is
     * drained once and fed at least once.  YOLOX-Nano's first ten layers are
     * all 16 channels.  A tensor qualifies when its producer can drain 16-byte
     * cells and every reader can feed from them; an add needs both operands and
     * its output alike, which is settled by iterating to a fixed point. */
    {
        static signed char cand[PLT_MAX_LAYERS];
        const int n = eng->n;
        for (int i = 0; i < n; i++) {
            const plt_node_t *nd = &eng->node[i];
            const uint8_t e = nd->rec->exec;
            cand[i] = nd->out.layout == PLT_FMT_NDHWC32 && nd->out.shape.c <= 16
                   && ((e == PLT_EX_NNA_PW && nd->n_in == 1 && nd->in[0].shape.c <= 32)
                       || (e == PLT_EX_NNA_K3 && nd->in[0].shape.c <= 32)
                       || (e == PLT_EX_NNA_DW && plt_conv_dw_cell16_ok(nd))
                       || nd->rec->op == PLT_OP_ADD);
        }
        for (int changed = 1; changed; ) {
            changed = 0;
            for (int i = 0; i < n; i++) {
                if (!cand[i]) continue;
                const uint32_t id = eng->node[i].rec->output_id;
                int readers = 0, ok = 1;
                for (int j = 0; j < n && ok; j++) {
                    const plt_node_t *cn = &eng->node[j];
                    for (int k = 0, seen = 0; k < PLT_MAX_IO; k++) {
                        if (cn->rec->input_ids[k] == PLT_UNUSED) continue;
                        if (cn->rec->input_ids[k] == id) {
                            readers++;
                            const uint8_t e = cn->rec->exec;
                            if (e == PLT_EX_NNA_PW || e == PLT_EX_NNA_K3) ok &= cn->n_in == 1;
                            else if (e == PLT_EX_NNA_DW) ok &= plt_conv_dw_cell16_ok(cn);
                            else if (cn->rec->op == PLT_OP_ADD) ok &= cand[j];
                            else if (cn->rec->op == PLT_OP_CONCAT) {
                                int off = 0;
                                for (int q = 0; q < seen; q++) off += cn->in[q].shape.c;
                                const volatile uint8_t *ct = plt_ctx_ptr(ctx, cn->table);
                                plt_requant_t rq;
                                if (ct) plt_requant_of(ct, cn->n_in, seen, &rq);
                                ok &= ct && rq.kind != PLT_RQ_TABLE && (off & 15) == 0
                                   && cn->in[seen].shape.c == 16;
                            } else ok = 0;
                        }
                        seen++;
                    }
                }
                /* an add writes 16-byte cells only if it also reads them */
                if (ok && eng->node[i].rec->op == PLT_OP_ADD)
                    for (int k = 0; k < PLT_MAX_IO; k++) {
                        const uint32_t in_id = eng->node[i].rec->input_ids[k];
                        if (in_id == PLT_UNUSED) continue;
                        int p = -1;
                        for (int q = 0; q < n; q++)
                            if (eng->node[q].rec->output_id == in_id) { p = q; break; }
                        if (p < 0 || !cand[p]) ok = 0;
                    }
                if (!ok || !readers) { cand[i] = 0; changed = 1; }
            }
        }
        for (int i = 0; i < n; i++) {
            if (!cand[i]) continue;
            plt_node_t *nd = &eng->node[i];
            nd->out.layout = PLT_FMT_NDHWC16;
            eng->plan[i].out_bytes = (uint32_t)plt_pad_h(nd->out.shape.h)
                                   * (uint32_t)plt_pad_w(nd->out.shape.w) * 16u;
            for (int j = 0; j < n; j++)
                for (int k = 0, seen = 0; k < PLT_MAX_IO; k++) {
                    if (eng->node[j].rec->input_ids[k] == PLT_UNUSED) continue;
                    if (eng->node[j].rec->input_ids[k] == nd->rec->output_id)
                        eng->node[j].in[seen].layout = PLT_FMT_NDHWC16;
                    seen++;
                }
        }
    }

    if (ctx->flags & PLT_F_LIVENESS) {
        static int last_use[PLT_MAX_LAYERS], slot_of[PLT_MAX_LAYERS], free_at[PLT_MAX_LAYERS];
        static int nuse[PLT_MAX_LAYERS], user[PLT_MAX_LAYERS], root[PLT_MAX_LAYERS];
        static int rdef[PLT_MAX_LAYERS], rlast[PLT_MAX_LAYERS];
        static uint32_t ssz[PLT_MAX_LAYERS], voff[PLT_MAX_LAYERS], rsize[PLT_MAX_LAYERS];
        static plt_mem_t smem[PLT_MAX_LAYERS];
        static char used[PLT_MAX_LAYERS];
        const int n = eng->n, BIG = n + 1;

        /* Who reads each layer's output, and how often. */
        for (int i = 0; i < n; i++) { last_use[i] = i; used[i] = 0; nuse[i] = 0; user[i] = -1; }
        for (int j = 0; j < n; j++)
            for (int k = 0; k < PLT_MAX_IO; k++) {
                const uint32_t id = eng->node[j].rec->input_ids[k];
                if (id == PLT_UNUSED) continue;
                for (int p = 0; p < n; p++)
                    if (eng->node[p].rec->output_id == id) {
                        if (j > last_use[p]) last_use[p] = j;
                        used[p] = 1; nuse[p]++; user[p] = j;
                        break;
                    }
            }

        /* Views.  Every output lives at an offset inside a ROOT output's
         * buffer, itself by default.  One way a buffer is shared, which saves
         * a whole pass of memory traffic over the tensor:
         *
         *   concat view  a producer read only by one concat writes its planes
         *                straight into the concat's output, already rescaled.
         *
         * Graph outputs never become views of anything, and the graph input is
         * never written over: a --repeat run and the camera loop reuse it. */
        for (int i = 0; i < n; i++) { root[i] = i; voff[i] = 0; eng->composed[i] = 0; }
        free(eng->luts);
        eng->luts = NULL;
        int nluts = 0;
        for (int c = 0; c < n; c++) {
            plt_node_t *cn = &eng->node[c];
            cn->in_view_mask = 0;
            cn->act_lut_override = NULL;
        }
        if (!(ctx->flags & PLT_F_NOVIEWS)) {
            eng->luts = malloc((size_t)n * 256u);
            for (int c = 0; eng->luts && c < n; c++) {
                plt_node_t *cn = &eng->node[c];
                if (cn->rec->op != PLT_OP_CONCAT) continue;
                const volatile uint8_t *ctbl = plt_ctx_ptr(ctx, cn->table);
                if (!ctbl) continue;
                const uint32_t oplane = plt_ndhwc32_plane_bytes(cn->out.shape.h, cn->out.shape.w);
                int off = 0, s = 0;
                for (int k = 0; k < PLT_MAX_IO; k++) {
                    const uint32_t id = cn->rec->input_ids[k];
                    if (id == PLT_UNUSED) continue;
                    const plt_tensor_t *in = &cn->in[s];
                    int p = -1;
                    for (int q = 0; q < n; q++)
                        if (eng->node[q].rec->output_id == id) { p = q; break; }
                    const int ci = in->shape.c;
                    int ok = p >= 0 && p < c && nuse[p] == 1 && user[p] == c
                          && root[p] == p && (off & 31) == 0 && (ci & 31) == 0
                          && in->shape.h == cn->out.shape.h && in->shape.w == cn->out.shape.w
                          && eng->plan[p].out_bytes == (uint32_t)(ci / 32) * oplane
                          && eng->node[p].rec->op != PLT_OP_CONCAT;
                    plt_requant_t r;
                    if (ok) {
                        plt_requant_of(ctbl, cn->n_in, s, &r);
                        const uint8_t e = eng->node[p].rec->exec;
                        const int nna = e == PLT_EX_NNA_PW || e == PLT_EX_NNA_K3
                                     || e == PLT_EX_NNA_DW || e == PLT_EX_NNA_DENSE
                                     || e == PLT_EX_NNA_STD;
                        /* Not an add: it has no in-register table pass, and a
                         * separate one over its output measured 0.4 ms net on
                         * YOLOX-S -- not worth a second path. */
                        if (r.kind != PLT_RQ_IDENTITY && !nna) ok = 0;
                        if (ok && r.kind != PLT_RQ_IDENTITY) {
                            /* compose: producer activation, then concat rescale */
                            uint8_t *lut = eng->luts + (uint32_t)nluts++ * 256u;
                            const volatile uint8_t *ptbl = plt_ctx_ptr(ctx, eng->node[p].table);
                            const volatile uint8_t *act = NULL;
                            if (eng->node[p].rec->act == PLT_ACT_SILU && ptbl) {
                                const uint32_t groups = e == PLT_EX_NNA_DW
                                    ? (uint32_t)((eng->node[p].in[0].shape.c + 31) / 32)
                                    : (uint32_t)((eng->node[p].out.shape.c + 31) / 32);
                                act = ptbl + groups * 256u;
                            }
                            for (int u = 0; u < 256; u++) {
                                const uint8_t v = act ? act[u] : (uint8_t)u;
                                lut[u] = ctbl[(uint32_t)s * 256u + v];
                            }
                            eng->node[p].act_lut_override = lut;
                            eng->composed[p] = 1;
                        }
                    }
                    if (ok) {
                        root[p] = c;
                        voff[p] = (uint32_t)(off / 32) * oplane;
                        cn->in_view_mask |= 1u << s;
                    }
                    off += ci; s++;
                }
            }
            /* Deliberately NOT done: writing a single-pass 1x1's output over its
             * own input as it goes.  It is legal (the drain trails the feed), and
             * it was measured -- 0.4 ms slower on MobileNetV1, 0.25 ms on
             * YOLOX-Nano, neutral on YOLOX-S -- presumably because every store
             * then lands in the lines the next loads are about to read. */
        }

        /* Resolve each output to its final root and absolute offset. */
        for (int i = 0; i < n; i++) {
            int r = i; uint32_t o = 0;
            while (root[r] != r) { o += voff[r]; r = root[r]; }
            root[i] = r; voff[i] = o;
        }
        if (ctx->flags & PLT_F_VERBOSE)
            for (int i = 0; i < n; i++)
                if (root[i] != i)
                    printf("VIEW L%d %s -> L%d +%u%s\n", i, eng->kernel[i]->name, root[i],
                           voff[i], eng->composed[i] ? " composed" : "");
        for (int i = 0; i < n; i++) { rsize[i] = 0; rdef[i] = BIG; rlast[i] = -1; }
        for (int i = 0; i < n; i++) {
            const int r = root[i];
            const uint32_t end = voff[i] + eng->plan[i].out_bytes;
            if (end > rsize[r]) rsize[r] = end;
            if (i < rdef[r]) rdef[r] = i;
            const int fa = used[i] ? last_use[i] : BIG;   /* a graph output never frees */
            if (fa > rlast[r]) rlast[r] = fa;
        }

        int nslots = 0;
        for (int i = 0; i < n; i++) {
            const int r = root[i];
            if (rdef[r] != i) continue;          /* the root's buffer is placed once */
            const uint32_t need = rsize[r];
            int pick = -1;
            for (int s = 0; s < nslots; s++)
                if (free_at[s] < i) { if (pick < 0 || ssz[s] >= need) pick = s;
                                      if (ssz[s] >= need) break; }
            if (pick < 0) { pick = nslots++; ssz[pick] = need; }
            else if (need > ssz[pick]) ssz[pick] = need;
            free_at[pick] = rlast[r];
            slot_of[r] = pick;
        }
        uint32_t acts = 0;
        for (int s = 0; s < nslots; s++) acts += (ssz[s] + 63u) & ~63u;
        if (plt_ctx_reserve_host(ctx, ((in_bytes + 63u) & ~63u) + acts+ 4096u) != 0) return -1;
        eng->input = plt_arena_alloc(&ctx->host, in_bytes, 64);
        if (!plt_mem_present(eng->input))
            return plt_ctx_fail(ctx, "no room for the %u-byte input", in_bytes);
        for (int s = 0; s < nslots; s++) {
            smem[s] = plt_arena_alloc(&ctx->host, ssz[s], 64);
            if (!plt_mem_present(smem[s]))
                return plt_ctx_fail(ctx, "no room for a %u-byte activation slot", ssz[s]);
        }
        for (int i = 0; i < n; i++) {
            eng->node[i].out.mem       = smem[slot_of[root[i]]];
            eng->node[i].out.mem.off  += voff[i];
            eng->node[i].out.mem.bytes = eng->plan[i].out_bytes;
        }
    } else {
        uint32_t most[2] = { 0, 0 }, sum = 0;
        for (int i = 0; i < eng->n; i++) {
            if (eng->plan[i].out_bytes > most[i & 1]) most[i & 1] = eng->plan[i].out_bytes;
            sum += (eng->plan[i].out_bytes + 63u) & ~63u;
        }
        const uint32_t acts = (ctx->flags & PLT_F_NOREUSE) ? sum : most[0] + most[1];
        if (plt_ctx_reserve_host(ctx, ((in_bytes + 63u) & ~63u) + acts + 4096u) != 0) return -1;
        eng->input = plt_arena_alloc(&ctx->host, in_bytes, 64);
        if (!plt_mem_present(eng->input))
            return plt_ctx_fail(ctx, "no room for the %u-byte input", in_bytes);

        plt_mem_t slot[2] = { plt_mem_none(), plt_mem_none() };
        if (!(ctx->flags & PLT_F_NOREUSE))
            for (int p = 0; p < 2; p++)
                if (most[p]) {
                    slot[p] = plt_arena_alloc(&ctx->host, most[p], 64);
                    if (!plt_mem_present(slot[p]))
                        return plt_ctx_fail(ctx, "no room for a %u-byte activation buffer", most[p]);
                }
        for (int i = 0; i < eng->n; i++) {
            plt_node_t *nd = &eng->node[i];
            if (plt_mem_present(slot[i & 1])) {
                nd->out.mem       = slot[i & 1];
                nd->out.mem.bytes = eng->plan[i].out_bytes;
            } else {
                nd->out.mem = plt_arena_alloc(&ctx->host, eng->plan[i].out_bytes, 64);
            }
            if (!plt_mem_present(nd->out.mem))
                return plt_ctx_fail(ctx, "layer %u: no room for a %u-byte output",
                                    nd->rec->id, eng->plan[i].out_bytes);
        }
    }

    /* Point every consumer of each tensor at the buffer it landed in, by id. */
    for (int i = 0; i < eng->n; i++) {
        plt_node_t *nd = &eng->node[i];
        for (int j = i + 1; j < eng->n; j++) {
            const plt_layer_rec_t *cons = eng->node[j].rec;
            for (int k = 0, seen = 0; k < PLT_MAX_IO; k++) {
                if (cons->input_ids[k] == PLT_UNUSED) continue;
                if (cons->input_ids[k] == nd->rec->output_id)
                    eng->node[j].in[seen].mem = nd->out.mem;
                seen++;
            }
        }
    }
    /* Whatever is still unplaced reads the graph input. */
    for (int i = 0; i < eng->n; i++)
        for (int k = 0; k < eng->node[i].n_in; k++)
            if (!plt_mem_present(eng->node[i].in[k].mem)) {
                eng->node[i].in[k].mem = eng->input;
                if (eng->input_compact) eng->node[i].in[k].layout = PLT_FMT_NHWC;
            }

    return 0;
}

void plt_engine_free(plt_engine_t *eng)
{
    free(eng->luts);
    eng->luts = NULL;
    plt_model_free(&eng->model);
}

int plt_engine_input(plt_engine_t *eng, const void *data, uint32_t bytes)
{
    if (eng->input_compact) {
        volatile uint8_t *dst = plt_ctx_ptr(eng->ctx, eng->input);
        const uint8_t *src = data;
        if (bytes == eng->input_packed_bytes) {
            plt_word_copy(dst, data, bytes);
            return 0;
        }
        if (bytes != eng->input_cells_bytes)
            return plt_ctx_fail(eng->ctx, "input is %u bytes, the model wants %u (cells) "
                                "or %u (packed)", bytes, eng->input_cells_bytes,
                                eng->input_packed_bytes);
        /* NDHWC32 cells in, packed pixels kept: the first three lanes of each. */
        const plt_tensor_rec_t *ir = plt_model_tensor(&eng->model, eng->model.header->input_ids[0]);
        plt_tensor_t t;
        tensor_of(&t, ir);
        const uint32_t row = plt_ndhwc32_row_bytes(t.shape.w);
        for (int y = 0; y < t.shape.h; y++)
            for (int x = 0; x < t.shape.w; x++)
                for (int c = 0; c < 3; c++)
                    dst[((uint32_t)y * t.shape.w + x) * 3u + c] = src[(uint32_t)y * row + (uint32_t)x * 32u + c];
        return 0;
    }
    if (bytes != eng->input.bytes)
        return plt_ctx_fail(eng->ctx, "input is %u bytes, the model wants %u",
                            bytes, eng->input.bytes);
    plt_word_copy(plt_ctx_ptr(eng->ctx, eng->input), data, bytes);
    return 0;
}

static uint64_t fnv_words(const volatile uint8_t *p, uint32_t bytes, uint64_t h)
{
    const volatile uint32_t *w = (const volatile uint32_t *)p;
    for (uint32_t i = 0; i < bytes / 4; i++) { h ^= w[i]; h *= 0x100000001b3ULL; }
    return h;
}

/* FNV-1a over the tensor's LOGICAL contents.  An NDHWC32 buffer is padded up to
 * whole tiles and the array writes computed values into that padding, so
 * hashing the whole allocation would compare bytes the host never produces. */
uint64_t plt_engine_checksum(const plt_engine_t *eng, int layer)
{
    const plt_tensor_t *t = &eng->node[layer].out;
    const volatile uint8_t *p = plt_ctx_ptr(eng->ctx, t->mem);
    uint64_t h = 0xcbf29ce484222325ULL;

    if (!p) return 0;
    if (t->layout == PLT_FMT_NDHWC16) {
        const uint32_t row = (uint32_t)plt_pad_w(t->shape.w) * 16u;
        for (int y = 0; y < t->shape.h; y++)
            h = fnv_words(p + y * row, (uint32_t)t->shape.w * 16, h);
        return h;
    }
    if (t->layout == PLT_FMT_NDHWC32) {
        const uint32_t row   = plt_ndhwc32_row_bytes(t->shape.w);
        const uint32_t plane = plt_ndhwc32_plane_bytes(t->shape.h, t->shape.w);
        for (int g = 0; g < plt_groups(t->shape.c); g++)
            for (int y = 0; y < t->shape.h; y++)
                h = fnv_words(p + g * plane + y * row, (uint32_t)t->shape.w * 32, h);
        return h;
    }
    return fnv_words(p, (uint32_t)t->shape.c * (t->dtype == PLT_DT_FP32 ? 4u : 1u), h);
}

int plt_engine_run(plt_engine_t *eng, int first, int last)
{
    if (first < 0) first = 0;
    if (last < 0 || last >= eng->n) last = eng->n - 1;

    for (int i = first; i <= last; i++) {
        const uint32_t t0 = (eng->ctx->flags & PLT_F_PROFILE) ? now_us() : 0;
        const plt_prof_t before = eng->ctx->prof;

        if (eng->kernel[i]->run(eng->ctx, &eng->node[i]) != 0) return -1;
        PLT_SYNC();          /* nothing of this layer still in flight; see plt_isa_nna.h */

        if (eng->ctx->flags & PLT_F_PROFILE) {
            eng->us[i] = now_us() - t0;
            eng->phase[i].assemble_us = eng->ctx->prof.assemble_us - before.assemble_us;
            eng->phase[i].array_us    = eng->ctx->prof.array_us    - before.array_us;
            eng->phase[i].tail_us     = eng->ctx->prof.tail_us     - before.tail_us;
        }
        if (eng->ctx->flags & PLT_F_CHECKSUM) eng->chk[i] = plt_engine_checksum(eng, i);
    }
    return 0;
}

void plt_engine_report(const plt_engine_t *eng, FILE *out)
{
    const plt_prof_t *p = &eng->ctx->prof;

    for (int i = 0; i < eng->n; i++) {
        fprintf(out, "L%-3u %-10s %-8s", eng->node[i].rec->id, eng->kernel[i]->name,
                eng->node[i].out.name);
        if (eng->ctx->flags & PLT_F_CHECKSUM) fprintf(out, " chk=%016llx%s",
                                                      (unsigned long long)eng->chk[i],
                                                      eng->composed[i] ? " composed"
                                                      : eng->node[i].out.layout == PLT_FMT_NDHWC16
                                                      ? " composed(16-byte cells)" : "");
        if (eng->ctx->flags & PLT_F_PROFILE)
            fprintf(out, " %6u us  asm=%-6u arr=%-6u tail=%u", eng->us[i],
                    eng->phase[i].assemble_us, eng->phase[i].array_us,
                    eng->phase[i].tail_us);
        fputc('\n', out);
    }
    if (eng->ctx->flags & PLT_F_PROFILE) {
        uint32_t total = 0;
        for (int i = 0; i < eng->n; i++) total += eng->us[i];
        fprintf(out, "PHASES assemble=%u array=%u tail=%u other=%d total=%u us\n",
                p->assemble_us, p->array_us, p->tail_us,
                (int)total - (int)(p->assemble_us + p->array_us + p->tail_us), total);
    }
}
