#include "exec/plt_add.h"
#include <stdlib.h>
#include "core/plt_layout.h"
#include "hal/plt_dev.h"
#include "hal/plt_isa_mxu.h"
#include "hal/plt_mxu3.h"

/* Elementwise residual add of two int8 activations on independent scales.
 *
 * The layer is six int32: Ca, Cb, K and the three zero points.  The add is
 *
 *     acc = Ca*(a - 128 - za) + Cb*(b - 128 - zb)
 *     out = clamp(((acc + 2^(K-1)) >> K) + zo, -128, 127) + 128
 *
 * and `K` is chosen by the compiler (compile/ops/add.py:_fixed) as the smallest
 * shift that reproduces the float reference for all 65536 (a,b) pairs, so this
 * is byte-identical to the table it replaces -- but expressible in integers,
 * which the float tables were not, and so vectorizable.
 *
 * On MXU3 it runs 64 bytes a pass: a byte vector widens to four int32 registers
 * of 16 lanes each, and the narrow is that widen's exact inverse, so the lane
 * scramble in between does not matter for an elementwise operation (verified on
 * hardware).
 *
 * Two things keep it in int32 without a single 64-bit operation:
 *
 * The zero points fold into the accumulator's SEED.  `Ca*(a-OA) + Cb*(b-OB)` is
 * `Ca*a + Cb*b - (Ca*OA + Cb*OB)`, and that last term is a constant -- so it is
 * added once as part of the rounding term instead of being subtracted from
 * every lane.  Two pipeline stages disappear with it.
 *
 * The multiply SPLITS when a coefficient is too big.  Layer 41's Ca is 1.3e7,
 * so Ca*255 alone overflows int32 and the layer used to fall back to a scalar
 * int64 loop that cost more than the other six adds put together.  Writing
 * `Ca = A1*2^s + A0` with `A0 = Ca & (2^s - 1)` splits the accumulator into
 *
 *     S = 2^s*T + U,   T = A1*a + B1*b,   U = A0*a + B0*b
 *
 * and then, for V = U + 2^(K-1) kept non-negative,
 *
 *     (S + 2^(K-1)) >> K  ==  (T + (V >> s)) >> (K - s)
 *
 * exactly -- because dropping V's low s bits shifts the result by less than one
 * unit, and a floor cannot cross an integer boundary on that.  At s = 0 the
 * split degenerates to the plain form (A0 = B0 = 0, V = 2^(K-1)), so there is
 * one code path rather than two.
 */

static uint32_t buf_bytes(const plt_tensor_t *t)
{ return plt_ndhwc32_bytes(t->shape.c, t->shape.h, t->shape.w); }

static int add_plan(plt_ctx_t *ctx, plt_node_t *node, plt_plan_t *plan)
{
    (void)ctx;
    const plt_tensor_t *o = &node->out;
    plan->out_pad_h = plt_pad_h(o->shape.h);
    plan->out_pad_w = plt_pad_w(o->shape.w);
    plan->out_bytes = buf_bytes(o);
    plan->scratch_bytes = 0;
    return 0;
}

/* The split above, as integers the kernel can load.  Returns the shift `s`, or
 * -1 if no split keeps every intermediate inside int32. */
typedef struct {
    int32_t a1, b1, a0, b0;     /* the split coefficients   */
    int32_t t0, v0;             /* the two accumulator seeds */
    int32_t s, ks;              /* the two shifts           */
} add_fixed_t;

static int add_split(int32_t Ca, int32_t Cb, int32_t K, int32_t OA, int32_t OB,
                     add_fixed_t *f)
{
    for (int s = 0; s < K; s++) {
        const long long m  = ((long long)1 << s) - 1;
        const long long a1 = Ca >> s, b1 = Cb >> s;
        const long long a0 = Ca & m,  b0 = Cb & m;
        const long long t0 = -(a1 * OA + b1 * OB);
        const long long v0 = ((long long)1 << (K - 1)) - (a0 * OA + b0 * OB);
        /* the operands run over 0..255, so these bound every lane */
        const long long tmax = (t0 < 0 ? -t0 : t0) + 255 * ((a1 < 0 ? -a1 : a1) + (b1 < 0 ? -b1 : b1));
        const long long vmin = v0;                       /* a0,b0 >= 0 */
        const long long vmax = v0 + 255 * (a0 + b0);
        const long long lim  = 2147483647LL;
        if (vmin < 0 || vmax > lim) continue;            /* V >> s must be a floor */
        if (tmax + (vmax >> s) > lim) continue;
        f->a1 = (int32_t)a1; f->b1 = (int32_t)b1;
        f->a0 = (int32_t)a0; f->b0 = (int32_t)b0;
        f->t0 = (int32_t)t0; f->v0 = (int32_t)v0;
        f->s = s; f->ks = K - s;
        return 0;
    }
    return -1;
}

/* Register map.  Constants are splatted across all 16 int32 lanes, so each is a
 * whole 64-byte register; `luo` walks the array that holds them.
 *
 *   0,1      the raw a/b chunk, and the packed result
 *   2..12    the constants          13,14   widen scratch
 *   16..19   a widened              20..23  b widened, then a product
 *   24..27   T                      28..31  V, then narrow scratch
 */
#define A_Z    2
#define A_CA1  3
#define A_CB1  4
#define A_CA0  5
#define A_CB0  6
#define A_T0   7
#define A_V0   8
#define A_ZO   9
#define A_HI  10
#define A_S   11
#define A_KS  12
#define A_NCONST 11

/* One pipeline stage across all four lane registers before the next begins.
 * Each is an independent chain, so front-loading the four keeps the unit busy
 * through the result latency instead of stalling on it. */
#define A_ST(op, e, f, g) \
    PLT_M3_OP(op, (e) + 0, f, (g) + 0) PLT_M3_OP(op, (e) + 1, f, (g) + 1) \
    PLT_M3_OP(op, (e) + 2, f, (g) + 2) PLT_M3_OP(op, (e) + 3, f, (g) + 3)
/* load 64 bytes of each input and widen; then narrow, store and step */
#define A_HEAD                                              \
    PLT_M3_LUO(0, 0, 8) PLT_M3_LUO(0, 1, 8)                 \
    PLT_M3_LUO(1, 0, 9) PLT_M3_LUO(1, 1, 9)                 \
    PLT_M3_WIDEN4(0, A_Z, 13, 14, 16, 17, 18, 19)           \
    PLT_M3_WIDEN4(1, A_Z, 13, 14, 20, 21, 22, 23)
#define A_TAIL                                              \
    PLT_M3_NARROW4(24, 25, 26, 27, 28, 29, 0)               \
    PLT_M3_SAO(0, 0, 10, 0) PLT_M3_SAO(0, 1, 10, 1)         \
    "addiu %[pd], %[pd], 64\n\t"                           \
    "addiu %[cnt], %[cnt], -1\n\t"                         \
    "bnez %[cnt], 1b\n\t"                                  \
    "nop\n\t"                                              \
    ".set pop\n\t"

#define A_ST2(op, e, f, g) \
    PLT_M3_OP(op, (e) + 0, (f) + 0, (g) + 0) PLT_M3_OP(op, (e) + 1, (f) + 1, (g) + 1) \
    PLT_M3_OP(op, (e) + 2, (f) + 2, (g) + 2) PLT_M3_OP(op, (e) + 3, (f) + 3, (g) + 3)

static int add_run(plt_ctx_t *ctx, const plt_node_t *node)
{
    if (node->n_in != 2) return plt_ctx_fail(ctx, "add: layer %u needs 2 inputs", node->rec->id);
    /* An NDHWC16 add (plt_engine_load) is the same elementwise pass over half
     * the bytes: its operands and its output share the layout. */
    const uint32_t n = node->in[0].layout == PLT_FMT_NDHWC16 ? node->in[0].mem.bytes
                                                            : buf_bytes(&node->in[0]);
    const volatile uint8_t *a = plt_ctx_ptr(ctx, node->in[0].mem);
    const volatile uint8_t *bb = plt_ctx_ptr(ctx, node->in[1].mem);
    volatile uint8_t *dst = plt_ctx_ptr(ctx, node->out.mem);
    const volatile int32_t *p = (const volatile int32_t *)plt_ctx_ptr(ctx, node->table);
    if (!a || !bb || !dst || !p) return plt_ctx_fail(ctx, "add: layer %u operand unmapped", node->rec->id);

    const int32_t Ca = p[0], Cb = p[1], K = p[2];
    const int32_t OA = 128 + p[3], OB = 128 + p[4], zo = p[5];
    const int32_t RND = (int32_t)1 << (K - 1);

    const uint8_t *pa = (const uint8_t *)a, *pb = (const uint8_t *)bb;
    uint8_t       *pd = (uint8_t *)dst;
    uint32_t i = 0;

    const uint32_t t0 = plt_prof_now(ctx);

    add_fixed_t f;
    const int ok = add_split(Ca, Cb, K, OA, OB, &f) == 0;

    if (ok && n >= 64) {
        int32_t cst[A_NCONST * 16] __attribute__((aligned(64)));
        const int32_t cv[A_NCONST] = { 0, f.a1, f.b1, f.a0, f.b0,
                                       f.s ? f.t0 : f.t0 + f.v0,
                                       f.v0, zo + 128, 255, f.s, f.ks };
        for (int c = 0; c < A_NCONST; c++)
            for (int l = 0; l < 16; l++) cst[c * 16 + l] = cv[c];

        uint32_t cnt = n >> 6;                       /* 64 bytes per pass */
        i = cnt << 6;

        {   /* The constants, once: two 32-byte halves each, the base walking.
             * $t4, deliberately not one of the loop's three pinned pointers --
             * a local register variable owns its register for its whole life,
             * so reusing one here would quietly destroy a live pointer. */
            register const int32_t *pc __asm__("t4") = cst;
            __asm__ __volatile__(".set push\n\t.set noreorder\n\t"
                PLT_M3_LUO(A_Z,   0, 12) PLT_M3_LUO(A_Z,   1, 12)
                PLT_M3_LUO(A_CA1, 0, 12) PLT_M3_LUO(A_CA1, 1, 12)
                PLT_M3_LUO(A_CB1, 0, 12) PLT_M3_LUO(A_CB1, 1, 12)
                PLT_M3_LUO(A_CA0, 0, 12) PLT_M3_LUO(A_CA0, 1, 12)
                PLT_M3_LUO(A_CB0, 0, 12) PLT_M3_LUO(A_CB0, 1, 12)
                PLT_M3_LUO(A_T0,  0, 12) PLT_M3_LUO(A_T0,  1, 12)
                PLT_M3_LUO(A_V0,  0, 12) PLT_M3_LUO(A_V0,  1, 12)
                PLT_M3_LUO(A_ZO,  0, 12) PLT_M3_LUO(A_ZO,  1, 12)
                PLT_M3_LUO(A_HI,  0, 12) PLT_M3_LUO(A_HI,  1, 12)
                PLT_M3_LUO(A_S,   0, 12) PLT_M3_LUO(A_S,   1, 12)
                PLT_M3_LUO(A_KS,  0, 12) PLT_M3_LUO(A_KS,  1, 12)
                ".set pop\n\t" : "+r"(pc) :: "memory");
        }

        /* the three walking pointers, pinned: the lane load/store words name
         * their base register number directly */
        register const uint8_t *sa __asm__("t0") = pa;
        register const uint8_t *sb __asm__("t1") = pb;
        register uint8_t       *sd __asm__("t2") = pd;

        if (f.s) __asm__ __volatile__(
            ".set push\n\t.set noreorder\n\t.set noat\n\t"
            "1:\n\t"
            A_HEAD
            A_ST (PLT_M3_MULW, 16, A_CA1, 24)            /* T  = a*A1        */
            A_ST (PLT_M3_MULW, 20, A_CB1, 28)            /* .. + b*B1        */
            A_ST2(PLT_M3_ADDW, 24, 28,    24)
            A_ST (PLT_M3_ADDW, 24, A_T0,  24)            /* .. + T0          */
            A_ST (PLT_M3_MULW, 16, A_CA0, 28)            /* V  = a*A0        */
            A_ST (PLT_M3_MULW, 20, A_CB0, 20)            /* .. + b*B0        */
            A_ST2(PLT_M3_ADDW, 28, 20,    28)
            A_ST (PLT_M3_ADDW, 28, A_V0,  28)            /* .. + V0          */
            A_ST (PLT_M3_SRAW, 28, A_S,   28)            /* V >>= s          */
            A_ST2(PLT_M3_ADDW, 24, 28,    24)            /* T += V           */
            A_ST (PLT_M3_SRAW, 24, A_KS,  24)            /* T >>= K - s      */
            A_ST (PLT_M3_ADDW, 24, A_ZO,  24)
            A_ST (PLT_M3_MAXSW, 24, A_Z,  24)
            A_ST (PLT_M3_MINSW, 24, A_HI, 24)
            A_TAIL
            : [pa] "+r"(sa), [pb] "+r"(sb), [pd] "+r"(sd), [cnt] "+r"(cnt)
            :: "memory");
        else {
            /* No split needed: A0 = B0 = 0, so V is just the rounding term and
             * folds into T's seed.  Four pipeline stages disappear -- which is
             * every add in this model except layer 41. */
            __asm__ __volatile__(
                ".set push\n\t.set noreorder\n\t.set noat\n\t"
                "1:\n\t"
                A_HEAD
                A_ST (PLT_M3_MULW, 16, A_CA1, 24)
                A_ST (PLT_M3_MULW, 20, A_CB1, 28)
                A_ST2(PLT_M3_ADDW, 24, 28,    24)
                A_ST (PLT_M3_ADDW, 24, A_T0,  24)        /* T0 already has V0 */
                A_ST (PLT_M3_SRAW, 24, A_KS,  24)
                A_ST (PLT_M3_ADDW, 24, A_ZO,  24)
                A_ST (PLT_M3_MAXSW, 24, A_Z,  24)
                A_ST (PLT_M3_MINSW, 24, A_HI, 24)
                A_TAIL
                : [pa] "+r"(sa), [pb] "+r"(sb), [pd] "+r"(sd), [cnt] "+r"(cnt)
                :: "memory");
        }
    }

    for (; i < n; i++) {                       /* the tail, and any layer that
                                                * no split keeps inside int32 */
        long long acc = (long long)Ca * ((int)pa[i] - OA)
                      + (long long)Cb * ((int)pb[i] - OB);
        int q = (int)((acc + RND) >> K) + zo;
        if (q < -128) q = -128; else if (q > 127) q = 127;
        pd[i] = (uint8_t)(q + 128);
    }
    plt_prof_mark(ctx, &ctx->prof.tail_us, t0);
    return 0;
}

const plt_kernel_t plt_kernel_add = { PLT_OP_ADD, PLT_EX_MXU_ADD, "add", add_plan, add_run };
