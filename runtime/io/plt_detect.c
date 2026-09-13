#include "io/plt_detect.h"

#include <math.h>
#include <stdlib.h>

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* One raw candidate before NMS. */
typedef struct { float x1, y1, x2, y2, score; int cls; } cand_t;

static int by_score_desc(const void *a, const void *b)
{
    float d = ((const cand_t *)b)->score - ((const cand_t *)a)->score;
    return (d > 0) - (d < 0);
}

static float iou_(const cand_t *a, const cand_t *b)
{
    float xx1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float yy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float xx2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float yy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float w = xx2 - xx1, h = yy2 - yy1;
    if (w <= 0 || h <= 0) return 0.0f;
    float inter = w * h;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter / (area_a + area_b - inter + 1e-9f);
}

int plt_yolox_decode(const plt_det_scale_t scales[PLT_DET_SCALES], int num_classes,
                     float conf_thr, float nms_thr, plt_det_t *out, int max_out)
{
    /* Upper bound on candidates: every cell of every scale. */
    long cap = 0;
    for (int s = 0; s < PLT_DET_SCALES; s++) cap += (long)scales[s].grid * scales[s].grid;
    cand_t *cand = malloc(sizeof(cand_t) * (size_t)cap);
    if (!cand) return 0;
    int nc = 0;

    for (int s = 0; s < PLT_DET_SCALES; s++) {
        const plt_det_scale_t *S = &scales[s];
        const int g = S->grid, n = g * g, st = S->stride;
        for (int y = 0; y < g; y++) {
            for (int x = 0; x < g; x++) {
                const int i = y * g + x;
                float r0 = S->reg_s * (S->reg[0 * n + i] - S->reg_zp);
                float r1 = S->reg_s * (S->reg[1 * n + i] - S->reg_zp);
                float r2 = S->reg_s * (S->reg[2 * n + i] - S->reg_zp);
                float r3 = S->reg_s * (S->reg[3 * n + i] - S->reg_zp);
                float cx = (r0 + x) * st, cy = (r1 + y) * st;
                float bw = expf(r2) * st, bh = expf(r3) * st;
                float objp = sigmoidf_(S->obj_s * (S->obj[i] - S->obj_zp));

                /* best class at this cell */
                int best = 0; float bestv = -1e30f;
                for (int c = 0; c < num_classes; c++) {
                    float v = S->cls[c * n + i] - S->cls_zp;   /* monotone in logit */
                    if (v > bestv) { bestv = v; best = c; }
                }
                float score = objp * sigmoidf_(S->cls_s * bestv);
                if (score <= conf_thr) continue;
                cand[nc].x1 = cx - bw / 2; cand[nc].y1 = cy - bh / 2;
                cand[nc].x2 = cx + bw / 2; cand[nc].y2 = cy + bh / 2;
                cand[nc].score = score; cand[nc].cls = best; nc++;
            }
        }
    }

    qsort(cand, nc, sizeof(cand_t), by_score_desc);
    char *dead = calloc((size_t)nc, 1);
    int nout = 0;
    for (int i = 0; i < nc && nout < max_out; i++) {
        if (dead[i]) continue;
        out[nout].x1 = cand[i].x1; out[nout].y1 = cand[i].y1;
        out[nout].x2 = cand[i].x2; out[nout].y2 = cand[i].y2;
        out[nout].score = cand[i].score; out[nout].cls = cand[i].cls;
        nout++;
        for (int j = i + 1; j < nc; j++)
            if (!dead[j] && cand[j].cls == cand[i].cls && iou_(&cand[i], &cand[j]) > nms_thr)
                dead[j] = 1;
    }
    free(dead); free(cand);
    return nout;
}
