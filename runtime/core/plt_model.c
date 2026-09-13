#include "core/plt_model.h"

#include <stdlib.h>
#include <string.h>

static long read_all(const char *path, uint8_t **out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return -1; }
    fclose(f);
    *out = buf;
    return n;
}

int plt_model_load(const char *path, plt_model_t *m)
{
    memset(m, 0, sizeof *m);

    uint8_t *buf = NULL;
    long n = read_all(path, &buf);
    if (n < 0)
        return -1;
    if ((size_t)n < sizeof(plt_header_rec_t)) { free(buf); return -3; }

    const plt_header_rec_t *h = (const plt_header_rec_t *)buf;
    if (h->magic != PLT_MAGIC) { free(buf); return -2; }
    if ((size_t)h->tensors_off + (size_t)h->num_tensors * sizeof(plt_tensor_rec_t) > (size_t)n ||
        (size_t)h->layers_off  + (size_t)h->num_layers  * sizeof(plt_layer_rec_t)  > (size_t)n ||
        (size_t)h->blob_off    + (size_t)h->blob_size                          > (size_t)n) {
        free(buf);
        return -3;
    }

    m->data    = buf;
    m->size    = (size_t)n;
    m->header  = h;
    m->tensors = (const plt_tensor_rec_t *)(buf + h->tensors_off);
    m->layers  = (const plt_layer_rec_t  *)(buf + h->layers_off);
    m->blob    = buf + h->blob_off;
    return 0;
}

void plt_model_free(plt_model_t *m)
{
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
}

void plt_model_dump(const plt_model_t *m, FILE *out)
{
    const plt_header_rec_t *h = m->header;

    fprintf(out, "header magic=%u version=%u.%u flags=%u\n",
            h->magic, h->version_major, h->version_minor, h->flags);
    fprintf(out, "header counts tensors=%u layers=%u inputs=%u outputs=%u\n",
            h->num_tensors, h->num_layers, h->num_inputs, h->num_outputs);
    fprintf(out, "header input_ids=%u,%u,%u,%u\n",
            h->input_ids[0], h->input_ids[1], h->input_ids[2], h->input_ids[3]);
    fprintf(out, "header output_ids=%u,%u,%u,%u\n",
            h->output_ids[0], h->output_ids[1], h->output_ids[2], h->output_ids[3]);
    fprintf(out, "header offsets tensors=%u layers=%u blob=%u blob_size=%u arena=%u\n",
            h->tensors_off, h->layers_off, h->blob_off, h->blob_size, h->nmem_arena_hint);

    for (uint32_t k = 0; k < h->num_tensors; k++) {
        const plt_tensor_rec_t *t = &m->tensors[k];
        fprintf(out,
                "tensor %u id=%u name=%s dtype=%u format=%u ndims=%u "
                "shape=%d,%d,%d,%d scale=%08x zero_point=%d data_off=%u data_size=%u\n",
                k, t->id, t->name, t->dtype, t->format, t->ndims,
                t->shape[0], t->shape[1], t->shape[2], t->shape[3],
                t->scale_bits, t->zero_point, t->data_off, t->data_size);
    }

    for (uint32_t k = 0; k < h->num_layers; k++) {
        const plt_layer_rec_t *l = &m->layers[k];
        fprintf(out,
                "layer %u id=%u op=%u exec=%u act=%u in=%u,%u,%u,%u out=%u "
                "kernel=%ux%u stride=%ux%u pad=%u,%u,%u,%u groups=%u dilation=%ux%u "
                "bits=%u/%u/%u in_zp=%d weight=%u/%u reqtbl=%u/%u params=",
                k, l->id, l->op, l->exec, l->act,
                l->input_ids[0], l->input_ids[1], l->input_ids[2], l->input_ids[3], l->output_id,
                l->kh, l->kw, l->sh, l->sw, l->pt, l->pb, l->pl, l->pr,
                l->groups, l->dh, l->dw,
                l->in_bits, l->w_bits, l->out_bits, l->in_zp,
                l->weight_off, l->weight_size, l->reqtbl_off, l->reqtbl_size);
        for (int i = 0; i < 32; i++)
            fprintf(out, "%02x", l->params[i]);
        fprintf(out, "\n");
    }
}

const plt_tensor_rec_t *plt_model_tensor(const plt_model_t *m, uint32_t id)
{
    for (uint32_t i = 0; i < m->header->num_tensors; i++)
        if (m->tensors[i].id == id) return &m->tensors[i];
    return NULL;
}

const uint8_t *plt_model_blob(const plt_model_t *m, uint32_t off, uint32_t size)
{
    if (off == PLT_UNUSED) return NULL;
    if ((uint64_t)off + size > m->header->blob_size) return NULL;
    return m->blob + off;
}
