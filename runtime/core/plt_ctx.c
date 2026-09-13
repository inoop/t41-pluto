#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core/plt_ctx.h"

int plt_ctx_open(plt_ctx_t *ctx)
{
    memset(ctx, 0, sizeof *ctx);
    if (plt_dev_open(&ctx->dev) != 0)
        return plt_ctx_fail(ctx, "cannot open the NNA device");
    plt_arena_init(&ctx->oram, PLT_SPACE_ORAM, PLT_ORAM_SIZE);
    plt_arena_init(&ctx->nmem, PLT_SPACE_NMEM, PLT_NMEM_SIZE);
    return 0;
}

void plt_ctx_close(plt_ctx_t *ctx)
{
    free(ctx->host_base);
    ctx->host_base  = NULL;
    ctx->model_base = NULL;   /* borrowed: the model image owns it */
    free(ctx->scratch);
    ctx->scratch = NULL;
    ctx->scratch_bytes = 0;
    plt_dev_close(&ctx->dev);
}

uint8_t *plt_ctx_scratch(const plt_ctx_t *ctx, uint32_t bytes)
{
    return ctx->scratch_bytes >= bytes ? ctx->scratch : NULL;
}

int plt_ctx_reserve_host(plt_ctx_t *ctx, uint32_t bytes)
{
    free(ctx->host_base);
    ctx->host_base = malloc(bytes);
    if (!ctx->host_base) return plt_ctx_fail(ctx, "no room for a %u-byte activation arena",
                                             bytes);
    plt_arena_init(&ctx->host, PLT_SPACE_HOST, bytes);
    return 0;
}

int plt_ctx_reserve_scratch(plt_ctx_t *ctx, uint32_t bytes)
{
    if (ctx->scratch_bytes >= bytes) return 0;
    free(ctx->scratch);
    ctx->scratch = malloc(bytes);
    ctx->scratch_bytes = ctx->scratch ? bytes : 0;
    return ctx->scratch ? 0 : plt_ctx_fail(ctx, "no room for %u bytes of scratch", bytes);
}

volatile uint8_t *plt_ctx_ptr(const plt_ctx_t *ctx, plt_mem_t mem)
{
    if (!plt_mem_present(mem)) return NULL;
    switch (mem.space) {
    case PLT_SPACE_ORAM: return ctx->dev.oram + mem.off;
    case PLT_SPACE_NMEM: return ctx->dev.nmem + mem.off;
    case PLT_SPACE_HOST:  return ctx->host_base  ? ctx->host_base  + mem.off : NULL;
    case PLT_SPACE_MODEL: return ctx->model_base ? ctx->model_base + mem.off : NULL;
    default:             return NULL;
    }
}

int plt_ctx_fail(plt_ctx_t *ctx, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->err, sizeof ctx->err, fmt, ap);
    va_end(ap);
    return -1;
}
