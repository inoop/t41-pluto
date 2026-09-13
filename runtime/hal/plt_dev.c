#define _GNU_SOURCE
#include "plt_dev.h"
#include "plt_isa_mxu.h"

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

/* soc-nna FLUSHCACHE ioctl: _IOWR('c',2,int); arg {addr|0x80000000, len, dir}. */
#define PLT_IOCTL_FLUSHCACHE 0xc0046302u
struct plt_flush { uint32_t addr, len, dir; };

static volatile uint8_t *map_win(int fd, uint32_t phys, uint32_t len)
{
    void *p = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phys);
    if (p == MAP_FAILED) { fprintf(stderr, "mmap %08x/%x failed\n", phys, len); return 0; }
    return (volatile uint8_t *)p;
}

int plt_dev_open(plt_dev_t *d)
{
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs);       /* pin BEFORE open (10.2) */
    sched_setaffinity(0, sizeof cs, &cs);

    d->fd_nna = open("/dev/soc-nna", O_RDWR);
    if (d->fd_nna < 0) { perror("/dev/soc-nna"); return -1; }
    d->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (d->fd_mem < 0) { perror("/dev/mem"); return -1; }

    /* nmem, IO and DESRAM through /dev/mem (uncached, coherent by construction);
     * ORAM through the driver (special-cased in soc_nna_mmap). */
    d->nmem = map_win(d->fd_mem, PLT_NMEM_PHYS, PLT_NMEM_SIZE);
    d->oram = map_win(d->fd_nna, PLT_ORAM_PHYS, PLT_ORAM_SIZE);
    d->des  = map_win(d->fd_mem, PLT_DESRAM_PHYS, PLT_DESRAM_SIZE);
    d->io   = map_win(d->fd_mem, PLT_IO_PHYS, PLT_IO_SIZE);
    return (d->nmem && d->oram && d->des && d->io) ? 0 : -1;
}

void plt_dev_close(plt_dev_t *d)
{
    if (d->fd_mem >= 0) close(d->fd_mem);
    if (d->fd_nna >= 0) close(d->fd_nna);
}

int plt_dev_cache_sync(plt_dev_t *d, uint32_t phys, uint32_t len, int dir)
{
    struct plt_flush fi = { 0x80000000u + phys, len, (uint32_t)dir };
    return ioctl(d->fd_nna, PLT_IOCTL_FLUSHCACHE, &fi);
}

void plt_word_copy(volatile void *dst, const void *src, size_t n)
{
    volatile uint32_t *dw = dst; const uint32_t *sw = src;
    for (size_t i = 0; i < n / 4; i++) dw[i] = sw[i];
    __asm__ __volatile__("sync" ::: "memory");   /* the writes must land before the array reads */
}

void plt_copy_fast(volatile void *dst, const volatile void *src, size_t n)
{
    volatile uint8_t       *d = (volatile uint8_t *)dst;
    const volatile uint8_t *s = (const volatile uint8_t *)src;

    /* vr4..vr7: the array owns vr0-3 (feed), vr10-13 (drain) and vr31 (words). */
    while (n >= 256) {
        PLT_VLD64(4, (const void *)(s +   0)); PLT_VLD64(5, (const void *)(s +  64));
        PLT_VLD64(6, (const void *)(s + 128)); PLT_VLD64(7, (const void *)(s + 192));
        PLT_VST64(4, (void *)(d +   0));       PLT_VST64(5, (void *)(d +  64));
        PLT_VST64(6, (void *)(d + 128));       PLT_VST64(7, (void *)(d + 192));
        d += 256; s += 256; n -= 256;
    }
    while (n >= 64) {
        PLT_VLD64(4, (const void *)s); PLT_VST64(4, (void *)d);
        d += 64; s += 64; n -= 64;
    }
    while (n >= 4) {
        *(volatile uint32_t *)d = *(const volatile uint32_t *)s;
        d += 4; s += 4; n -= 4;
    }
    while (n--) *d++ = *s++;
    __asm__ __volatile__("sync" ::: "memory");
}

void plt_fill_fast(volatile void *dst, uint8_t value, size_t n)
{
    volatile uint8_t *d = (volatile uint8_t *)dst;
    const uint32_t word = (uint32_t)value * 0x01010101u;

    while (n >= 4 && ((uintptr_t)d & 3) == 0) {
        *(volatile uint32_t *)d = word;
        d += 4; n -= 4;
    }
    while (n--) *d++ = value;
}

uint32_t plt_now_us(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (uint32_t)(t.tv_sec * 1000000u + t.tv_usec);
}
