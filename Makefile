# t41-pluto: the runtime, cross-compiled for the Ingenic T41 (MIPS32r2, uClibc).
#
# Three things live outside this repo and are pointed at by variable:
#
#   CROSS   the mipsel uClibc cross-compiler
#   ROOTFS  an extracted device rootfs, for its libc and friends
#   IMPSDK  Ingenic SDK 1.2.6 -- run `python3 tools/fetch_sdk.py` once
#
# See docs/BUILD.md.  The previous host/deploy/test targets are in Makefile.old.

CROSS  ?= $(HOME)/x-tools/uclibc/bin/mipsel-buildroot-linux-uclibc-gcc
ROOTFS ?= $(abspath ..)/rootfs-extracted
# libimp/libalog come from SDK 1.2.6, whose build (20230620a) matches the driver
# this camera loads from flash.  The 1.1.1 SDK reports 20211215a, which
# IMP_ISP_Open refuses -- that mismatch is the reason this variable exists.
IMPSDK ?= $(abspath ..)/references/ingenic-t41-sdk-1.2.6
# `/` on the camera is a read-only squashfs, so the matching libimp cannot
# replace the stale one in /usr/lib; it ships beside the binary and is found by
# rpath.
IMPDEV ?= /run/pluto/lib

BUILD  := build
TARGET := $(BUILD)/plt

CFLAGS ?= -O1 -g
CFLAGS += -march=mips32r2 -Wall -Iruntime -I$(IMPSDK)/include

LDFLAGS = -nodefaultlibs \
          -L$(IMPSDK)/lib -L$(ROOTFS)/usr/lib -L$(ROOTFS)/lib \
          -Wl,-rpath-link,$(IMPSDK)/lib \
          -Wl,-rpath-link,$(ROOTFS)/usr/lib -Wl,-rpath-link,$(ROOTFS)/lib \
          -Wl,-rpath,$(IMPDEV) \
          -Wl,--no-as-needed,-l:libimp.so,-l:libalog.so,-l:libpthread.so.0,-l:librt.so.0,-l:libdl.so.0,-l:libm.so.0,-l:libc.so.0 \
          -Wl,--unresolved-symbols=ignore-in-shared-libs \
          -Wl,--dynamic-linker=/lib/ld-uClibc-mipsn8.so.0 \
          -lgcc

SRCS := runtime/tools/plt.c runtime/plt_engine.c \
        $(wildcard runtime/hal/*.c) $(wildcard runtime/core/*.c) \
        $(wildcard runtime/exec/*.c) $(wildcard runtime/io/*.c)
HDRS := runtime/plt_engine.h \
        $(wildcard runtime/hal/*.h) $(wildcard runtime/core/*.h) \
        $(wildcard runtime/exec/*.h) $(wildcard runtime/io/*.h)

.PHONY: all clean

all: $(TARGET)

# Dynamic, because libimp is: a process gets one libc, so -static is not an
# option once the camera is involved.  patch_fp64 sets EF_MIPS_NAN2008 and
# rewrites .MIPS.abiflags to FP64 -- without it the device's loader rejects the
# binary outright (docs/BUILD.md).
$(TARGET): $(SRCS) $(HDRS) Makefile
	@mkdir -p $(@D)
	$(CROSS) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)
	python3 tools/deploy/patch_fp64.py $@
	@$(CROSS:gcc=readelf) -h $@ | grep -q nan2008 || { echo "plt: FP64/nan2008 patch did not take"; exit 1; }
	@$(CROSS:gcc=readelf) -d $@ | grep -q 'libc\.so\.1' && { echo "plt: linked our libc, not the device's"; exit 1; } || true

clean:
	rm -f $(TARGET)
