"""YOLOX's letterbox preprocessing, in one place.

This was copied into three modules -- dump_yolox.py, dump_detect.py and
pack_input.py -- and the third dropped the scale factor `r`, which is exactly
what is needed to map a detection back to original-image coordinates.  Any
evaluation over a dataset needs `r`, so the duplicate that loses it is the one
that would have been copied next.

The recipe is Megvii's own (YOLOX/yolox/data/data_augment.py ValTransform with
legacy=False): resize by a single factor that fits the long side, paste into a
114-grey square at the top-left, and hand the network BGR in CHW with values
still on the 0..255 scale -- no mean/std, no channel swap.
"""
import cv2
import numpy as np


def letterbox(img, size):
    """BGR HWC uint8 image -> (float32 CHW on the 0..255 scale, scale factor r).

    `r` is the factor the image was multiplied by, so a box in network pixels
    divides by `r` to land back in original-image pixels.
    """
    p = np.ones((size, size, 3), np.float32) * 114.0
    r = min(size / img.shape[0], size / img.shape[1])
    rw, rh = int(img.shape[1] * r), int(img.shape[0] * r)
    p[:rh, :rw] = cv2.resize(img, (rw, rh)).astype(np.float32)
    return p.transpose(2, 0, 1), r


def letterbox_file(path, size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return letterbox(img, size)
