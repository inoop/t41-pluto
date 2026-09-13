/* A baseline JPEG writer, for looking at what the camera actually gave us.
 *
 * This exists for debugging and nothing else: the classifier never encodes
 * anything.  It is here rather than on the host because "dump a frame and
 * convert it later" is one step too many when you are trying to work out
 * whether a bad prediction is the model or the picture.
 *
 * The camera hands us NV12, which is already YCbCr with 2x2-subsampled chroma
 * -- exactly what a 4:2:0 JPEG stores.  So the planes go in as they are: no
 * RGB round-trip, no colour conversion, and nothing to get wrong twice.
 */
#ifndef PLT_IO_JPEG_H
#define PLT_IO_JPEG_H

#include "plt_pre.h"

/* Write the view as a baseline JPEG.  `quality` is the usual 1..100 (85 is a
 * good default); it only scales the quantization tables.  Returns 0, or -1 with
 * errno set if the file could not be written. */
int plt_jpeg_write(const char *path, const plt_pre_view_t *v, int quality);

#endif /* PLT_IO_JPEG_H */
