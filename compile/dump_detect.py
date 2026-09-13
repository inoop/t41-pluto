"""Dump the integer datapath's 9 head outputs (planar int8) + meta for the C
decode host test, and print the Python reference detections."""
import os, sys, numpy as np, cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from compile import onnx_import as oi, detect_int as di, detect_ref as dr, preproc

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/detvec"
os.makedirs(OUT, exist_ok=True)
S = 416
m = oi.load(os.path.join(_ROOT, "fixtures/yolox/yolox_nano_int8.onnx"))
img = cv2.imread(os.path.join(_ROOT, "fixtures", "dog.jpg"))
chw, r = preproc.letterbox(img, S)
outs = di.run_graph(m, chw)                       # byte = q+128, per output name
byname = {L.output: L for L in m.layers}

def find(kind, s):
    for n in m.output_names:
        if f"{kind}_preds.{s}/" in n: return n
    raise KeyError
STR = [8, 16, 32]
meta = ["num_classes 80"]
for s in range(3):
    g = S // STR[s]
    for k, kind in (("reg","reg"), ("obj","obj"), ("cls","cls")):
        n = find(kind, s); q = (outs[n].astype(np.int16) - 128).astype(np.int8)  # planar [C,H,W]
        q.tofile(os.path.join(OUT, f"{kind}{s}.bin"))
    rq, oq, cq = byname[find("reg",s)].act_out, byname[find("obj",s)].act_out, byname[find("cls",s)].act_out
    meta.append(f"scale {s} grid {g} stride {STR[s]} "
                f"reg {rq.scale:.9g} {rq.zero_point} obj {oq.scale:.9g} {oq.zero_point} "
                f"cls {cq.scale:.9g} {cq.zero_point}")
open(os.path.join(OUT, "meta.txt"), "w").write("\n".join(meta) + "\n")

# Python reference (same integer outputs), in letterbox pixels
real = di.outputs_real(m, chw)
boxes, scores = dr.decode(real, m)
ref = []
for c in range(scores.shape[1]):
    sc = scores[:, c]; keep = sc > 0.30
    if not keep.any(): continue
    b, s2 = boxes[keep], sc[keep]
    for i in dr.nms(b, s2):
        ref.append((c, s2[i], *b[i]))
ref.sort(key=lambda t: -t[1])
with open(os.path.join(OUT, "ref.txt"), "w") as f:
    for c, sv, x1, y1, x2, y2 in ref:
        f.write(f"{c} {sv:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
print("wrote vectors + ref to", OUT, "(%d ref dets)" % len(ref))
