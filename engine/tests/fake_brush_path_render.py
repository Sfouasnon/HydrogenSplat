#!/usr/bin/env python3
import sys, os, json, time
ply = sys.argv[1]; a = sys.argv[2:]
path = a[a.index("--path")+1]; out = a[a.index("-o")+1]; width = int(a[a.index("--width")+1]) if "--width" in a else None
sp = json.load(open(path)); n = len(sp["frames"]); rw = width or sp["width"]; rh = int(round(rw * sp["height"]/sp["width"]))
print(f"path: {n} frames, native {sp['width']}x{sp['height']}, rendering {rw}x{rh}, fov 53.0x31.0 deg, centre_uv 0.5,0.5")
print(f"loaded {int(open(ply,'rb').read(200).split(b'element vertex ')[1].split()[0])} splats"); sys.stdout.flush()
os.makedirs(out, exist_ok=True)
import numpy as np, cv2
t0 = time.time()
for i in range(n):
    im = np.full((rh, rw, 3), (i*3) % 255, np.uint8); cv2.putText(im, f"frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4)
    cv2.imwrite(os.path.join(out, f"frame_{i:04d}.png"), im)
    if i % 10 == 0 or i + 1 == n: print(f"  frame {i+1}/{n}  ({time.time()-t0:.1f}s elapsed)"); sys.stdout.flush()
print(f"wrote {n} frames to {out} in {time.time()-t0:.1f}s")
