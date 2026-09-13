#!/usr/bin/env python3
# fake Brush: same argv contract, same stdout lines, writes real binary plys into --export-path
import sys, os, time, numpy as np, struct
args = sys.argv[1:]
def opt(name, default):
    return args[args.index(name)+1] if name in args else default
total = int(opt("--total-train-iters", 30000)); every = int(opt("--export-every", 5000))
exp = opt("--export-path", "."); name = opt("--export-name", "export_{iter}.ply"); refine = int(opt("--refine-every", 200))
growth_stop = int(opt("--growth-stop-iter", 15000)); start = int(opt("--start-iter", 0))
print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')} INFO  brush_cli] Compute backend: Metal"); sys.stdout.flush()
print("[.. INFO  brush_cli] Loaded dataset with 132 training, 0 eval views"); sys.stdout.flush()
os.makedirs(exp, exist_ok=True)
digits = len(str(total))
props = ["x","y","z","nx","ny","nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)] + ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
def write_ply(path, n):
    rng = np.random.default_rng(n)
    arr = np.zeros((n, len(props)), "<f4")
    arr[:, 0:3] = rng.normal([-0.0126, 0.0002, 0.2021], 0.15, (n, 3)); arr[:, props.index("opacity")] = rng.normal(2, 2, n)
    arr[:, props.index("scale_0"):props.index("scale_0")+3] = np.log(rng.uniform(0.002, 0.05, (n, 3)))
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {p}\n" for p in props) + "end_header\n"
    open(path, "wb").write(hdr.encode() + arr.tobytes())
splats = 70000
for it in range(start, total + 1):
    if it % refine == 0 and it > 0:
        if it <= growth_stop: splats += 400
        print(f"[.. INFO  brush_cli] Refine iter {it}, {splats} splats."); sys.stdout.flush()
    if it % every == 0 and it > 0:
        write_ply(os.path.join(exp, name.replace("{iter}", str(it).zfill(digits))), splats); time.sleep(0.05)
print("Training took 1s"); print("[.. INFO  brush_cli] Done training! Took 1s.")
