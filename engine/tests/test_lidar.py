"""hs/lidar.py: scan parsing, units, Umeyama / ICP / global registration, up, ground, coverage.

Parsing runs on small hand-written files in every layout a phone app might export (binary and
ascii PLY with any property order and type, a mesh with faces, an element ahead of the
vertices, big-endian, a splat PLY, OBJ, XYZ / CSV / PTS). Registration runs on the synthetic
room of lidar_synth.py: a "solve" of 12,000 points covering 60 % of the scan with 3 mm noise,
moved by a random Sim(3). Bounds: global registration alone recovers the scale within 0.3 % and
the rotation within 0.3 degrees; ICP after it reaches a trimmed RMS <= 5 mm; three noisy
--pairs converge to the same pose; a cloud with nothing in common with the room is reported
as not aligned. HS_TIMING=1 prints how long global registration took.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import lidar as L  # noqa: E402
import lidar_synth as S  # noqa: E402


def _grid(n=12, seed=0):
    """A small corner (floor + two walls), metres, with colours."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 3.0, n)
    u, v = np.meshgrid(a, a * 0.8)
    u, v = u.ravel(), v.ravel()
    P = np.vstack([np.stack([u, np.zeros_like(u), v], 1), np.stack([u, v, np.zeros_like(u)], 1),
                   np.stack([np.zeros_like(u), v, u], 1)])
    C = rng.integers(0, 256, (len(P), 3)).astype(np.uint8)
    return P, C


class Parse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-lidar-")
        self.P, self.C = _grid()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_binary_point_cloud_in_metres(self):
        p = self.path("cloud.ply")
        S.write_ply(p, self.P, self.C, binary=True, normals=np.tile([0, 1.0, 0], (len(self.P), 1)))
        sc = L.load_scan(p)
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-3)
        np.testing.assert_array_equal(sc.colors, self.C)
        self.assertEqual(sc.normals.shape, (len(self.P), 3))
        m = sc.meta
        self.assertEqual((m["format"], m["units"], m["units_source"], m["faces"], m["is_mesh"]),
                         ("ply-binary_little_endian", "m", "extent", 0, False))
        self.assertTrue(m["units_confident"])
        self.assertEqual(m["vertex_properties"], ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"])
        self.assertAlmostEqual(max(m["extent_mm"]), 3000, delta=40)

    def test_ascii_mesh_with_faces(self):
        p = self.path("mesh.ply")
        faces = np.array([[0, 1, 12], [1, 13, 12], [2, 3, 14]])
        S.write_ply(p, self.P, self.C, binary=False, faces=faces)
        sc = L.load_scan(p)
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-2)
        np.testing.assert_array_equal(sc.colors, self.C)
        self.assertEqual(sc.meta["format"], "ply-ascii")
        self.assertEqual(sc.meta["faces"], 3)
        self.assertTrue(sc.meta["is_mesh"])
        # and the binary mesh (faces after the vertices, list uchar int)
        S.write_ply(p, self.P, None, binary=True, faces=faces)
        sc = L.load_scan(p)
        self.assertEqual((sc.meta["faces"], sc.colors), (3, None))
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-3)

    def test_units_mm_file_and_flags(self):
        pm, pmm = self.path("m.ply"), self.path("mm.ply")
        S.write_ply(pm, self.P, self.C)
        S.write_ply(pmm, self.P * 1000, self.C)
        a, b = L.load_scan(pm), L.load_scan(pmm)
        self.assertEqual((a.meta["units"], b.meta["units"]), ("m", "mm"))
        self.assertTrue(b.meta["units_confident"])
        np.testing.assert_allclose(a.points, b.points, atol=0.05)
        c = L.load_scan(pm, units="cm")
        self.assertEqual((c.meta["units"], c.meta["units_source"]), ("cm", "flag"))
        np.testing.assert_allclose(c.points, self.P * 10, atol=1e-3)

    def test_header_comment_declares_units(self):
        # a 120 mm object in mm would read as metres by extent; the header says otherwise
        p = self.path("small.ply")
        S.write_ply(p, self.P * 40.0, None, comments=["Exported by some app", "units: millimeters"])
        sc = L.load_scan(p)
        self.assertEqual((sc.meta["units"], sc.meta["units_source"]), ("mm", "header"))
        self.assertIn("units: millimeters", sc.meta["comments"])
        np.testing.assert_allclose(sc.points, self.P * 40.0, atol=1e-3)

    def test_any_property_layout(self):
        """double xyz after other properties, float r/g/b in 0..1, an element ahead of the
        vertices, big-endian: all read by name, not by position."""
        n = len(self.P)
        for endian, fmt in (("<", "binary_little_endian"), (">", "binary_big_endian")):
            dt = np.dtype([("confidence", "u1"), ("r", endian + "f4"), ("g", endian + "f4"), ("b", endian + "f4"),
                           ("z", endian + "f8"), ("x", endian + "f8"), ("y", endian + "f8")])
            arr = np.empty(n, dt)
            arr["confidence"] = 2
            arr["x"], arr["y"], arr["z"] = self.P[:, 0], self.P[:, 1], self.P[:, 2]
            arr["r"], arr["g"], arr["b"] = (self.C[:, i] / 255.0 for i in range(3))
            cam = np.array([(1.0, 2.0, 3.0)], dtype=[("px", endian + "f4"), ("py", endian + "f4"), ("pz", endian + "f4")])
            head = ["ply", f"format {fmt} 1.0", "comment ARKit export", "element camera 1", "property float px",
                    "property float py", "property float pz", f"element vertex {n}", "property uchar confidence",
                    "property float r", "property float g", "property float b", "property double z",
                    "property double x", "property double y", "element face 0",
                    "property list uchar int vertex_indices", "end_header"]
            p = self.path(f"odd_{fmt}.ply")
            with open(p, "wb") as f:
                f.write(("\n".join(head) + "\n").encode())
                f.write(cam.tobytes())
                f.write(arr.tobytes())
            sc = L.load_scan(p)
            np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-6, err_msg=fmt)
            np.testing.assert_array_equal(sc.colors, self.C)
            self.assertEqual(sc.meta["elements"], {"camera": 1, "vertex": n, "face": 0})
            self.assertEqual(sc.meta["colour_from"], "r/g/b")

    def test_crlf_header_and_nonfinite_rows(self):
        P = self.P.copy()
        P[5] = np.nan
        p = self.path("crlf.ply")
        S.write_ply(p, P, None, binary=True)
        raw = open(p, "rb").read()
        k = raw.index(b"end_header\n") + len(b"end_header\n")
        open(p, "wb").write(raw[:k].replace(b"\n", b"\r\n") + raw[k:])
        sc = L.load_scan(p)
        self.assertEqual(sc.meta["dropped_nonfinite"], 1)
        self.assertEqual(len(sc.points), len(P) - 1)

    def test_splat_ply_colour_from_dc(self):
        n = len(self.P)
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3"]
        arr = np.zeros(n, np.dtype([(k, "<f4") for k in names]))
        arr["x"], arr["y"], arr["z"] = self.P.T
        arr["f_dc_0"] = 1.0
        p = self.path("splat.ply")
        with open(p, "wb") as f:
            f.write(("\n".join(["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
                               + [f"property float {k}" for k in names] + ["end_header"]) + "\n").encode())
            f.write(arr.tobytes())
        sc = L.load_scan(p)
        self.assertTrue(sc.meta["gaussian_splat"])
        self.assertEqual(int(sc.colors[0, 0]), int(round(255 * (0.5 + L.SH_C0))))
        self.assertEqual(int(sc.colors[0, 1]), 128)

    def test_obj(self):
        p = self.path("scan.obj")
        with open(p, "w") as f:
            f.write("# Scaniverse\nmtllib scan.mtl\n")
            for q, c in zip(self.P, self.C):
                f.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {c[0] / 255:.6f} {c[1] / 255:.6f} {c[2] / 255:.6f}\n")
            f.write("vn 0 1 0\nvt 0.5 0.5\n")
            f.write("f 1/1/1 2/1/1 13/1/1\nf 2 14 13\n")
        sc = L.load_scan(p)
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-2)
        np.testing.assert_allclose(sc.colors.astype(int), self.C.astype(int), atol=1)
        self.assertEqual((sc.meta["format"], sc.meta["faces"], sc.meta["units"]), ("obj", 2, "m"))

    def test_xyz_csv_pts(self):
        p = self.path("scan.xyz")
        np.savetxt(p, np.hstack([self.P, self.C]), fmt="%.6f %.6f %.6f %d %d %d")
        sc = L.load_scan(p)
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-2)
        np.testing.assert_array_equal(sc.colors, self.C)
        p = self.path("scan.csv")
        with open(p, "w") as f:
            f.write("X,Y,Z,Red,Green,Blue,Nx,Ny,Nz\n")
            for q, c in zip(self.P * 1000, self.C):
                f.write(f"{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{c[0]},{c[1]},{c[2]},0,1,0\n")
        sc = L.load_scan(p)
        self.assertEqual((sc.meta["units"], sc.meta["header"][:3]), ("mm", ["x", "y", "z"]))
        np.testing.assert_allclose(sc.points, self.P * 1000, atol=1e-2)
        np.testing.assert_array_equal(sc.colors, self.C)
        np.testing.assert_allclose(sc.normals[0], [0, 1, 0])
        p = self.path("scan.pts")            # Leica-style: a count, then x y z intensity r g b
        with open(p, "w") as f:
            f.write(f"{len(self.P)}\n")
            for q, c in zip(self.P, self.C):
                f.write(f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} -1200 {c[0]} {c[1]} {c[2]}\n")
        sc = L.load_scan(p)
        self.assertEqual(len(sc.points), len(self.P))
        np.testing.assert_array_equal(sc.colors, self.C)

    def test_refusals(self):
        for name, body, why in (("scan.las", b"LASF", "unsupported"),
                                ("noxyz.ply", b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float a\nend_header\n1\n", "no x, y, z"),
                                ("notply.ply", b"hello\n", "not a PLY"),
                                ("short.ply", b"ply\nformat binary_little_endian 1.0\nelement vertex 100\nproperty float x\n"
                                              b"property float y\nproperty float z\nend_header\n" + b"\0" * 40, "truncated")):
            p = self.path(name)
            open(p, "wb").write(body)
            with self.assertRaises(ValueError, msg=name) as e:
                L.load_scan(p)
            self.assertIn(why, str(e.exception))

    def test_infer_units(self):
        self.assertEqual(L.infer_units(4.5), ("m", True))
        self.assertEqual(L.infer_units(4500.0), ("mm", True))
        self.assertEqual(L.infer_units(120.0), ("m", False))       # a 120 m scan: metres, but say so
        self.assertEqual(L.infer_units(600.0), ("mm", False))      # a 6 m room in cm, or 0.6 m in mm


class Geometry(unittest.TestCase):
    def test_umeyama_exact(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 3)) * 1000
        R = S.random_rotation(rng)
        s, R2, t, res = L.umeyama(X, 0.8 * X @ R.T + [1, 2, 3])
        self.assertAlmostEqual(s, 0.8, places=10)
        np.testing.assert_allclose(R2, R, atol=1e-10)
        self.assertLess(res.max(), 1e-6)
        s, R2, t, res = L.umeyama(X, X @ R.T + [1, 2, 3], with_scale=False)
        self.assertEqual(s, 1.0)
        np.testing.assert_allclose(R2, R, atol=1e-10)
        inv = L.invert((0.8, R, np.array([1.0, 2, 3])))
        np.testing.assert_allclose(L.apply(inv, L.apply((0.8, R, np.array([1.0, 2, 3])), X)), X, atol=1e-8)

    def test_subsample_is_deterministic_and_sized(self):
        P = S.room(density=2000)[0] * 1000
        a, b = L.subsample(P, 5000), L.subsample(P, 5000)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(4000 <= len(a) <= 5000, len(a))
        self.assertEqual(len(np.unique(a)), len(a))
        self.assertFalse(np.array_equal(a, L.subsample(P, 5000, seed=1)))
        self.assertEqual(len(L.subsample(P, 5000, method="random")), 5000)
        np.testing.assert_array_equal(L.subsample(P[:100], 5000), np.arange(100))
        # voxel: evenly spread, so a dense corner does not take the sample
        dense = np.vstack([P, np.random.default_rng(0).normal(0, 50, (60000, 3)) + [500, 500, 500]])
        vi = L.subsample(dense, 3000)
        self.assertLess(np.mean(vi >= len(P)), 0.05)

    def test_ground_plane_and_up(self):
        P = S.room(density=1500)[0] * 1000
        R = S.rot([1, 0.2, 0], 25.0)
        g = L.ground_plane(P @ R.T + [100, -40, 7], R @ [0, 1.0, 0])
        self.assertLess(L.angle_deg(g["normal"], R @ [0, 1.0, 0]), 0.2)
        self.assertLess(g["rms_mm"], 1.0)
        self.assertGreater(g["inlier_share"], 0.8)
        up, ang = L.up_axis(R, "y", current_up=R @ [0, 0.9962, 0.0872])
        np.testing.assert_allclose(up, R @ [0, 1.0, 0], atol=1e-12)
        self.assertAlmostEqual(ang, 5.0, delta=0.01)
        up, ang = L.up_axis(np.eye(3), "z")
        np.testing.assert_allclose(up, [0, 0, 1.0])
        self.assertIsNone(ang)

    def test_denoise_before_subsampling(self):
        P = S.room(density=1500)[0] * 1000
        rng = np.random.default_rng(2)
        strays = P.min(0) + rng.random((len(P) // 4, 3)) * (P.max(0) - P.min(0))
        Q = np.vstack([P, strays])
        keep = L.denoise(Q)
        self.assertGreater(keep[:len(P)].mean(), 0.99)
        self.assertLess(keep[len(P):].mean(), 0.1)
        # why it has to come first: a voxel subsample favours the strays
        vi = L.subsample(Q, 5000)
        self.assertGreater(np.mean(vi >= len(P)), 0.3)

    def test_constraints(self):
        rng = np.random.default_rng(5)

        def plane(o, u, v, n):
            ab = rng.random((n, 2))
            return o + ab[:, :1] * u + ab[:, 1:] * v

        X, Y, Z = np.eye(3) * 3000
        corner = [(plane(np.zeros(3), X, Z, 3000), np.tile([0, 1.0, 0], (3000, 1))),
                  (plane(np.zeros(3), X, Y, 3000), np.tile([0, 0, 1.0], (3000, 1))),
                  (plane(np.zeros(3), Z, Y, 3000), np.tile([1.0, 0, 0], (3000, 1)))]
        P = np.vstack([p for p, _ in corner])
        N = np.vstack([n for _, n in corner])
        c = L.constraints(P, N)
        self.assertLess(c["scale_sensitivity"], 1e-6)            # scaling about the corner moves nothing
        self.assertGreater(c["pose_min_eig"], 0.05)              # but the pose is fixed
        # a box standing in the room fixes the scale
        box = plane(np.array([1000.0, 0, 1000]), np.array([600.0, 0, 0]), np.array([0, 500.0, 0]), 1500)
        c = L.constraints(np.vstack([P, box]), np.vstack([N, np.tile([0, 0, 1.0], (1500, 1))]))
        self.assertGreater(c["scale_sensitivity"], 0.1)
        # a floor alone fixes neither the slide along it nor the turn about its normal
        c = L.constraints(corner[0][0], corner[0][1])
        self.assertLess(c["pose_min_eig"], 1e-9)

    def test_coverage_check(self):
        P = S.room(density=1500)[0] * 1000
        floor = P[(P[:, 1] == 0) & (np.abs(P[:, 0] - 1500) < 500) & (np.abs(P[:, 2] - 2000) < 500)]
        C = np.array([[1500.0, 1200, 2000], [1500.0, 1200, 9000]])      # inside the room; 5 m past its wall
        rows = L.coverage_check(C, P, pts=floor + [0, 2, 0], names=["in", "out"], near_k=200)
        self.assertTrue(rows[0]["ok"] and rows[0]["camera_ok"] and rows[0]["points_ok"])
        self.assertLess(rows[0]["points_median_to_scan_mm"], 5)
        self.assertEqual(rows[0]["points_seen"], 200)
        self.assertFalse(rows[1]["camera_ok"])
        self.assertFalse(rows[1]["ok"])
        # the camera is in the room but what it sees is not on the scan
        rows = L.coverage_check(C[:1], P, pts=floor + [0, 400, 0], near_k=200)
        self.assertTrue(rows[0]["camera_ok"])
        self.assertFalse(rows[0]["points_ok"])
        self.assertFalse(rows[0]["ok"])
        # with a frustum: a camera looking straight down sees the floor patch below it
        K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        Rd = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])         # x right, y down = +z world, z = -y world
        t = -Rd @ C[0]
        rows = L.coverage_check(C[:1], P, pts=np.vstack([floor, floor + [0, 5000, 0]]), K=K[None], R=Rd[None],
                                t=t[None], wh=[(640, 480)])
        self.assertEqual(rows[0]["points_seen"], len(floor))           # the copy above the camera is behind it


class Registration(unittest.TestCase):
    """The synthetic room: a mono solve (scale k in 0.7-1.4) and a stereo one (k = 1)."""

    @classmethod
    def setUpClass(cls):
        cls.mono = S.Scene(k=1.27, seed=11)
        cls.stereo = S.Scene(k=1.0, seed=12)
        cls.data = {}
        for name, sc in (("mono", cls.mono), ("stereo", cls.stereo)):
            X = sc.scan_m * 1000
            dense = X[L.subsample(X, 400000)]
            cls.data[name] = (X, dense, L.NN(dense))

    def truth(self, sc):
        """solve -> scan: X = (1/k) Rg^T (x - tg)."""
        return 1.0 / sc.k, sc.Rg.T, -(sc.Rg.T @ sc.tg) / sc.k

    def register(self, name, with_scale):
        sc = getattr(self, name)
        X, dense, _nn = self.data[name]
        t0 = time.monotonic()
        r = L.global_register(sc.pts[L.subsample(sc.pts, 5000)], X[L.subsample(X, 5000)], with_scale,
                              dst_ref=dense)
        dt = time.monotonic() - t0
        if os.environ.get("HS_TIMING"):
            sys.stderr.write(f"\nglobal_register {name} 5000 x 5000: {dt:.2f} s, {r['samples']} samples, "
                             f"{r['hypotheses']} hypotheses\n")
        self.assertLess(dt, 30.0)
        return r

    def test_global_register_mono_scale_and_rotation(self):
        r = self.register("mono", True)
        s0, R0, t0 = self.truth(self.mono)
        self.assertTrue(r["aligned"], r)
        self.assertLess(abs(r["s"] / s0 - 1), 0.003, r["s"] / s0)
        self.assertLess(L.rotation_angle_deg(r["R"], R0), 0.3)
        # ICP against the dense scan from there
        _X, _dense, nn = self.data["mono"]
        q = L.icp(self.mono.pts, None, (r["s"], r["R"], r["t"]), with_scale=True, index=nn, inlier_dist=50.0,
                  keep_within=50.0, scale_bounds=1.1)
        self.assertLessEqual(q["rms"], 5.0, q["rms_history"])
        self.assertGreater(q["inlier_fraction"], 0.99)
        self.assertLess(abs(q["s"] / s0 - 1), 0.0005)
        self.assertLess(L.rotation_angle_deg(q["R"], R0), 0.05)
        self.__class__.icp_mono = q

    def test_global_register_known_scale(self):
        r = self.register("stereo", False)
        s0, R0, t0 = self.truth(self.stereo)
        self.assertTrue(r["aligned"], r)
        self.assertEqual(r["s"], 1.0)
        self.assertLess(L.rotation_angle_deg(r["R"], R0), 0.3)
        self.assertLess(np.linalg.norm(r["t"] - t0), 20.0)

    def test_three_noisy_pairs_converge_to_the_same(self):
        sc = self.mono
        X, _dense, nn = self.data["mono"]
        s0, R0, t0 = self.truth(sc)
        rng = np.random.default_rng(4)
        # three points a person would click: the box's top, the ball's top, high on the far wall
        picks = [np.argmin(np.linalg.norm(sc.scan_m - q, axis=1)) for q in ([2.6, 0.5, 1.8], [1.3, 0.6, 2.8], [0.0, 2.3, 0.4])]
        scan_m = sc.scan_m[picks] + rng.normal(0, 0.01, (3, 3))                    # 10 mm of clicking error
        solve = sc.to_solve(sc.scan_m[picks] * 1000) + rng.normal(0, 10 * sc.k, (3, 3))
        s, R, t, res = L.umeyama(scan_m * 1000, solve, with_scale=True)           # scan mm -> solve
        T0 = L.invert((s, R, t))
        self.assertLess(abs(T0[0] / s0 - 1), 0.03)                                 # rough, from three clicks
        q = L.icp(sc.pts, None, T0, with_scale=True, index=nn, inlier_dist=50.0, keep_within=50.0, scale_bounds=1.25)
        self.assertLessEqual(q["rms"], 5.0)
        self.assertLess(abs(q["s"] / s0 - 1), 0.0005)
        self.assertLess(L.rotation_angle_deg(q["R"], R0), 0.05)

    def test_no_overlap_is_reported_not_hidden(self):
        X, dense, _nn = self.data["mono"]
        B = S.blob_cloud()
        r = L.global_register(B[L.subsample(B, 5000)], X[L.subsample(X, 5000)], True, dst_ref=dense,
                              max_samples=40)
        self.assertFalse(r["aligned"])
        self.assertLess(r["inlier_fraction"], 0.5)
        self.assertIn("inlier fraction", r["reason"])

    def test_strays_and_noise(self):
        # 10 mm noise and 20 % strays, removed from the full cloud before the subsample
        sc = S.Scene(k=0.93, seed=14, noise_mm=10.0, outliers=0.2)
        X = sc.scan_m * 1000
        P = sc.pts[L.denoise(sc.pts)]
        r = L.global_register(P[L.subsample(P, 5000)], X[L.subsample(X, 5000)], True,
                              dst_ref=X[L.subsample(X, 400000)])
        s0, R0, _t0 = self.truth(sc)
        self.assertTrue(r["aligned"], r["reason"])
        self.assertLess(abs(r["s"] / s0 - 1), 0.005)      # 10 mm of noise on a 5 m room: 0.44 % seen on the Mac
        self.assertLess(L.rotation_angle_deg(r["R"], R0), 0.3)

    def test_a_bare_corner_is_refused(self):
        # the solve covers only the corner (floor + two walls, a sliver of the ball): symmetric
        # under 120-degree turns and scaling about the apex, so any answer would be a guess
        sc = S.Scene(k=1.2, seed=15, covered=0.3)
        X = sc.scan_m * 1000
        r = L.global_register(sc.pts[L.subsample(sc.pts, 5000)], X[L.subsample(X, 5000)], True,
                              dst_ref=X[L.subsample(X, 400000)])
        self.assertFalse(r["aligned"])
        self.assertRegex(r["reason"], "ambiguous|scale")

    def test_icp_trims_and_bounds_the_scale(self):
        # a partial overlap with outliers: 20 % of the solve is junk off the surfaces
        sc = S.Scene(k=0.8, seed=13, outliers=0.2, n_solve=5000)
        X = sc.scan_m * 1000
        nn = L.NN(X)
        s0, R0, t0 = self.truth(sc)
        q = L.icp(sc.pts, None, (s0 * 1.01, S.rot([0, 1, 0], 1.0) @ R0, t0 + 30), with_scale=True, index=nn,
                  inlier_dist=50.0, keep_within=50.0)
        self.assertTrue(q["converged"])
        self.assertLess(abs(q["s"] / s0 - 1), 0.001)
        self.assertAlmostEqual(q["inlier_fraction"], 0.8, delta=0.03)
        self.assertLessEqual(q["rms"], 5.0)
        # the scale never leaves [s0 / b, s0 * b], however the correspondences pull
        q = L.icp(sc.pts, None, (s0 * 0.5, R0, t0), with_scale=True, index=nn, scale_bounds=1.2, max_iter=10)
        self.assertLessEqual(q["s"], s0 * 0.5 * 1.2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
