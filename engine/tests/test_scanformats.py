"""E57 and LAS scans are read as they come off a survey scanner.

Until 2026-10-05 `hs scale --lidar` took PLY, OBJ and text points — what a phone scanning app
exports — and an E57 from a Leica or a Faro had to go through CloudCompare first. hs/scanformats.py
reads E57 and LAS with numpy alone (the engine's Python has no libE57 wheel), and LAZ through
laspy when that is installed.

What these tests can and cannot show. The files are written by tests/scan_synth.py, so on their
own they prove the reader and that writer agree. That they agree with the formats was checked
once, by hand, against the reference libraries (pye57 0.4.19 / libE57, laspy 2.7):
  * this reader and libE57 gave identical points on libE57Format's reference files — bunnyDouble,
    bunnyInt32 (32-bit ScaledInteger), ColouredCubeFloat / Double (with colour), and las2e57's
    ColourRepresentation (10-bit ScaledInteger, 16-bit colour: the non-byte-aligned case) — and on
    a two-scan, posed file pye57 wrote;
  * libE57 read every kind of file scan_synth writes (double, single, scaled with invalid states,
    spherical; packets cut mid-record; page checksums) to the points this reader gets;
  * this reader and laspy agreed to the last bit on LAS 1.2 formats 0, 2, 3 and 1.4 formats 6, 7,
    8, and on a LAZ of the same points.
No file from a real scanner has been through it.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import lidar, scanformats  # noqa: E402

import scan_synth as S  # noqa: E402

QUAT = (0.9238795325112867, 0.0, 0.0, 0.3826834323650898)       # 45 degrees about z
SHIFT = (12.0, -7.5, 3.0)


class Base(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        rng = np.random.default_rng(3)
        self.n = 6000
        self.xyz = rng.normal(size=(self.n, 3)) * [4, 3, 1.5]
        self.rgb = rng.integers(0, 256, (self.n, 3)).astype(np.uint8)
        self.invalid = (rng.random(self.n) < 0.05).astype(int) * 2
        self.intensity = rng.random(self.n)

    def path(self, name):
        return os.path.join(self.t.name, name)

    def posed(self, p):
        return p @ scanformats._quat_matrix(*QUAT).T + np.array(SHIFT)


class E57(Base):
    def test_doubles_with_a_pose_colour_and_a_stream_to_step_over(self):
        f = self.path("a.e57")
        S.write_e57(f, [dict(xyz=self.xyz, rgb=self.rgb, pose=(QUAT, SHIFT), intensity=self.intensity, name="hall")])
        pts, rgb, nrm, meta, units = scanformats.load_e57(f)
        np.testing.assert_allclose(pts, self.posed(self.xyz), atol=1e-12)
        np.testing.assert_array_equal(rgb, self.rgb)
        self.assertIsNone(nrm)
        self.assertEqual(units, "m", "E57 is metres by its standard")
        self.assertEqual((meta["format"], meta["scans"], meta["records"], meta["stride"], meta["scan_names"]),
                         ("e57", 1, self.n, 1, ["hall"]))

    def test_two_scans_each_moved_by_its_own_pose(self):
        f = self.path("b.e57")
        S.write_e57(f, [dict(xyz=self.xyz, encoding="single"),
                        dict(xyz=self.xyz[:900] + 5, encoding="single", pose=(QUAT, SHIFT))])
        pts, rgb, _n, meta, _u = scanformats.load_e57(f)
        a = self.xyz.astype(np.float32).astype(np.float64)
        b = self.posed((self.xyz[:900] + 5).astype(np.float32).astype(np.float64))
        np.testing.assert_allclose(pts, np.concatenate([a, b]), atol=1e-12)
        self.assertIsNone(rgb, "no colour in either scan")
        self.assertEqual(meta["scans"], 2)

    def test_bit_packed_integers_invalid_returns_and_sixteen_bit_colour(self):
        """ScaledInteger at 0.5 mm: 15 to 16 bits a value here, so nothing sits on a byte boundary
        and the writer's packets cut values in half. Returns marked invalid are dropped."""
        f = self.path("c.e57")
        S.write_e57(f, [dict(xyz=self.xyz, rgb=self.rgb, colour_max=65535, encoding="scaled", scale=0.0005,
                             invalid=self.invalid, pose=(QUAT, SHIFT))])
        pts, rgb, _n, meta, _u = scanformats.load_e57(f)
        keep = self.invalid == 0
        np.testing.assert_allclose(pts, self.posed(np.round(self.xyz / 0.0005) * 0.0005)[keep], atol=1e-9)
        np.testing.assert_array_equal(rgb, self.rgb[keep])
        self.assertEqual(meta["dropped_invalid"], int((~keep).sum()))

    def test_spherical_coordinates(self):
        f = self.path("d.e57")
        S.write_e57(f, [dict(xyz=self.xyz, encoding="spherical", invalid=self.invalid)])
        pts, _rgb, _n, meta, _u = scanformats.load_e57(f)
        np.testing.assert_allclose(pts, self.xyz[self.invalid == 0], atol=1e-9)
        self.assertEqual(meta["coordinates"], "spherical")

    def test_a_field_that_never_changes_takes_no_bytes(self):
        flat = self.xyz.copy()
        flat[:, 2] = 1.25                                   # a ScaledInteger whose maximum is its minimum: 0 bits
        f = self.path("e.e57")
        S.write_e57(f, [dict(xyz=flat, encoding="scaled", scale=0.001)])
        pts, _rgb, _n, _m, _u = scanformats.load_e57(f)
        np.testing.assert_allclose(pts, np.round(flat / 0.001) * 0.001, atol=1e-9)

    def test_a_large_scan_is_thinned_while_it_is_read(self):
        f = self.path("f.e57")
        S.write_e57(f, [dict(xyz=self.xyz, rgb=self.rgb, encoding="scaled", scale=0.0005),
                        dict(xyz=self.xyz[:1000], rgb=self.rgb[:1000], encoding="double")])
        pts, rgb, _n, meta, _u = scanformats.load_e57(f, max_points=1000)
        k = meta["stride"]
        self.assertEqual(k, 7, "7,000 records into 1,000 points")
        want = np.concatenate([(np.round(self.xyz / 0.0005) * 0.0005)[::k], self.xyz[:1000][::k]])
        np.testing.assert_allclose(pts, want, atol=1e-9)
        np.testing.assert_array_equal(rgb, np.concatenate([self.rgb[::k], self.rgb[:1000][::k]]))

    def test_what_is_not_a_scan_is_refused_by_name(self):
        bad = self.path("not.e57")
        open(bad, "wb").write(b"PK\x03\x04" + b"\0" * 100)
        with self.assertRaisesRegex(ValueError, "not an E57 file"):
            scanformats.load_e57(bad)
        empty = self.path("empty.e57")
        S.write_e57(empty, [])
        with self.assertRaisesRegex(ValueError, "no scans"):
            scanformats.load_e57(empty)
        cut = self.path("cut.e57")
        S.write_e57(cut, [dict(xyz=self.xyz)])
        whole = open(cut, "rb").read()
        open(cut, "wb").write(whole[:48] + whole[48:4096])  # the header still points at an XML section further on
        with self.assertRaisesRegex(ValueError, "truncated"):
            scanformats.load_e57(cut)


class LAS(Base):
    def setUp(self):
        super().setUp()
        self.survey = self.xyz * [8, 6, 2] + [512345.0, 4212345.0, 120.0]     # a projected coordinate system

    def test_point_formats_with_and_without_colour(self):
        for version, rgb, fmt in (((1, 2), None, 0), ((1, 2), self.rgb, 2), ((1, 4), None, 6), ((1, 4), self.rgb, 7)):
            f = self.path(f"v{version[1]}_{fmt}.las")
            S.write_las(f, self.survey, rgb, version=version)
            pts, got, _n, meta, units = scanformats.load_las(f)
            np.testing.assert_allclose(pts, self.survey, atol=0.00051, err_msg=str((version, fmt)))   # stored to the mm
            self.assertEqual((meta["las_version"], meta["las_point_format"], meta["records"]),
                             (f"{version[0]}.{version[1]}", fmt, self.n))
            self.assertIsNone(units, "LAS does not say what its units are")
            if rgb is None:
                self.assertIsNone(got)
            else:
                np.testing.assert_array_equal(got, rgb)

    def test_thinned_while_it_is_read(self):
        f = self.path("big.las")
        S.write_las(f, self.survey, self.rgb, version=(1, 4))
        pts, rgb, _n, meta, _u = scanformats.load_las(f, max_points=1000)
        self.assertEqual(meta["stride"], 6)
        np.testing.assert_allclose(pts, self.survey[::6], atol=0.00051)
        np.testing.assert_array_equal(rgb, self.rgb[::6])

    def test_compressed_las_names_what_it_needs(self):
        f = self.path("scan.laz")
        S.write_las(f, self.survey)
        had = sys.modules.get("laspy", "absent")
        sys.modules["laspy"] = None                         # as on a machine without it
        try:
            with self.assertRaisesRegex(ValueError, r"laspy\[lazrs\]"):
                scanformats.load_las(f)
        finally:
            if had == "absent":
                sys.modules.pop("laspy", None)
            else:
                sys.modules["laspy"] = had
        with self.assertRaisesRegex(ValueError, "not a LAS file"):
            scanformats.load_las(__file__)


class ThroughLoadScan(Base):
    """hs/lidar.py's load_scan is what `hs scale --lidar` calls: millimetres, and one frame."""

    def test_an_e57_comes_back_in_millimetres(self):
        f = self.path("room.e57")
        S.write_e57(f, [dict(xyz=self.xyz, rgb=self.rgb)])
        scan = lidar.load_scan(f)
        np.testing.assert_allclose(scan.points, self.xyz * 1000.0, atol=1e-6)
        np.testing.assert_array_equal(scan.colors, self.rgb)
        self.assertEqual((scan.meta["format"], scan.meta["units"], scan.meta["units_source"]), ("e57", "m", "header"))
        self.assertIsNone(scan.meta["origin_shift_mm"], "a scan near its own origin stays where it is")

    def test_survey_coordinates_are_brought_to_the_origin_and_the_move_is_recorded(self):
        """A georeferenced scan sits thousands of kilometres from zero; the nearest-neighbour trees
        and the files written from it are single precision, good to a quarter of a metre out there."""
        f = self.path("stage.las")
        survey = self.xyz * [8, 6, 2] + [512345.0, 4212345.0, 120.0]
        S.write_las(f, survey)
        scan = lidar.load_scan(f)
        shift = np.array(scan.meta["origin_shift_mm"])
        self.assertEqual((scan.meta["units"], scan.meta["units_source"], scan.meta["units_confident"]),
                         ("m", "assumed", False), "no coordinate system in the file: metres, and said to be a guess")
        self.assertTrue(np.all(shift % 1000.0 == 0), "moved by whole metres")
        self.assertLess(float(np.abs(np.median(scan.points, axis=0)).max()), 1000.0)
        np.testing.assert_allclose(scan.points + shift, survey * 1000.0, atol=0.51)

    def test_a_survey_in_feet_says_so_in_its_coordinate_system(self):
        """US state plane surveys are often in US survey feet. The LAS header has no unit; the
        coordinate system record does, as WKT or as a GeoTIFF key."""
        metres = self.xyz * [8, 6, 2] + [1000.0, 2000.0, 30.0]
        wkt = (b'PROJCS["NAD83 / California zone 5 (ftUS)",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
               b'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
               b'PROJECTION["Lambert_Conformal_Conic_2SP"],UNIT["US survey foot",0.3048006096012192],AXIS["X",EAST]]\0')
        import struct
        geokeys = struct.pack("<16H", 1, 1, 0, 3, 1024, 0, 1, 1, 3072, 0, 1, 2229, 3076, 0, 1, 9002)   # ...units: 9002 = foot
        for name, vlr, unit, per in (("wkt", ("LASF_Projection", 2112, wkt), "usft", 1200.0 / 3937.0),
                                     ("keys", ("LASF_Projection", 34735, geokeys), "ft", 0.3048)):
            f = self.path(name + ".las")
            S.write_las(f, metres / per, vlrs=[("someone_else", 7, b"\1\2\3"), vlr])     # the file is in feet
            scan = lidar.load_scan(f)
            self.assertEqual((scan.meta["units"], scan.meta["units_source"], scan.meta["units_confident"]),
                             (unit, "header", True), name)
            moved = np.array(scan.meta["origin_shift_mm"] or [0.0, 0.0, 0.0])                # 2 km out: brought in
            np.testing.assert_allclose(scan.points + moved, metres * 1000.0, atol=0.2, err_msg=name)   # 0.001 ft steps

    def test_other_extensions_are_still_refused_and_say_what_is_read(self):
        f = self.path("scan.usdz")
        open(f, "wb").write(b"PK")
        with self.assertRaisesRegex(ValueError, "E57"):
            lidar.load_scan(f)


if __name__ == "__main__":
    unittest.main()
