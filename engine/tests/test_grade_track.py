"""hs grade finds the head track of the MOVE, whatever the render was named.

hs render names a render of an archived model <move>_<model>; the track lives at
move/<move>_frame.json. Looking it up by render name missed every time and fell back to a
centred crop without saying so.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.project import Project  # noqa: E402
from hs.stages.grade import source_move  # noqa: E402


class TrackLookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pj = Project(self.tmp.name, create=True)

    def test_archive_render_maps_back_to_its_move(self):
        self.pj.m["stages"]["render"] = {"status": "done",
                                         "runs": {"shot02_exposure-excl": {"move": "shot02"}}}
        self.assertEqual(source_move(self.pj, "shot02_exposure-excl"), "shot02")

    def test_current_export_render_is_its_own_move(self):
        self.pj.m["stages"]["render"] = {"status": "done", "runs": {"shot02": {"move": "shot02"}}}
        self.assertEqual(source_move(self.pj, "shot02"), "shot02")

    def test_no_record_falls_back_to_the_name(self):
        self.assertEqual(source_move(self.pj, "boom"), "boom")


if __name__ == "__main__":
    unittest.main()
