#!/usr/bin/env python3
"""Stand-in for ffprobe: reports every file as a Holocam 1.18.2 2x1 video."""
import json, os, sys
path = sys.argv[-1]
print(json.dumps({
    "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264", "width": 3840, "height": 1080,
                 "r_frame_rate": "30/1", "avg_frame_rate": "30/1", "nb_frames": "90"}],
    "format": {"format_name": "mov,mp4", "duration": "3.0", "size": str(os.path.getsize(path)),
               "tags": {"comment": "leia3d_layout=2x1;leia3d_width_per_view=1920;leia3d_height_per_view=1080;"
                                   "leia3d_recording_software_version=1.18.2;"}}}))
