#!/usr/bin/env python3
"""Stand-in for adb: one phone (serial FAKE01) whose camera folder holds the file named by
HS_FAKE_ADB_CLIP, exposed as /sdcard/DCIM/Camera/<its basename>. Enough for hs phone and
hs ingest --phone. HS_FAKE_ADB_MD5 overrides the md5 the phone reports."""
import hashlib, os, shutil, sys, time

CLIP = os.environ.get("HS_FAKE_ADB_CLIP", "")
CAM = "/sdcard/DCIM/Camera"
args = sys.argv[1:]
if args[:1] == ["-s"]:
    if args[1] != "FAKE01":
        sys.stderr.write(f"adb: device '{args[1]}' not found\n"); sys.exit(1)
    args = args[2:]


def remote_to_local(p):
    p = p.strip("'\"")
    return CLIP if CLIP and p == f"{CAM}/{os.path.basename(CLIP)}" else None


if args[:2] == ["devices", "-l"]:
    print("List of devices attached")
    print("FAKE01                 device usb:1-1 product:HydrogenONE model:H1A1000 device:HydrogenONE transport_id:9")
    print()
elif args[:1] == ["shell"]:
    cmd = " ".join(args[1:])
    if cmd.startswith("ls -l"):
        print("total 1\r")
        if CLIP:
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(CLIP)))
            print(f"-rw-rw---- 1 root sdcard_rw {os.path.getsize(CLIP)} {t} {os.path.basename(CLIP)}\r")
        print("-rw-rw---- 1 root sdcard_rw 8864362 2026-09-13 15:20 IMG_20260913_15200364_2x1.jpg\r")
    elif cmd.startswith("stat -c %s"):
        loc = remote_to_local(cmd.split(" ", 3)[3])
        if not loc:
            print(f"stat: '{cmd.split(' ', 3)[3]}': No such file or directory"); sys.exit(1)
        print(os.path.getsize(loc))
    elif cmd.startswith("md5sum"):
        loc = remote_to_local(cmd.split(" ", 1)[1])
        md5 = os.environ.get("HS_FAKE_ADB_MD5") or hashlib.md5(open(loc, "rb").read()).hexdigest()
        print(f"{md5}  {cmd.split(' ', 1)[1].strip(chr(39))}")
elif args[:1] == ["pull"]:
    loc = remote_to_local(args[1])
    if not loc:
        sys.stderr.write(f"adb: error: failed to stat remote object '{args[1]}'\n"); sys.exit(1)
    shutil.copyfile(loc, args[2])
    print(f"{args[1]}: 1 file pulled, 0 skipped.")
else:
    sys.stderr.write(f"fake adb: unsupported {args}\n"); sys.exit(1)
