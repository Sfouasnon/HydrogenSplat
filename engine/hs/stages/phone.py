"""hs phone — list Hydrogen One phones and their 3D video clips over adb (strategy §4.1).

    hs phone [--adb PATH] [--serial SERIAL]

Emits one ``devices`` metric (every serial adb reports, with its state and model) and one
``clips`` metric per ready device: the ``VID_*_2x1.h4v`` files in ``/sdcard/DCIM/Camera``,
newest first, with size and phone-local mtime. USB and wireless adb are both just serials.
Pulling is ``hs ingest --phone SERIAL --remote PATH`` so the clip lands straight in the
project's ``source/`` and is MD5-checked against the phone.
"""
import os
import re
import shutil
import subprocess

from .. import events

STAGE = "phone"
CAMERA_DIR = "/sdcard/DCIM/Camera"
RE_LS = re.compile(r"^\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(VID_\S+_2x1\.h4v)\s*$")


def add_parser(sub):
    p = sub.add_parser("phone", help="list phones and their 3D video clips over adb")
    p.add_argument("--adb", default=os.environ.get("HS_ADB", "adb"))
    p.add_argument("--serial", default=None, help="only this device")
    return p


def adb_exe(adb):
    exe = shutil.which(os.path.expanduser(adb))
    if not exe:
        raise events.StageError(f"adb not found ({adb})", hint="brew install android-platform-tools")
    return exe


def adb(exe, *args, serial=None, timeout=30):
    argv = [exe] + (["-s", serial] if serial else []) + list(args)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise events.StageError(f"adb timed out: {' '.join(args)}", hint="unplug and replug the phone, or `adb kill-server`")
    return r


def parse_devices(text):
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        d = {"serial": parts[0], "state": parts[1]}
        for kv in parts[2:]:
            if ":" in kv:
                k, v = kv.split(":", 1)
                if k in ("model", "product", "device", "usb", "transport_id"):
                    d[k] = v
        out.append(d)
    return out


def parse_ls(text, directory=CAMERA_DIR):
    clips = []
    for line in text.replace("\r", "").splitlines():
        m = RE_LS.match(line.strip())
        if m:
            clips.append({"name": m.group(3), "path": f"{directory}/{m.group(3)}",
                          "bytes": int(m.group(1)), "mtime": m.group(2)})
    clips.sort(key=lambda c: (c["mtime"], c["name"]), reverse=True)
    return clips


def run(a):
    exe = adb_exe(a.adb)
    events.start(STAGE, "devices")
    r = adb(exe, "devices", "-l")
    if r.returncode != 0:
        raise events.StageError("adb devices failed", hint=(r.stderr or r.stdout).strip()[-300:])
    devs = parse_devices(r.stdout)
    if a.serial:
        devs = [d for d in devs if d["serial"] == a.serial]
    events.metric(STAGE, "devices", devs)
    events.check(STAGE, "device_connected", any(d["state"] == "device" for d in devs),
                 value=", ".join(f"{d['serial']} {d['state']}" for d in devs) or
                 "no device — plug in the phone and allow USB debugging")
    for d in devs:
        if d["state"] != "device":
            continue
        events.start(STAGE, "list", serial=d["serial"])
        r = adb(exe, "shell", f"ls -l {CAMERA_DIR}/", serial=d["serial"])
        clips = parse_ls(r.stdout)
        events.metric(STAGE, "clips", clips, serial=d["serial"], model=d.get("model"))
