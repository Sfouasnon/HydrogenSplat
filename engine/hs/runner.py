"""Run a child process, stream its output line by line, tee it verbatim to a log file.

Every computing step in the engine is a subprocess (strategy §2): the six vendored scripts
run under the same interpreter as ``hs`` (``sys.executable``), Brush and ffmpeg run as the
binaries they are. Output is split on ``\\r`` as well as ``\\n`` because Brush's indicatif
spinner redraws with carriage returns when it has a TTY. Both stdout and stderr of the
child are merged into one stream — pycolmap logs through glog on stderr and prints on
stdout, and the order between them is what a human reading ``logs/<stage>.log`` wants.
"""
import os
import subprocess
import sys
import threading
import time

from . import events

VENDORED = os.path.dirname(os.path.abspath(__file__))


def script(name):
    """Absolute path of a vendored script, e.g. script("select_frames.py")."""
    p = os.path.join(VENDORED, name)
    if not os.path.exists(p):
        raise events.StageError(f"vendored script missing: {p}")
    return p


def python_argv(script_name, *args):
    return [sys.executable, script(script_name)] + [str(a) for a in args]


class ChildResult:
    def __init__(self):
        self.returncode = None
        self.lines = []          # every line, after \r/\n splitting, stripped of the newline
        self.started = None
        self.finished = None
        self.pid = None

    @property
    def elapsed(self):
        if self.started is None:
            return 0.0
        return (self.finished or time.monotonic()) - self.started

    def text(self):
        return "\n".join(self.lines)


def run(argv, stage, log_path=None, on_line=None, cwd=None, env=None, tick=None,
        tick_interval=2.0, pid_file=None, check=True):
    """Run argv; call on_line(line) for each output line and tick() every tick_interval
    seconds while the child runs (used to poll export folders). Returns ChildResult.

    Raises StageError on non-zero exit when check=True, with the last lines as the hint.
    """
    res = ChildResult()
    res.started = time.monotonic()
    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8", errors="replace")
        log_f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')}  $ {' '.join(argv)}\n")
        log_f.flush()

    full_env = dict(os.environ)
    full_env["PYTHONUNBUFFERED"] = "1"
    if env:
        full_env.update(env)

    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=cwd, env=full_env, bufsize=0)
    except FileNotFoundError as e:
        if log_f:
            log_f.write(f"!! could not start: {e}\n")
            log_f.close()
        raise events.StageError(f"could not start {argv[0]}: {e}",
                                hint="check the tool path (hs tools)")
    res.pid = proc.pid
    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))

    stop = threading.Event()

    def reader():
        buf = b""
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # split on \r and \n; keep the trailing partial line in buf
            while True:
                i_n = buf.find(b"\n")
                i_r = buf.find(b"\r")
                cands = [i for i in (i_n, i_r) if i >= 0]
                if not cands:
                    break
                i = min(cands)
                raw, buf = buf[:i], buf[i + 1:]
                _handle(raw)
        if buf:
            _handle(buf)

    def _handle(raw):
        line = raw.decode("utf-8", "replace").rstrip()
        if log_f:
            log_f.write(line + "\n")
            log_f.flush()
        if not line:
            return
        res.lines.append(line)
        events.log(stage, line)
        if on_line:
            try:
                on_line(line)
            except Exception as e:  # a parser bug must never kill a 55-minute run
                events.log(stage, f"[hs] on_line error: {e!r}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while True:
            t.join(timeout=tick_interval if tick else None)
            if tick:
                try:
                    tick()
                except Exception as e:
                    events.log(stage, f"[hs] tick error: {e!r}")
            if not t.is_alive():
                break
        res.returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        stop.set()
        res.finished = time.monotonic()
        if pid_file and os.path.exists(pid_file):
            os.remove(pid_file)
        if log_f:
            log_f.write(f"### exit {res.returncode} after {res.elapsed:.1f}s\n")
            log_f.close()

    if check and res.returncode != 0:
        tail = [l for l in res.lines[-8:] if not l.startswith("I2")]  # skip glog info spam
        raise events.StageError(f"{os.path.basename(argv[0])} exited {res.returncode}",
                                hint=" | ".join(tail[-4:]) if tail else None)
    return res


def which(name):
    """shutil.which with ~ expansion; returns None when absent."""
    import shutil
    p = os.path.expanduser(name)
    if os.path.sep in p:
        return p if os.path.isfile(p) and os.access(p, os.X_OK) else None
    return shutil.which(p)
