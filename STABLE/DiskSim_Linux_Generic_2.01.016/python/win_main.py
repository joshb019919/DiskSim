#!/usr/bin/env python3
"""Windows wrapper for main.py (sweeps) and examples/dqn_smoke.py (single run).

DiskSim is loaded through src/libdsim.so (a Linux shared library), so on Windows the scripts
must run inside WSL. This wrapper translates the working directory and forwards all arguments.
On Linux/macOS it just runs the scripts directly.

  python win_main.py sweep --dry-run                 # runs main.py
  python win_main.py sweep --episodes 1000 --jobs 4
  python win_main.py smoke 200 --seed 1              # runs examples/dqn_smoke.py
  python win_main.py --distro Ubuntu sweep --resume  # pick a WSL distro

Environment: WSL_PYTHON (default python3) selects the interpreter inside WSL.
"""
import os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"sweep": "main.py", "main": "main.py", "smoke": os.path.join("examples", "dqn_smoke.py"),
           "dqn": os.path.join("examples", "dqn_smoke.py")}


def usage():
    sys.exit(__doc__)


def main():
    argv = sys.argv[1:]
    distro = None
    if argv[:1] == ["--distro"]:
        if len(argv) < 3: usage()
        distro, argv = argv[1], argv[2:]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in SCRIPTS:
        usage()
    script, rest = SCRIPTS[argv[0]].replace(os.sep, "/"), argv[1:]
    py = os.environ.get("WSL_PYTHON", "python3")

    if os.name != "nt":
        return subprocess.call([sys.executable, os.path.join(HERE, script), *rest], cwd=HERE)

    # Windows: hop into WSL, converting this folder to its /mnt/... path.
    wsl = ["wsl"] + (["-d", distro] if distro else [])
    try:
        linux_dir = subprocess.check_output(wsl + ["wslpath", "-a", HERE], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"WSL is required (libdsim.so is Linux-only) but could not be used: {e}")
    cmd = f"cd {shlex.quote(linux_dir)} && exec {shlex.quote(py)} {shlex.quote(script)} " + " ".join(map(shlex.quote, rest))
    try:
        return subprocess.call(wsl + ["bash", "-lc", cmd])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
