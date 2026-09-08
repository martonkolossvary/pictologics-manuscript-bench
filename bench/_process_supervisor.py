"""Stdlib-only adapter launcher; stdin is the controller's lifetime lease.

The initial byte gates launch until Windows Job assignment has completed.
On POSIX, EOF kills the isolated process group even if the controller was
forcibly killed. Never pass this private pipe to the adapter or its children.
"""

import os
import signal
import subprocess
import sys


def main() -> int:
    if sys.stdin.buffer.read(1) != b"1":
        return 1

    if os.name != "nt" and os.fork() == 0:
        # A separate watcher outlives the supervisor when the adapter exits
        # first. It also survives long enough to kill TERM-resistant children.
        def kill_group(*_args):
            os.killpg(os.getpgrp(), signal.SIGKILL)

        signal.signal(signal.SIGTERM, kill_group)
        os.close(1)
        os.close(2)
        sys.stdin.buffer.read()
        kill_group()
    child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL)
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
