"""Durable ownership of workers, including descendants of an exited parent."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


class _WindowsJob:
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_longlong * 4),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self.ctypes = ctypes
        self.accounting_type = Accounting
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [
                    wintypes.HANDLE,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                ],
                wintypes.BOOL,
            ),
            "AssignProcessToJobObject": (
                [wintypes.HANDLE, wintypes.HANDLE],
                wintypes.BOOL,
            ),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes = argtypes
            function.restype = restype
        # NULL security attributes produce a non-inheritable handle. Only the
        # controller holds it, so controller death closes the last job handle.
        self.handle = self.kernel.CreateJobObjectW(None, None)
        self._check(self.handle)
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        try:
            self._check(
                self.kernel.SetInformationJobObject(
                    self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
        except BaseException:
            self.close()
            raise

    def _check(self, result):
        if not result:
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def assign(self, process):
        self._check(
            self.kernel.AssignProcessToJobObject(self.handle, int(process._handle))
        )

    def terminate(self):
        self._check(self.kernel.TerminateJobObject(self.handle, 1))

    def active(self):
        info = self.accounting_type()
        self._check(
            self.kernel.QueryInformationJobObject(
                self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None
            )
        )
        return info.ActiveProcesses

    def close(self):
        if self.handle:
            self._check(self.kernel.CloseHandle(self.handle))
            self.handle = None


class ProcessTree:
    def __init__(self, command, *, environment=None):
        self.job = _WindowsJob() if os.name == "nt" else None
        self.process = None
        self.closed = False
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_process_supervisor.py")),
                    *command,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(environment) if environment is not None else None,
                **(
                    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt"
                    else {"start_new_session": True}
                ),
            )
            if self.job is not None:
                self.job.assign(self.process)
            self.process.stdin.write("1")
            self.process.stdin.flush()
        except BaseException:
            if self.process is not None:
                self.process.kill()
                self.process.wait()
                self.process.stdin.close()
            if self.job is not None:
                self.job.close()
            raise

    def _group_members(self):
        members = []
        for process in psutil.process_iter():
            try:
                if (
                    os.getpgid(process.pid) == self.process.pid
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    members.append(process)
            except (psutil.Error, OSError):
                pass
        return members

    def close(self, grace_seconds=3.0):
        if self.closed:
            return
        try:
            if self.job is not None:
                self.job.terminate()
                deadline = time.monotonic() + max(3.0, grace_seconds)
                while self.job.active():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Windows adapter job did not terminate")
                    time.sleep(0.01)
            else:
                members = self._group_members()
                if members:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    _, alive = psutil.wait_procs(members, timeout=max(0, grace_seconds))
                    if alive:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        _, alive = psutil.wait_procs(alive, timeout=3)
                        if any(p.status() != psutil.STATUS_ZOMBIE for p in alive):
                            raise RuntimeError("POSIX adapter group did not terminate")
            self.process.wait(timeout=3)
        finally:
            self.process.stdin.close()
            if self.job is not None:
                self.job.close()
            self.closed = True
