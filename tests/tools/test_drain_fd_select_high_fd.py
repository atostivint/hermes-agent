"""POSIX stdout drain must keep working when the process holds more than FD_SETSIZE fds.

``select.select()`` refuses any descriptor number >= ``FD_SETSIZE`` (1024 on Linux) with
``ValueError: filedescriptor out of range in select()``. The drain thread used to call it
inside ``try: ... except (ValueError, OSError): return``, so a process whose fd table had
grown past 1024 — any leaked descriptor allocates above the ceiling, and the command's
stdout pipe inherits that range — silently captured **zero** bytes for every command while
still reporting the command's real exit code. The visible symptom is a terminal tool that
answers ``{"output": "", "exit_code": 0}`` for commands that demonstrably ran.

These tests hold more than 1024 descriptors so the pipe lands above the ceiling, then assert
the output still arrives. They fail against a ``select()``-only drain and pass with
``poll()``.
"""

from __future__ import annotations

import codecs
import os
from unittest.mock import MagicMock

import pytest

from tools.environments.base_output import _BoundedOutputCollector, _drain_fd_select

pytestmark = pytest.mark.platforms("linux", "macos")

# One above FD_SETSIZE: the descriptors handed out after this are all unselectable.
_FDS_TO_HOLD = 1100


def _hold_fds(count: int) -> list[int]:
    """Open *count* descriptors, or return what the process limit allowed."""
    held: list[int] = []
    try:
        while len(held) < count:
            held.append(os.open("/dev/null", os.O_RDONLY))
    except OSError:
        pass  # a low RLIMIT_NOFILE: the test below skips instead of failing
    return held


def _close_fds(fds: list[int]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def test_drain_fd_select_captures_output_from_a_pipe_above_fd_setsize():
    """A pipe whose read fd is >= 1024 still yields its data."""
    held = _hold_fds(_FDS_TO_HOLD)
    try:
        if len(held) < _FDS_TO_HOLD:
            pytest.skip(f"cannot hold {_FDS_TO_HOLD} descriptors (RLIMIT_NOFILE), held {len(held)}")
        read_fd, write_fd = os.pipe()
        try:
            assert read_fd >= 1024, "test setup: pipe fd must be above FD_SETSIZE"
            os.write(write_fd, b"high_fd_marker\n")
            proc = MagicMock()
            proc.poll.return_value = 0  # bash already exited: drain stops after the tail
            output = _BoundedOutputCollector(1000)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")

            _drain_fd_select(proc, read_fd, output, decoder)

            assert "high_fd_marker" in output.render(), (
                f"drain lost the output of a pipe at fd {read_fd} "
                f"(select() cannot poll fd >= FD_SETSIZE)")
        finally:
            os.close(read_fd)
            os.close(write_fd)
    finally:
        _close_fds(held)


def test_local_environment_executes_and_captures_with_a_large_fd_table(tmp_path):
    """End to end: ``execute()`` still returns the command's output when the process holds
    more than FD_SETSIZE descriptors before the command runs."""
    from tools.environments.local import LocalEnvironment

    held = _hold_fds(_FDS_TO_HOLD)
    try:
        if len(held) < _FDS_TO_HOLD:
            pytest.skip(f"cannot hold {_FDS_TO_HOLD} descriptors (RLIMIT_NOFILE), held {len(held)}")
        env = LocalEnvironment(cwd=str(tmp_path))
        try:
            marker = "large_fd_table_marker"
            result = env.execute(f"echo {marker}", timeout=30)
        finally:
            env.cleanup()

        assert result["returncode"] == 0
        assert marker in result["output"], (
            f"output lost with {len(held)} descriptors held: {result!r}")
    finally:
        _close_fds(held)
