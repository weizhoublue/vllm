# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import subprocess
import sys
import textwrap

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    assert 1 + 1 == 2


@pytest.mark.xfail(raises=RuntimeError, strict=True)
@spawn_new_process_for_each_test
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def test_spawn_decorator_skip():
    """pytest.skip inside subprocess should propagate correctly."""
    pytest.skip("intentional skip")


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected


def test_spawn_decorator_sets_spawn_start_method_for_parent_process():
    """The parent pytest process should use spawn after calling the wrapper."""
    script = textwrap.dedent(
        """
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch.multiprocessing as mp

        import tests.utils
        from tests.utils import spawn_new_process_for_each_test

        def test_func():
            pass

        decorated = spawn_new_process_for_each_test(test_func)

        with patch.object(tests.utils.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0)):
            decorated()

        print(mp.get_start_method(allow_none=True))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "spawn"
