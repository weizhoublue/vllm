# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that delayed imports in the AITER shared expert router resolve
to valid module paths.  PR #41979 restructured the fused_moe directory
but missed two import paths in this router file."""

import importlib
from pathlib import Path

import pytest

ROUTER_FILE = (
    Path(__file__).parents[3]
    / "vllm/model_executor/layers/fused_moe/router/"
    "aiter_shared_routed_fused_moe_router.py"
)


def test_router_class_importable():
    """The router class itself should be importable."""
    mod = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.router."
        "aiter_shared_routed_fused_moe_router"
    )
    assert hasattr(mod, "AiterSharedRoutedFusedMoERouter")


@pytest.mark.parametrize(
    "module_name, attr_name",
    [
        ("vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
         "aiter_topK_meta_data"),
        ("vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
         "inject_shared_expert_weights"),
    ],
)
def test_delayed_import_paths_resolve(module_name: str, attr_name: str):
    """All delayed imports inside _compute_routing must point to modules
    that exist and export the expected symbols."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, attr_name), (
        f"{module_name} does not export {attr_name}"
    )


def test_no_stale_import_paths_in_router():
    """Sanity-check: the router source must not reference the old
    (now-removed) module path."""
    text = ROUTER_FILE.read_text()
    assert "rocm_aiter_fused_moe" not in text, (
        "Router file still contains stale imports referencing "
        "rocm_aiter_fused_moe (the file was moved to experts/rocm_aiter_moe.py)"
    )
