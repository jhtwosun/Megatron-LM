# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Import-boundary contracts for model-owned Nemotron Omni MDP hooks."""

import subprocess
import sys


def test_core_mdp_import_never_loads_nemotron_model_modules():
    script = r"""
import sys

prefix = "examples.multimodal_dev.models.nemotron_omni"
assert not any(name.startswith(prefix) for name in sys.modules)
import megatron.core.mdp  # noqa: F401
loaded = sorted(name for name in sys.modules if name.startswith(prefix))
assert loaded == [], loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_registry_keeps_nemotron_hooks_lazy_until_dispatch():
    script = r"""
import sys
from types import SimpleNamespace

prefix = "examples.multimodal_dev.models.nemotron_omni"
from examples.multimodal_dev.models import MODEL_REGISTRY

assert not any(name.startswith(prefix) for name in sys.modules)
entry = MODEL_REGISTRY["nemotron_omni"]
assert entry["dataset_providers"]
factory = entry["mdp_adapter_factory"]
assert isinstance(factory, str)
assert not any(name.startswith(prefix) for name in sys.modules)

module_name, attribute = factory.rsplit(".", 1)
resolved = getattr(__import__(module_name, fromlist=[attribute]), attribute)
assert callable(resolved)
assert any(name.startswith(prefix) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
