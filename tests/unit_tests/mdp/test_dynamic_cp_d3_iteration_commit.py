# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the post-Gate-6 D3 runtime commit capability."""

from importlib import import_module

import pytest
import torch

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.runtime import MdpRuntimeState
from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _runtime


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_iteration_commit")


def _finalized_runtime():
    runtime, _ = _runtime(contributor=False)
    runtime._chunk_payload_bases = ()
    runtime._captured_num_tokens = torch.ones((), device=runtime.device)
    runtime._token_capture_count = 1
    runtime._token_consumed = True
    return runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA token authority")
def test_commit_clears_success_state_and_advances_exactly_once():
    api = _api()
    runtime = _finalized_runtime()
    token = runtime._captured_num_tokens
    ready = api._mint_d3_iteration_commit_ready(runtime, token, 0)

    api._execute_d3_iteration_commit(ready)

    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._iteration == 1
    assert runtime._captured_num_tokens is None
    assert runtime._token_capture_count == 0 and runtime._token_consumed is False
    assert ready.runtime is ready.token is None and ready.token_authority == ()
    with pytest.raises(MdpStateError, match="finalized authority"):
        api._execute_d3_iteration_commit(ready)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA token authority")
def test_same_runtime_commits_two_successive_iterations():
    api = _api()
    runtime = _finalized_runtime()

    for iteration in range(2):
        if iteration:
            runtime._captured_num_tokens = torch.ones((), device=runtime.device)
            runtime._token_capture_count = 1
            runtime._token_consumed = True
        ready = api._mint_d3_iteration_commit_ready(
            runtime, runtime._captured_num_tokens, iteration
        )
        api._execute_d3_iteration_commit(ready)

    assert runtime._iteration == 2
    assert runtime.state is MdpRuntimeState.EMPTY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA token authority")
def test_commit_rejects_iteration_or_token_mutation_without_partial_reset():
    api = _api()
    runtime = _finalized_runtime()
    token = runtime._captured_num_tokens
    ready = api._mint_d3_iteration_commit_ready(runtime, token, 0)
    runtime._iteration = 1

    with pytest.raises(MdpStateError, match="finalized authority"):
        api._execute_d3_iteration_commit(ready)

    assert runtime._captured_num_tokens is token
    assert runtime._token_consumed is True
    assert ready.runtime is runtime


def test_commit_capability_rejects_forged_inputs():
    api = _api()
    with pytest.raises(MdpStateError, match="factory"):
        api._D3IterationCommitReady(None, None, 0, ())
    with pytest.raises(MdpConfigurationError, match="exact capability"):
        api._execute_d3_iteration_commit(object())
