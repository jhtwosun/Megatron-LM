# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private fixed-CP4 repeated-D4 facade tests."""

from types import SimpleNamespace

import pytest

from megatron.core.mdp import dynamic_cp_d4_fixed_facade as api
from megatron.core.mdp.errors import MdpPlanError

_PHASES = (
    "execution",
    "forward",
    "publication",
    "replay",
    "native_schedule",
    "gradient",
    "backward_authorization",
    "backward",
    "finalize",
)


class _Lease:
    def __init__(self, owner, stage):
        self.owner = owner
        self.stage = stage
        self.prior = owner.capability
        self.active = True

    def adopt(self, successor):
        assert self.active
        self.owner.events.append(("adopt", self.stage))
        if self.owner.fail == f"adopt_{self.stage}":
            self.active = False
            self.owner.retired = True
            self.owner.cleaned.append(successor)
            raise RuntimeError(f"adopt {self.stage}")
        self.active = False
        self.owner.capability = successor
        return successor

    def fail(self, primary):
        assert self.active
        self.active = False
        self.owner.events.append(("fail", self.stage, primary))
        self.owner.cleaned.append(self.prior)
        self.owner.retired = True

    def arguments(self):
        assert self.active and self.stage == "commit"
        self.owner.events.append("arguments")
        if self.owner.fail == "arguments":
            raise RuntimeError("arguments")
        return self.owner.binding, self.owner.authority, self.prior.owner, self.prior

    def complete(self, result):
        assert self.active and self.stage == "commit"
        self.owner.events.append(("complete", result))
        self.active = False
        if self.owner.fail == "complete":
            self.owner.cleaned.append(self.prior)
            self.owner.retired = True
            raise RuntimeError("complete")
        assert result is None
        self.owner.retired = True


class _Transaction:
    def __init__(self, capture, fail=None):
        self.capture = capture
        self.binding = capture.binding
        self.fail = fail
        self.events = []
        self.cleaned = []
        self.retired = False
        self.capability = capture
        self.authority = None

    def attach_source_catalog(self, projection):
        self.events.append("attach_catalog")
        if self.fail == "catalog":
            raise RuntimeError("catalog")
        self.projection = projection

    def attach_authority(self, authority):
        self.events.append("attach_authority")
        if self.fail == "authority":
            raise RuntimeError("authority")
        self.authority = authority

    def abort(self, primary=None):
        assert not self.retired
        self.retired = True
        self.events.append(("abort", primary))
        self.cleaned.append(self.capability)

    def _begin(self, stage):
        self.events.append(("begin", stage))
        if self.fail == f"begin_{stage}":
            raise RuntimeError(f"begin {stage}")
        return _Lease(self, stage)


for _stage in (*_PHASES, "commit"):
    setattr(_Transaction, f"begin_{_stage}", lambda self, stage=_stage: self._begin(stage))


def _install(monkeypatch, *, fail=None, role="source"):
    events = []
    transactions = []
    binding = object()
    capture = SimpleNamespace(binding=binding, role=role)
    projection = SimpleNamespace(metadata=object())
    authority = object()
    objects = {name: SimpleNamespace(name=name, role=role) for name in _PHASES}
    objects["native_schedule"].completion = object()
    objects["finalize"].owner = SimpleNamespace(name="gate6_owner")
    dependencies = SimpleNamespace(
        runtime=object(),
        decoder_solver=object(),
        bridge_dtype=object(),
        workload_query=lambda *_args, **_kwargs: 1,
        rebuild=lambda *_args, **_kwargs: None,
        forward_backward=lambda **_kwargs: None,
        iterator=object(),
        all_to_all=lambda *_args, **_kwargs: None,
        broadcast=lambda *_args, **_kwargs: None,
        byte_generator=lambda _size: object(),
    )

    def begin(value):
        transaction = _Transaction(value, fail)
        transactions.append(transaction)
        events.append("transaction")
        return transaction

    monkeypatch.setattr(api._transaction, "_begin_d4_transaction", begin)

    def gather(value, actual_binding):
        assert value is capture and actual_binding is binding
        events.append("metadata")
        if fail == "metadata":
            raise RuntimeError("metadata")
        return projection

    monkeypatch.setattr(api, "_gather_d4_source_catalog", gather)

    def build(actual_binding, metadata, **kwargs):
        assert actual_binding is binding and metadata is projection.metadata
        assert kwargs == {
            "decoder_max_seqlen_per_rank": 8,
            "decoder_minimum_cp_size": 4,
            "decoder_solver": dependencies.decoder_solver,
            "encoder_max_seqlen_per_rank": 8,
            "encoder_minimum_cp_size": 1,
            "encoder_workload_query": dependencies.workload_query,
            "bridge_width": 3,
            "bridge_dtype": dependencies.bridge_dtype,
        }
        events.append("authority")
        if fail == "build_authority":
            raise RuntimeError("build authority")
        return authority

    monkeypatch.setattr(api, "build_repeated_d4_joint_iteration_authority", build)

    def fixed(actual_authority, actual_binding):
        assert actual_authority is authority and actual_binding is binding
        events.append("fixed_cp4")
        if fail == "fixed_cp4":
            raise MdpPlanError("fixed CP4")
        return ((object(), object()),)

    monkeypatch.setattr(api._replay, "_fixed_assignments", fixed)

    def check(stage, args, kwargs):
        completion = objects["native_schedule"].completion
        common = {"byte_generator": dependencies.byte_generator}
        expected = {
            "execution": ((capture, authority), {}),
            "forward": (
                (dependencies.runtime, objects["execution"], authority),
                {
                    "all_to_all_single": dependencies.all_to_all,
                    "broadcast": dependencies.broadcast,
                    **common,
                },
            ),
            "publication": (
                (objects["forward"],),
                {"all_to_all_single": dependencies.all_to_all, **common},
            ),
            "replay": (
                (objects["publication"], authority),
                {
                    "rebuild_microbatch": dependencies.rebuild,
                    "cp_partition_mode": "fixed",
                    **common,
                },
            ),
            "native_schedule": (
                (dependencies.forward_backward, objects["replay"], api._scheduled_abort),
                {"data_iterator": dependencies.iterator, "forward_only": False},
            ),
            "gradient": (
                (objects["replay"], authority, completion),
                {"all_to_all_single": dependencies.all_to_all, **common},
            ),
            "backward_authorization": ((objects["gradient"], authority, completion), common),
            "backward": ((objects["backward_authorization"], authority, completion), common),
            "finalize": ((objects["backward"], authority, completion), common),
        }[stage]
        assert args == expected[0]
        assert kwargs == expected[1]
        predecessor = (
            args[0] if stage == "execution" else args[1] if stage == "forward" else args[0]
        )
        assert (args[1] if stage == "native_schedule" else predecessor).role == role

    calls = (
        (api._execution, "claim_d4_encoder_execution", "execution"),
        (api._forward, "run_repeated_d4_encoder_forward", "forward"),
        (api._forward, "run_repeated_d4_encoder_publication", "publication"),
        (api._replay, "_run_repeated_d4_fixed_decoder_replay_from_publication", "replay"),
        (api._native, "_run_d4_native_schedule", "native_schedule"),
        (api._gradient, "_run_repeated_d4_encoder_gradient_from_replay", "gradient"),
        (api._gate4, "run_repeated_d4_encoder_backward_authorization", "backward_authorization"),
        (api._gate5, "run_repeated_d4_encoder_selected_backward", "backward"),
        (api._gate6, "run_repeated_d4_encoder_gradient_finalize", "finalize"),
    )
    for module, name, stage in calls:

        def adapter(*args, _stage=stage, **kwargs):
            check(_stage, args, kwargs)
            events.append(_stage)
            if fail == _stage:
                raise RuntimeError(_stage)
            return objects[_stage]

        monkeypatch.setattr(module, name, adapter)

    def commit(*args, **kwargs):
        assert args == (binding, authority, objects["finalize"].owner, objects["finalize"])
        assert kwargs == {"byte_generator": dependencies.byte_generator}
        events.append("commit")
        if fail == "commit":
            raise RuntimeError("commit")
        return None

    monkeypatch.setattr(api._gate7, "run_repeated_d4_encoder_iteration_commit", commit)
    return SimpleNamespace(
        capture=capture,
        events=events,
        transactions=transactions,
        objects=objects,
        dependencies=dependencies,
    )


def _run(parts):
    return api._run_repeated_d4_fixed_iteration(
        parts.dependencies.runtime,
        parts.capture,
        decoder_max_seqlen_per_rank=8,
        decoder_solver=parts.dependencies.decoder_solver,
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=parts.dependencies.workload_query,
        bridge_width=3,
        bridge_dtype=parts.dependencies.bridge_dtype,
        rebuild_microbatch=parts.dependencies.rebuild,
        cp_partition_mode="fixed",
        forward_backward_func=parts.dependencies.forward_backward,
        native_schedule_kwargs={
            "data_iterator": parts.dependencies.iterator,
            "forward_only": False,
        },
        all_to_all_single=parts.dependencies.all_to_all,
        broadcast=parts.dependencies.broadcast,
        byte_generator=parts.dependencies.byte_generator,
    )


@pytest.mark.parametrize("role", ("source", "follower", "nonmember", "text"))
def test_fixed_facade_sequences_one_metadata_protocol_and_gates_zero_through_seven(
    monkeypatch, role
):
    parts = _install(monkeypatch, role=role)

    assert _run(parts) is None

    assert parts.events == [
        "transaction",
        "metadata",
        "authority",
        "fixed_cp4",
        "execution",
        "forward",
        "publication",
        "replay",
        "native_schedule",
        "gradient",
        "backward_authorization",
        "backward",
        "finalize",
        "commit",
    ]
    transaction = parts.transactions[0]
    assert [event for event in transaction.events if isinstance(event, tuple)] == [
        *(item for stage in _PHASES for item in (("begin", stage), ("adopt", stage))),
        ("begin", "commit"),
        ("complete", None),
    ]
    assert transaction.events.count("attach_catalog") == 1
    assert transaction.events.count("attach_authority") == 1
    assert transaction.events.count("arguments") == 1
    assert transaction.retired and transaction.cleaned == []


def test_non_cp4_plan_rejects_before_gate0_and_cleans_capture(monkeypatch):
    parts = _install(monkeypatch, fail="fixed_cp4")

    with pytest.raises(MdpPlanError, match="^fixed CP4$"):
        _run(parts)

    assert parts.events == ["transaction", "metadata", "authority", "fixed_cp4"]
    transaction = parts.transactions[0]
    assert transaction.retired and transaction.cleaned == [parts.capture]


def test_native_scheduled_abort_delegates_exact_owner_and_primary():
    calls = []
    owner = SimpleNamespace(abort=lambda primary: calls.append(primary))
    primary = RuntimeError("native")

    assert api._scheduled_abort(owner, primary) is None

    assert calls == [primary]


@pytest.mark.parametrize(
    "failure",
    (
        "metadata",
        "catalog",
        "build_authority",
        "authority",
        *(_stage for phase in _PHASES for _stage in (f"begin_{phase}", phase, f"adopt_{phase}")),
        "begin_commit",
        "arguments",
        "commit",
        "complete",
    ),
)
def test_every_boundary_failure_is_terminal_without_later_commit_and_fresh_run_succeeds(
    monkeypatch, failure
):
    failed = _install(monkeypatch, fail=failure)

    with pytest.raises(RuntimeError):
        _run(failed)

    assert failed.transactions[0].retired
    if failure != "complete":
        assert "commit" not in failed.events or failure == "commit"

    fresh = _install(monkeypatch)
    assert _run(fresh) is None
    assert fresh.events[-1] == "commit"
