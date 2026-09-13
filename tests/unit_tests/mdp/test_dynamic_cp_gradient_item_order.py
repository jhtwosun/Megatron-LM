"""Regression: decoder assignment order is not global manifest order."""
from types import MappingProxyType, SimpleNamespace
import pytest
import torch
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_execution import DecoderMicrobatchKey
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

def case(order=(2, 0, 1), *, manifest=(0, 1, 2, 3), local=(1, 2, 0)):
    ids = {i: GlobalVisionItemId(0, i) for i in range(5)}
    records, leaves, expected = [], {}, {}
    for mb, group in enumerate((order[:2], order[2:])):
        items, rows, offset = [], [], 0
        for i in group:
            n = 2 if i == 0 else 1
            items.append(MdpMicrobatchVisionRecord(ids[i], 0, i, (1, 1, n), n, tuple(range(offset, offset+n))))
            rows.extend([[10.0*i+j+1]*2 for j in range(n)])
            offset += n
        leaf = torch.ones((offset, 2), requires_grad=True)
        (leaf * torch.tensor(rows)).sum().backward()
        offset = 0
        for item in items:
            expected[item.global_item_id] = leaf.grad.narrow(0, offset, item.output_rows)
            offset += item.output_rows
        leaves[DecoderMicrobatchKey(mb)] = leaf
        records.append(MdpMicrobatchRecord(mb, False, tuple(items), object(), MappingProxyType({})))
    keys = tuple(DynamicBridgeKey(ids[i], 1) for i in local)
    authority = SimpleNamespace(bridge_dtype=torch.float32, bridge_width=2,
        global_manifest=SimpleNamespace(items=tuple(SimpleNamespace(item_id=ids[i]) for i in manifest)),
        gradient_ledger=SimpleNamespace(entries=tuple(SimpleNamespace(key=k, src_global_rank=1) for k in keys)))
    return dict(records=tuple(records), embedding_leaves=MappingProxyType(leaves), authority=authority, global_rank=1), keys, expected

@pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0), (0, 1, 2)])
def test_replay_order_keeps_exact_id_views_and_ledger_order(order):
    kwargs, keys, expected = case(order)
    projected = api._project_leaf_gradients(**kwargs)
    assert tuple(projected) == keys
    for key in keys:
        actual, reference = projected[key], expected[key.item_id]
        assert torch.equal(actual, reference)
        assert actual.untyped_storage().data_ptr() == reference.untyped_storage().data_ptr()
        assert actual.storage_offset() == reference.storage_offset()
        assert actual.stride() == reference.stride()

def test_valid_manifest_subset_does_not_require_unowned_item():
    kwargs, keys, _ = case((0, 1, 2), local=(1,))
    assert tuple(api._project_leaf_gradients(**kwargs)) == keys

@pytest.mark.parametrize("kwargs,message", [
    (dict(order=(0, 0, 1), local=(0, 1)), "visits every manifest item once"),
    (dict(order=(0, 1, 4), local=(0, 1)), "manifest"),
    (dict(order=(0, 1, 2), local=(0, 3)), "cover exact local route authority"),
])
def test_invalid_id_coverage_rejected(kwargs, message):
    args, _, _ = case(**kwargs)
    with pytest.raises(MdpPlanError, match=message):
        api._project_leaf_gradients(**args)
