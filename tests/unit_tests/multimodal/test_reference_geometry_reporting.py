"""Allocated reference-only geometry reporting; no change to backward normalization."""

import ast
from pathlib import Path

import torch

from examples.multimodal_dev.data.blend_dataset import Qwen35VLDataset
from examples.multimodal_dev.forward_step import loss_func


def test_geometry_uses_original_content_and_global_grids():
    docs = [dict(content_len=1028, real_len=1088, reference_labels=torch.zeros(1028),
                 image_grid_thw=torch.tensor([[2, 32, 16], [1, 16, 16]]),
                 pixel_values=torch.empty(0, 1536)),
            dict(content_len=513, real_len=576, reference_labels=torch.zeros(513),
                 image_grid_thw=torch.empty(0, 3, dtype=torch.long))]
    out = {}
    Qwen35VLDataset._reference_geometry(out, docs)
    actual = out["_reference_flops_geometry"]
    assert actual.dtype == torch.int64
    assert actual.tolist() == [1541, 1028**2 + 513**2, 1280, 2 * 512**2 + 256**2]
    plain = {}
    Qwen35VLDataset._reference_geometry(plain, [dict(content_len=1)])
    assert "_reference_flops_geometry" not in plain


def test_reporting_independent_of_supervised_tokens_and_backward():
    geometry = torch.tensor([100, 2**25 + 3, 800, 2**40 + 7], dtype=torch.int64, device="cuda")
    outputs = torch.arange(4, dtype=torch.float32, device="cuda", requires_grad=True)
    mask = torch.tensor([1., 0., 1., 0.], device="cuda")
    loss, tokens, reporting = loss_func(mask, outputs, geometry)
    base_loss, base_tokens, base_reporting = loss_func(mask, outputs)
    assert torch.equal(loss, base_loss) and torch.equal(tokens, base_tokens)
    assert torch.equal(reporting["lm loss"], base_reporting["lm loss"])
    grad = torch.autograd.grad(loss, outputs, retain_graph=True)[0]
    assert torch.equal(grad, torch.autograd.grad(base_loss, outputs)[0])
    for i, key in enumerate(("T", "U", "R", "A")):
        pair = reporting["ref_geom_" + key]
        assert pair.dtype == torch.int64
        assert pair.tolist() == [geometry[i].item(), 1]
        # Two unequal bins, each replicated twice by CP: count denominator, not token denominator.
        other = pair.clone()
        other[0] += 10
        aggregate = torch.stack([pair, pair, other, other]).sum(0)
        average = aggregate[0].double() / aggregate[1].double()
        assert aggregate[1].item() == 4
        assert average.item() * 2 == pair[0].item() + other[0].item()


def test_actual_logger_accumulator_and_reset_preserve_float64():
    # Execute the actual narrow logger arithmetic blocks, without invoking its unrelated I/O.
    path = Path(__file__).resolve().parents[3] / "megatron/training/training.py"
    tree = ast.parse(path.read_text())
    logger = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "training_log")
    accumulation = next(node for node in logger.body if isinstance(node, ast.For)
                        and isinstance(node.iter, ast.Name) and node.iter.id == "loss_dict")
    reset = next(node for node in ast.walk(logger) if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == "should_reset"
                 and any(isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id == "dtype" for n in node.body))
    namespace = dict(torch=torch, skipped_iter=0, got_nan=False, total_loss_dict={},
                     loss_dict={"ref_geom_A": torch.tensor(2**40 + 7, dtype=torch.float64, device="cuda")})
    code = compile(ast.Module(body=[accumulation], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    assert namespace["total_loss_dict"]["ref_geom_A"].item() == 2**40 + 7
    namespace.update(key="ref_geom_A", should_reset=True)
    exec(compile(ast.Module(body=[reset], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["total_loss_dict"]["ref_geom_A"].dtype == torch.float64
    exec(code, namespace)
    assert namespace["total_loss_dict"]["ref_geom_A"].item() == 2**40 + 7
