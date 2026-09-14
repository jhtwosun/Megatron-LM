"""Opt-in reference5885 sample content through the legacy descriptor protocol.

Scenario generator copied verbatim from BestJuly/Megatron-LM
5885d65a9c91d867d38c38cf00048014fd66bf04, examples/multimodal_dev/data/mdp_scenarios.py.
Token/pixel recipe follows the same commit's mdp_mock.py:128–179.
Legacy packing and MDP ownership are intentionally not replaced.
"""

import json

import torch

from examples.multimodal_dev.data.dataset_utils import RawSample
from examples.multimodal_dev.data.reference_mdp_scenarios import build_scenarios
from megatron.training import get_args


class ReferenceMock:
    def __init__(self, config: str, split: str):
        args = get_args()
        self.vocab_size = args.padded_vocab_size
        self.image_token_id = args.image_token_id
        self.start_token_id = 248053
        self.seed = 1234 + {"train": 0, "val": 1, "valid": 1, "test": 2}[split]
        self.scenarios = build_scenarios(length_config=json.loads(config))

    def __getitem__(self, idx: int) -> RawSample:
        grids, text_chunks = self.scenarios[idx % len(self.scenarios)]
        generator = torch.Generator(device="cpu").manual_seed(self.seed + idx)

        def text(length):
            tokens = torch.randint(1, self.vocab_size, (length,), generator=generator,
                                   dtype=torch.long, device="cpu")
            for special in (self.image_token_id, self.start_token_id):
                tokens[tokens == special] = 1
            return tokens

        pieces = [text(text_chunks[0])]
        descriptors = []
        for ordinal, (t, h, w) in enumerate(grids):
            pieces.extend([
                torch.tensor([self.start_token_id], dtype=torch.long, device="cpu"),
                torch.full((t * (h // 2) * (w // 2),), self.image_token_id,
                           dtype=torch.long, device="cpu"),
                text(text_chunks[ordinal + 1]),
            ])
            descriptors.append({
                "kind": "reference_sentinel", "grid_thw": [t, h, w],
                "materializer": "examples.multimodal_dev.data.reference_mock",
                "sentinel": 1000 * (idx + 1) + ordinal,
            })
        ids = torch.cat(pieces)
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1] = 0
        mask = (ids != self.image_token_id).float()
        mask[ids == self.start_token_id] = 0
        mask[-1] = 0
        return RawSample(images=[], text="", image_descriptors=descriptors,
                         reference_input_ids=ids, reference_labels=labels,
                         reference_loss_mask=mask)


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    """Materialize only the descriptor-owned constant reference patch rows."""
    if desc["kind"] != "reference_sentinel" or list(grid_thw) != desc["grid_thw"]:
        raise ValueError("reference descriptor/grid mismatch")
    if pixel_dim != 1536 or patch_size != 16:
        raise ValueError("reference fixture requires native 16x16x2 RGB patches")
    t, h, w = grid_thw
    return torch.full((t * h * w, pixel_dim), float(desc["sentinel"]),
                      dtype=torch.float32, device="cpu")
