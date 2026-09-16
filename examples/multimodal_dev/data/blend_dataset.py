# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Multi-image Qwen-VL blend dataset adapter.

The loader uses Megatron's tokenizer for text, inserts Qwen-VL image marker
spans, and pads packed document segments to the MIMO 64-token granularity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from examples.multimodal_dev.balance_data import (
    balance_per_image_lpt_by_flops,
    lpt_flops_from_grid_rows,
)
from examples.multimodal_dev.data.blend import RawDatasetBlend
from examples.multimodal_dev.data.dataset_utils import RawSample, preprocess_image_to_patches
from examples.multimodal_dev.data.energon_vision_balance import vision_rows_from_grid
from examples.multimodal_dev.mdp_image_materialize import (
    encode_image_descriptors,
    materialize_descriptor,
)
from examples.multimodal_dev.mdp_parallel_groups import (
    compute_pp_cp_inner_dp_layout,
    find_pp_cp_inner_dp_group_for_rank,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    QWEN35_VL_VOCAB_SIZE,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from megatron.core import parallel_state
from megatron.training import get_args, get_tokenizer

# Backward-compatible private alias used by older focused tests.
_RawSample = RawSample

_MIMO_PACK_PAD_MULTIPLE = 64


# ---------------------------------------------------------------------------
# The Dataset: drop-in for MockQwen35VLDataset.
# ---------------------------------------------------------------------------

class Qwen35VLDataset(Dataset):
    """Build Qwen-VL training samples from multimodal backend records."""

    def __init__(
        self,
        backend: str | Sequence[str] = "mantis-instruct",
        root: str | dict | None = None,
        seq_length: int = 4096,
        vocab_size: int = QWEN35_VL_VOCAB_SIZE,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        image_size_max: int = 0,
        image_max_pixels: int = 0,
        image_min_pixels: int = 0,
        max_samples: Optional[int] = None,
        cp_size: int = 1,
        emit_cu_seqlens: bool = True,
        pack_samples_per_item: int = 1,
        pack_scan_multiplier: int = 1,
        dataloader_sequence_packing: bool = False,
        dataloader_dp_rank: int = 0,
        metadata_only_batch: bool = True,
        mdp_loader_prepartition: bool = False,
        mdp_loader_prepartition_rank: int = 0,
        mdp_loader_prepartition_world: int = 1,
        mdp_loader_prepartition_encoder_stage: bool = True,
        mdp_loader_prepartition_hidden: int = 1280,
        tokenizer=None,
        seed: int = 0,
        subsets: Optional[Sequence[str]] = None,
        split: str = "train",
        thd_static_packing: bool = False,
        thd_max_packed_sequences: int = 32,
        max_seqlen_per_dp_cp_rank: Optional[int] = None,
    ):
        self.seq_length = int(seq_length)
        self.thd_static_packing = bool(thd_static_packing)
        self.thd_max_packed_sequences = int(thd_max_packed_sequences)
        if self.thd_static_packing:
            if self.thd_max_packed_sequences < 2:
                raise ValueError("static THD requires a real-document and tail slot")
            if not max_seqlen_per_dp_cp_rank or max_seqlen_per_dp_cp_rank * cp_size != self.seq_length:
                raise ValueError("static THD local/CP/global token budgets disagree")
        self.vocab_size = int(vocab_size)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.image_size_max = int(image_size_max)
        self.image_max_pixels = int(image_max_pixels)
        self.image_min_pixels = int(image_min_pixels)
        self.cp_size = int(cp_size)
        self.emit_cu_seqlens = bool(emit_cu_seqlens)
        self.align = max(2 * self.cp_size, _MIMO_PACK_PAD_MULTIPLE)
        self.pack_samples_per_item = max(1, int(pack_samples_per_item))
        self.pack_scan_multiplier = max(1, int(pack_scan_multiplier))
        self._pack_start_stride = (
            self.pack_samples_per_item * self.pack_scan_multiplier
        )
        self._pack_scan_span = max(
            self._pack_start_stride,
            (self.seq_length + self.align - 1) // self.align,
        )
        self.dataloader_sequence_packing = bool(
            dataloader_sequence_packing
        )
        self.dataloader_dp_rank = max(0, int(dataloader_dp_rank))

        # The image-side patch dimension required by the encoder.
        self._pixel_dim = (
            3 * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        self.metadata_only_batch = bool(metadata_only_batch)
        self.mdp_loader_prepartition = bool(mdp_loader_prepartition)
        self.mdp_loader_prepartition_rank = int(mdp_loader_prepartition_rank)
        self.mdp_loader_prepartition_world = max(1, int(mdp_loader_prepartition_world))
        self.mdp_loader_prepartition_encoder_stage = bool(
            mdp_loader_prepartition_encoder_stage
        )
        self.mdp_loader_prepartition_hidden = int(mdp_loader_prepartition_hidden)
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self._raw_blend = RawDatasetBlend(
            backend=backend,
            root=root,
            max_samples=max_samples,
            subsets=subsets,
            split=split,
            seed=seed,
        )
        # Megatron may request more samples than the underlying dataset has.
        # Expose the requested virtual length and wrap indices modulo the source
        # length so finite datasets behave like the infinite mock generator.
        self._virtual_len = max(len(self._raw_blend), int(max_samples or 0))

    # ----- length -----
    def __len__(self):
        # Return the virtual length so the Megatron sampler can produce indices
        # up to the requested count. ``__getitem__`` wraps modulo the actual
        # underlying-data count.
        return self._virtual_len

    @staticmethod
    def _normalize_token_ids(ids) -> List[int]:
        if ids is None:
            return []
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()
        elif hasattr(ids, "ids"):
            ids = ids.ids
        return [int(value) for value in ids]

    def _text_to_tokens(self, text: str, length: Optional[int] = None) -> torch.Tensor:
        """Tokenize text with Megatron's configured tokenizer.

        ``length`` is an optional truncation budget. The loader never repeats,
        hashes, or otherwise fabricates lexical token ids.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            tokenizer = get_tokenizer()
            self.tokenizer = tokenizer

        text = text or ""
        ids = tokenizer.tokenize(text)

        if (
            isinstance(ids, (list, tuple))
            and ids
            and isinstance(ids[0], str)
        ):
            if hasattr(tokenizer, "tokens_to_ids"):
                ids = tokenizer.tokens_to_ids(list(ids))
            elif hasattr(tokenizer, "convert_tokens_to_ids"):
                ids = tokenizer.convert_tokens_to_ids(list(ids))
        token_ids = self._normalize_token_ids(ids)
        if length is not None:
            token_ids = token_ids[: max(0, int(length))]
        return torch.tensor(token_ids, dtype=torch.long)

    @staticmethod
    def _mimo_padded_len(length: int) -> int:
        length = int(length)
        if length <= 0:
            return 0
        return int(math.ceil(length / _MIMO_PACK_PAD_MULTIPLE) * _MIMO_PACK_PAD_MULTIPLE)

    def _pad_doc_to_mimo_multiple(self, doc: dict, max_length: Optional[int] = None) -> Optional[dict]:
        """Pad one built document to the MIMO/SFT segment multiple.

        ``content_len`` stays as the original token span. ``real_len`` becomes
        the padded segment length used for THD cu_seqlens.
        """
        content_len = int(doc.get("content_len", doc["real_len"]))
        padded_len = self._mimo_padded_len(content_len)
        if max_length is not None and padded_len > int(max_length):
            return None
        input_ids = doc["input_ids"].reshape(-1)
        if int(input_ids.numel()) < content_len:
            raise RuntimeError(
                f"doc input length {int(input_ids.numel())} shorter than "
                f"content_len {content_len}"
            )
        input_ids = input_ids[:content_len].clone()
        if padded_len > content_len:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.zeros(padded_len - content_len, dtype=torch.long),
                ],
                dim=0,
            )
        padded = dict(doc)
        padded["input_ids"] = input_ids
        padded["content_len"] = int(content_len)
        padded["real_len"] = int(padded_len)
        return padded

    def _image_sources(self, raw: RawSample):
        raw_descriptors = raw.image_descriptors or []
        if raw_descriptors and not raw.images:
            return [(None, desc) for desc in raw_descriptors]
        return [
            (
                image,
                raw_descriptors[image_idx] if image_idx < len(raw_descriptors) else None,
            )
            for image_idx, image in enumerate(raw.images)
        ]

    def _grid_from_descriptor_or_image(self, image, descriptor):
        cap_enabled = (
            self.image_size_max > 0
            or self.image_max_pixels > 0
            or self.image_min_pixels > 0
        )
        if cap_enabled and image is not None:
            return self._image_grid_thw_from_image(image)
        if isinstance(descriptor, dict):
            if cap_enabled and "width" in descriptor and "height" in descriptor:
                return self._image_grid_thw_from_size(
                    descriptor["width"],
                    descriptor["height"],
                )
            grid = descriptor.get("grid_thw")
            if grid is not None:
                return tuple(int(x) for x in grid)
            if "width" in descriptor and "height" in descriptor:
                return self._image_grid_thw_from_size(
                    descriptor["width"],
                    descriptor["height"],
                )
        if image is not None:
            return self._image_grid_thw_from_image(image)
        return None

    def _image_block(self, n_tokens: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.tensor([self.vision_start_token_id], dtype=torch.long),
                torch.full((int(n_tokens),), self.image_token_id, dtype=torch.long),
            ]
        )

    def _loss_mask_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        for token_id in (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
        ):
            loss_mask[input_ids == int(token_id)] = 0.0
        return loss_mask

    def _shifted_labels_and_loss_mask(
        self,
        input_ids: torch.Tensor,
        segment_lens: Sequence[int],
        content_lens: Sequence[int],
        reference_docs: Optional[Sequence[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(segment_lens) != len(content_lens):
            raise RuntimeError(
                "dataset segment metadata mismatch: "
                f"segment_lens={len(segment_lens)} content_lens={len(content_lens)}"
            )
        if int(input_ids.numel()) == 0:
            raise RuntimeError("cannot build labels for an empty dataset item")

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0

        loss_mask = self._loss_mask_from_input_ids(input_ids)
        input_len = int(input_ids.numel())
        offset = 0
        for seg_len, content_len in zip(segment_lens, content_lens):
            seg_len = int(seg_len)
            content_len = int(content_len)
            if seg_len < 0 or content_len < 0 or content_len > seg_len:
                raise RuntimeError(
                    "invalid dataset segment lengths: "
                    f"content_len={content_len}, seg_len={seg_len}"
                )
            if offset + seg_len > input_len:
                raise RuntimeError(
                    "dataset segment lengths exceed input length: "
                    f"offset={offset}, seg_len={seg_len}, input_len={input_len}"
                )
            if content_len > 0:
                loss_mask[offset + content_len - 1:offset + seg_len] = 0.0
            else:
                loss_mask[offset:offset + seg_len] = 0.0
            offset += seg_len

        if offset < input_len:
            loss_mask[offset:] = 0.0
        loss_mask[-1] = 0.0
        labels[loss_mask == 0.0] = -100
        if reference_docs is not None:
            if len(reference_docs) != len(segment_lens):
                raise ValueError("reference document count mismatch")
            offset = 0
            for doc, seg_len, content_len in zip(reference_docs, segment_lens, content_lens):
                if "reference_labels" in doc:
                    labels[offset:offset + content_len] = doc["reference_labels"]
                    loss_mask[offset:offset + content_len] = doc["reference_loss_mask"]
                offset += seg_len
        return labels, loss_mask

    def _prepartition_assignment_tensor(self, assignment: Dict[int, Sequence[Tuple[int, int]]]):
        rows = []
        for rank in sorted(assignment):
            for sample_idx, image_idx in assignment[rank]:
                rows.append([int(rank), int(sample_idx), int(image_idx)])
        if not rows:
            return torch.zeros(0, 3, dtype=torch.int32)
        return torch.tensor(rows, dtype=torch.int32)

    def _image_row_counts(self, rows: Sequence[Sequence[int]]) -> List[int]:
        return [
            vision_rows_from_grid(row, spatial_merge_size=self.spatial_merge_size)
            for row in rows
        ]

    def _attach_loader_prepartition(self, out: dict, descriptors: Sequence[Dict]) -> dict:
        if not getattr(self, "mdp_loader_prepartition", False):
            return out

        image_grid_thw = out.get("image_grid_thw")
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw):
            return out
        rows = [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in image_grid_thw.detach().cpu().tolist()
        ]
        descriptors = list(descriptors or [])
        if len(descriptors) != len(rows):
            raise RuntimeError(
                "loader_prepartition descriptor/grid mismatch: "
                f"descriptors={len(descriptors)} rows={len(rows)}"
            )

        world = max(1, int(self.mdp_loader_prepartition_world))
        rank = int(self.mdp_loader_prepartition_rank)
        if rank < 0 or rank >= world:
            raise RuntimeError(
                "loader_prepartition rank is outside its world: "
                f"rank={rank} world={world}"
            )

        costs = lpt_flops_from_grid_rows(
            rows,
            hidden=int(self.mdp_loader_prepartition_hidden),
        )
        assignment = balance_per_image_lpt_by_flops([costs], num_ranks=world)
        local_pixels = []
        local_grids = []
        local_raw_counts = []
        if bool(self.mdp_loader_prepartition_encoder_stage):
            for _sample_idx, image_idx in assignment.get(rank, []):
                image_idx = int(image_idx)
                grid = rows[image_idx]
                patches = materialize_descriptor(
                    descriptors[image_idx],
                    grid,
                    pixel_dim=int(self._pixel_dim),
                    patch_size=int(self.patch_size),
                )
                local_pixels.append(patches.to(torch.float32))
                local_grids.append(torch.tensor(grid, dtype=torch.long))
                local_raw_counts.append(int(grid[0]) * int(grid[1]) * int(grid[2]))

        if local_pixels:
            out["pixel_values"] = torch.cat(local_pixels, dim=0)
            local_grid = torch.stack(local_grids, dim=0)
        else:
            out["pixel_values"] = torch.zeros(0, self._pixel_dim, dtype=torch.float32)
            local_grid = torch.zeros(0, 3, dtype=torch.long)

        out["_mdp_prepartitioned_image_grid_thw"] = local_grid
        out["_mdp_prepartitioned_assignment"] = self._prepartition_assignment_tensor(
            assignment
        )
        out["_mdp_prepartitioned_row_counts"] = torch.tensor(
            self._image_row_counts(rows),
            dtype=torch.int32,
        )
        out["_mdp_prepartitioned_local_raw_counts"] = torch.tensor(
            local_raw_counts,
            dtype=torch.int32,
        )
        return out

    # ----- image helpers -----
    def _round_dim(self, x: int) -> int:
        unit = self.patch_size * self.spatial_merge_size  # 32 default
        if self.image_size_max > 0:
            x = min(x, self.image_size_max)
        x = max(unit, x)
        return (x // unit) * unit

    def _resize_hw(self, width: int, height: int) -> Tuple[int, int]:
        unit = self.patch_size * self.spatial_merge_size  # 32 default
        H = int(height)
        W = int(width)
        if self.image_size_max > 0:
            return self._round_dim(H), self._round_dim(W)
        min_pixels = int(getattr(self, "image_min_pixels", 0))
        max_pixels = int(getattr(self, "image_max_pixels", 0))
        if max_pixels > 0 and H * W > max_pixels:
            scale = math.sqrt(max_pixels / (H * W))
            H = math.floor(H * scale / unit) * unit
            W = math.floor(W * scale / unit) * unit
        elif min_pixels > 0 and H * W < min_pixels:
            scale = math.sqrt(min_pixels / (H * W))
            H = math.ceil(H * scale / unit) * unit
            W = math.ceil(W * scale / unit) * unit
        H = max(unit, H)
        W = max(unit, W)
        return (H // unit) * unit, (W // unit) * unit

    def _image_grid_thw_from_image(self, im: Image.Image) -> Tuple[int, int, int]:
        return self._image_grid_thw_from_size(im.size[0], im.size[1])

    def _image_grid_thw_from_size(self, width: int, height: int) -> Tuple[int, int, int]:
        H, W = self._resize_hw(width, height)
        return (1, H // self.patch_size, W // self.patch_size)

    def _preprocess_image(
        self,
        im: Image.Image,
        grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Return Qwen-VL flat patches and ``image_grid_thw`` for one image."""
        if grid is None:
            grid = self._image_grid_thw_from_image(im)
        grid = tuple(int(x) for x in grid)
        patches = preprocess_image_to_patches(
            im,
            grid,
            patch_size=int(self.patch_size),
            temporal_patch_size=int(self.temporal_patch_size),
            spatial_merge_size=int(self.spatial_merge_size),
        )
        return patches, grid

    def _fetch_raw(self, idx) -> RawSample:
        if hasattr(self, "_raw_blend"):
            return self._raw_blend[idx]
        if not self._index:
            raise RuntimeError(
                "Qwen35VLDataset index is empty; check "
                "dataset_root, subsets, split, and container mounts."
            )
        idx = idx % len(self._index)
        bi, li = self._index[idx]
        _, backend = self._backends[bi]
        return backend[li]

    def _build_doc(self, raw: RawSample) -> dict:
        reference = self._reference_content(raw)
        per_image_grid_thw: List[List[int]] = []
        per_image_patches: List[torch.Tensor] = []
        per_image_patch_count: List[int] = []
        per_image_descriptors: List[Dict] = []
        per_image_token_count: List[int] = []
        merge = self.spatial_merge_size
        metadata_only = getattr(self, "metadata_only_batch", False)
        image_block_cost = 0

        image_sources = list(self._image_sources(raw))
        for image, raw_descriptor in image_sources:
            grid = self._grid_from_descriptor_or_image(image, raw_descriptor)
            if grid is None:
                continue
            t_p, h_p, w_p = [int(x) for x in grid]

            tokens_for_this_image = t_p * (h_p // merge) * (w_p // merge)
            this_cost = 1 + tokens_for_this_image
            if image_block_cost + this_cost > self.seq_length:
                if not per_image_token_count:
                    raise ValueError(
                        "single image token span exceeds seq_length: "
                        f"image_tokens={this_cost}, seq_length={self.seq_length}"
                    )
                break

            if raw_descriptor is not None:
                descriptor = dict(raw_descriptor)
            else:
                descriptor = {
                    "kind": "mock_grid",
                    "materializer": "examples.multimodal_dev.data.mock",
                }
            descriptor["grid_thw"] = [int(t_p), int(h_p), int(w_p)]
            descriptor.setdefault("pixel_dim", int(self._pixel_dim))
            descriptor.setdefault(
                "temporal_patch_size",
                int(self.temporal_patch_size),
            )
            descriptor.setdefault("spatial_merge_size", int(self.spatial_merge_size))

            if metadata_only:
                patches = None
            elif image is not None:
                patches, (t_p, h_p, w_p) = self._preprocess_image(
                    image,
                    (t_p, h_p, w_p),
                )
            else:
                patches = materialize_descriptor(
                    descriptor,
                    (t_p, h_p, w_p),
                    pixel_dim=self._pixel_dim,
                    patch_size=self.patch_size,
                )

            per_image_grid_thw.append([t_p, h_p, w_p])
            if patches is not None:
                per_image_patches.append(patches)
            per_image_patch_count.append(t_p * h_p * w_p)
            descriptor["grid_thw"] = [int(t_p), int(h_p), int(w_p)]
            per_image_descriptors.append(descriptor)
            per_image_token_count.append(tokens_for_this_image)
            image_block_cost += this_cost

        text_budget = max(0, self.seq_length - image_block_cost)
        text_tokens_full = (self._text_to_tokens(raw.text) if reference is None
                            else torch.empty(0, dtype=torch.long))
        text_tokens = text_tokens_full[:text_budget]
        parts = []
        for n_tok in per_image_token_count:
            parts.append(self._image_block(n_tok))
        if int(text_tokens.numel()) > 0:
            parts.append(text_tokens)
        if not parts and reference is None:
            raise ValueError("empty multimodal sample")

        real_input_ids = torch.cat(parts) if reference is None else reference[0]
        if reference is not None:
            if len(per_image_grid_thw) != len(raw.image_descriptors or []):
                raise ValueError("reference images must not be dropped or truncated")
            starts = (real_input_ids == self.vision_start_token_id).nonzero().flatten().tolist()
            if len(starts) != len(per_image_token_count):
                raise ValueError("reference image-start count mismatch")
            for start, count in zip(starts, per_image_token_count):
                span = real_input_ids[start + 1:start + 1 + count]
                if span.numel() != count or not bool((span == self.image_token_id).all()):
                    raise ValueError("reference image-placeholder ordering mismatch")
            if int((real_input_ids == self.image_token_id).sum()) != sum(per_image_token_count):
                raise ValueError("reference image-placeholder count mismatch")
        real_len = int(real_input_ids.numel())
        if real_len > self.seq_length:
            raise RuntimeError(
                f"Expected unpadded length <= seq_length={self.seq_length}, got {real_len}"
            )

        if getattr(self, "metadata_only_batch", False):
            pixel_values = torch.zeros(0, self._pixel_dim, dtype=torch.float32)
        elif per_image_patches:
            pixel_values = torch.cat(per_image_patches, dim=0).to(torch.float32)
        else:
            pixel_values = torch.zeros(0, self._pixel_dim, dtype=torch.float32)

        if per_image_grid_thw:
            image_grid_thw = torch.tensor(per_image_grid_thw, dtype=torch.long)
        else:
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        result = {
            "input_ids": real_input_ids,
            "real_len": int(real_len),
            "content_len": int(real_len),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_images": len(per_image_grid_thw),
            "num_patches": int(sum(per_image_patch_count)),
            "_mdp_image_descriptors": per_image_descriptors,
        }
        if reference is not None:
            result["reference_labels"] = reference[1]
            result["reference_loss_mask"] = reference[2]
        return result

    def _reference_content(self, raw: RawSample):
        values = (raw.reference_input_ids, raw.reference_labels, raw.reference_loss_mask)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("reference IDs, labels and mask must be supplied together")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
               or value.ndim != 1 for value in values):
            raise ValueError("reference content requires one-dimensional CPU tensors")
        ids, labels, mask = values
        if not (0 < ids.numel() == labels.numel() == mask.numel() <= self.seq_length):
            raise ValueError("reference content length mismatch or truncation")
        if ids.dtype != torch.long or labels.dtype != torch.long:
            raise ValueError("reference IDs and labels require int64")
        if mask.dtype != torch.float32:
            raise ValueError("reference loss mask requires float32")
        if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("reference loss mask must be finite binary values")
        if raw.images:
            raise ValueError("reference content requires descriptor-owned pixels")
        for desc in raw.image_descriptors or []:
            grid = desc.get("grid_thw", [])
            if len(grid) != 3 or any(not isinstance(v, int) or v <= 0 for v in grid):
                raise ValueError("invalid reference temporal/spatial grid")
            if grid[1] % self.spatial_merge_size or grid[2] % self.spatial_merge_size:
                raise ValueError("reference spatial grid must be merge divisible")
        if mask[-1] != 0 or labels[-1] != 0:
            raise ValueError("reference final token must have zero label/loss")
        return values

    def _finalize_container(self, doc: dict) -> dict:
        S = self.seq_length
        doc = self._pad_doc_to_mimo_multiple(doc, max_length=S)
        if doc is None:
            raise RuntimeError(f"document does not fit seq_length {S}")
        real_total = int(doc["real_len"])
        content_len = int(doc.get("content_len", real_total))
        if real_total > S:
            raise RuntimeError(
                f"unpadded length {real_total} exceeds seq_length {S}"
            )
        pad_len = S - real_total
        input_parts = [doc["input_ids"]]
        if pad_len > 0:
            input_parts.append(torch.zeros(pad_len, dtype=torch.long))
        input_ids = torch.cat(input_parts)
        if input_ids.shape[0] != S:
            raise RuntimeError(
                f"Expected container sequence length {S}, got {input_ids.shape[0]}"
            )

        labels, loss_mask = self._shifted_labels_and_loss_mask(
            input_ids,
            [real_total],
            [content_len],
            reference_docs=[doc],
        )

        if int(doc["pixel_values"].shape[0]) > 0:
            pixel_values = doc["pixel_values"].to(torch.float32)
        else:
            pixel_values = torch.zeros(0, self._pixel_dim, dtype=torch.float32)

        if int(doc["image_grid_thw"].shape[0]) > 0:
            image_grid_thw = doc["image_grid_thw"]
        else:
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        # 3D MRoPE positions over the full container.
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        position_ids = position_ids.squeeze(1)
        if position_ids.shape != (3, S):
            raise RuntimeError(
                f"Expected position_ids shape {(3, S)}, got "
                f"{tuple(position_ids.shape)}"
            )

        out = {
            "input_ids": input_ids,
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_images": int(doc["num_images"]),
        }
        if getattr(self, "metadata_only_batch", False):
            descriptors = doc.get("_mdp_image_descriptors", [])
            out["_mdp_image_descriptors_json"] = encode_image_descriptors(descriptors)
            out = self._attach_loader_prepartition(out, descriptors)
        if self.emit_cu_seqlens:
            real_len = int(doc["real_len"])
            out["cu_seqlens"] = torch.tensor([0, real_len], dtype=torch.int32)
            out["cu_seqlens_padded"] = torch.tensor([0, S], dtype=torch.int32)
            out["max_seqlen"] = torch.tensor(S, dtype=torch.int32)
            out["image_cu_seqlens"] = torch.tensor(
                [0, int(doc["num_images"])], dtype=torch.int32,
            )
            out["pixel_cu_seqlens"] = torch.tensor(
                [0, int(doc["num_patches"])], dtype=torch.int32,
            )
        self._reference_geometry(out, [doc])
        return self._static_thd_metadata(out)

    def _position_ids_for_doc(self, input_ids: torch.Tensor,
                                   image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Return MRoPE ids for one document, resetting positions at its start."""
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        return position_ids.squeeze(1)

    def _finalize_packed_container(self, docs: Sequence[dict]) -> dict:
        """Build a fixed-length SFT-style THD packed item from documents."""
        if not docs:
            raise RuntimeError("Cannot build an empty packed item")

        S = self.seq_length
        padded_docs = []
        used = 0
        for doc in docs:
            padded = self._pad_doc_to_mimo_multiple(doc, max_length=S - used)
            if padded is None:
                raise RuntimeError(
                    f"packed document does not fit remaining sequence budget {S - used}"
                )
            padded_docs.append(padded)
            used += int(padded["real_len"])
        docs = padded_docs
        real_lens = [int(doc["real_len"]) for doc in docs]
        content_lens = [int(doc.get("content_len", doc["real_len"])) for doc in docs]
        real_total = sum(real_lens)
        if real_total > S:
            raise RuntimeError(
                f"packed unpadded length {real_total} exceeds seq_length {S}"
            )

        input_ids_real = torch.cat([doc["input_ids"] for doc in docs], dim=0)
        if input_ids_real.shape[0] != real_total:
            raise RuntimeError(
                f"Expected packed unpadded length {real_total}, got "
                f"{input_ids_real.shape[0]}"
            )
        if self.cp_size > 1:
            pad_granularity = 2 * int(self.cp_size)
            for doc_len in real_lens:
                if doc_len % pad_granularity != 0:
                    raise RuntimeError(
                        "packed document length must be divisible by "
                        f"2*CP={pad_granularity}; got {doc_len}"
                    )

        pad_len = S - real_total
        if pad_len > 0:
            input_ids = torch.cat([
                input_ids_real,
                torch.zeros(pad_len, dtype=torch.long),
            ])
        else:
            input_ids = input_ids_real
        if input_ids.shape[0] != S:
            raise RuntimeError(
                f"Expected packed container length {S}, got {input_ids.shape[0]}"
            )

        labels, loss_mask = self._shifted_labels_and_loss_mask(
            input_ids,
            real_lens,
            content_lens,
            reference_docs=docs,
        )

        pixel_parts = [
            doc["pixel_values"].to(torch.float32)
            for doc in docs if int(doc["pixel_values"].shape[0]) > 0
        ]
        if pixel_parts:
            pixel_values = torch.cat(pixel_parts, dim=0)
        else:
            pixel_values = torch.zeros(0, self._pixel_dim, dtype=torch.float32)

        grid_parts = [
            doc["image_grid_thw"]
            for doc in docs if int(doc["image_grid_thw"].shape[0]) > 0
        ]
        if grid_parts:
            image_grid_thw = torch.cat(grid_parts, dim=0)
        else:
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        position_parts = [
            self._position_ids_for_doc(doc["input_ids"], doc["image_grid_thw"])
            for doc in docs
        ]
        position_ids_real = torch.cat(position_parts, dim=1)
        if position_ids_real.shape != (3, real_total):
            raise RuntimeError(
                f"Expected packed position_ids shape {(3, real_total)}, got "
                f"{tuple(position_ids_real.shape)}"
            )
        if pad_len > 0:
            if position_ids_real.shape[1] == 0:
                pad_positions = torch.arange(pad_len, dtype=torch.long).repeat(3, 1)
            else:
                increments = torch.arange(
                    1, pad_len + 1, dtype=position_ids_real.dtype,
                ).view(1, -1)
                pad_positions = position_ids_real[:, -1:] + increments
            position_ids = torch.cat([position_ids_real, pad_positions], dim=1)
        else:
            position_ids = position_ids_real
        if position_ids.shape != (3, S):
            raise RuntimeError(
                f"Expected packed container position_ids shape {(3, S)}, got "
                f"{tuple(position_ids.shape)}"
            )

        out = {
            "input_ids": input_ids,
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_images": int(sum(int(doc["num_images"]) for doc in docs)),
        }
        if getattr(self, "metadata_only_batch", False):
            descriptors = []
            for doc in docs:
                descriptors.extend(doc.get("_mdp_image_descriptors", []))
            out["_mdp_image_descriptors_json"] = encode_image_descriptors(
                descriptors
            )
            out = self._attach_loader_prepartition(out, descriptors)
        if self.emit_cu_seqlens:
            cu = [0]
            for doc_len in real_lens:
                cu.append(cu[-1] + doc_len)
            cu_padded = list(cu)
            cu_padded[-1] = S
            image_cu = [0]
            pixel_cu = [0]
            for doc in docs:
                image_cu.append(image_cu[-1] + int(doc["num_images"]))
                pixel_cu.append(pixel_cu[-1] + int(doc["num_patches"]))
            padded_seg_lens = [
                cu_padded[i + 1] - cu_padded[i]
                for i in range(len(cu_padded) - 1)
            ]
            if self.cp_size > 1:
                pad_granularity = 2 * int(self.cp_size)
                for seg_len in padded_seg_lens:
                    if seg_len % pad_granularity != 0:
                        raise RuntimeError(
                            "packed padded segment length must be divisible by "
                            f"2*CP={pad_granularity}; got {seg_len}"
                        )

            out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
            out["cu_seqlens_padded"] = torch.tensor(cu_padded, dtype=torch.int32)
            out["max_seqlen"] = torch.tensor(max(padded_seg_lens), dtype=torch.int32)
            out["image_cu_seqlens"] = torch.tensor(image_cu, dtype=torch.int32)
            out["pixel_cu_seqlens"] = torch.tensor(pixel_cu, dtype=torch.int32)
        self._reference_geometry(out, docs)
        return self._static_thd_metadata(out)

    @staticmethod
    def _reference_geometry(out, docs):
        """Original global content/grid geometry, never padded tokens or owner-local pixels."""
        is_reference = ["reference_labels" in doc for doc in docs]
        if not any(is_reference):
            return
        if not all(is_reference):
            raise ValueError("mixed reference/non-reference geometry is not supported")
        lengths = [int(doc["content_len"]) for doc in docs]
        rows = [row for doc in docs for row in doc["image_grid_thw"].tolist()]
        out["_reference_flops_geometry"] = torch.tensor([
            sum(lengths), sum(length * length for length in lengths),
            sum(t * h * w for t, h, w in rows),
            sum(t * (h * w) ** 2 for t, h, w in rows),
        ], dtype=torch.int64, device="cpu")

    def _static_thd_metadata(self, out):
        """Pad decoder metadata only; dummy slots never create vision items."""
        if not getattr(self, "thd_static_packing", False):
            return out
        cu = out["cu_seqlens"].tolist()
        cap = self.thd_max_packed_sequences
        if len(cu) - 1 > cap - 1:
            raise ValueError("static THD real documents exceed reserved capacity")
        if cu[-1] < self.seq_length:
            if bool(out["loss_mask"][cu[-1]:].any()):
                raise ValueError("static THD tail must have zero loss")
            cu.append(self.seq_length)
        if cu[-1] != self.seq_length or any((b - a) % (2 * self.cp_size)
                                          for a, b in zip(cu, cu[1:])):
            raise ValueError("static THD boundaries violate CP/global budget")
        cu.extend([self.seq_length] * (cap + 1 - len(cu)))
        out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
        out["cu_seqlens_padded"] = out["cu_seqlens"].clone()
        out["max_seqlen"] = torch.tensor(self.seq_length, dtype=torch.int32)
        # Real image-index ownership is unaffected by repeats of the final count.
        for key in ("image_cu_seqlens", "pixel_cu_seqlens"):
            values = out[key].tolist()
            values.extend([values[-1]] * (cap + 1 - len(values)))
            out[key] = torch.tensor(values, dtype=torch.int32)
        return out

    def _build_packed_item(self, idx) -> dict:
        """Pack raw records until the next padded doc would exceed ``seq_length``."""
        scan_span = self._pack_scan_span
        start_stride = int(
            getattr(self, "_pack_start_stride", self._pack_scan_span)
        )
        start = int(idx) * start_stride
        docs = []
        used = 0

        for off in range(scan_span):
            if (getattr(self, "thd_static_packing", False)
                    and len(docs) >= self.thd_max_packed_sequences - 1):
                break
            raw_idx = start + off
            raw = self._fetch_raw(raw_idx)
            doc = self._build_doc(raw)
            doc_padded = self._pad_doc_to_mimo_multiple(doc)
            if doc_padded is None:
                raise RuntimeError("document produced no packable tokens")
            doc_len = int(doc_padded["real_len"])
            remaining = int(self.seq_length - used)
            if remaining <= 0:
                break
            elif doc_len <= remaining:
                docs.append(doc_padded)
                used += doc_len
            else:
                break

            if used >= self.seq_length:
                break

        out = self._finalize_packed_container(docs)
        return out

    # ----- main item -----
    def __getitem__(self, idx):
        # Wrap indices for cyclic behavior so fixed-iteration throughput runs
        # do not exhaust smaller dataset subsets mid-training.
        if (
            getattr(self, "dataloader_sequence_packing", False)
            or self.pack_samples_per_item > 1
        ):
            return self._build_packed_item(idx)

        raw = self._fetch_raw(idx)
        doc = self._build_doc(raw)
        out = self._finalize_container(doc)
        return out


# ---------------------------------------------------------------------------
# train_valid_test_datasets_provider - match mock.py's signature so this can
# slot directly into the Megatron-LM data wiring.
# ---------------------------------------------------------------------------

def _safe_parallel_int(getter, default: int) -> int:
    if not parallel_state.is_initialized():
        return int(default)
    return int(getter())


def _parallel_order(args) -> str:
    if bool(getattr(args, "use_tp_pp_dp_mapping", False)):
        return "tp-cp-ep-pp-dp"
    return "tp-cp-ep-dp-pp"


def _pp_cp_prepartition_rank(args, pp_size: int, cp_size: int) -> int:
    if int(pp_size) <= 1:
        return _safe_parallel_int(parallel_state.get_context_parallel_rank, 0)

    fallback = (
        _safe_parallel_int(parallel_state.get_pipeline_model_parallel_rank, 0)
        * int(cp_size)
        + _safe_parallel_int(parallel_state.get_context_parallel_rank, 0)
    )
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(fallback)
    tp_size = _safe_parallel_int(
        parallel_state.get_tensor_model_parallel_world_size,
        int(getattr(args, "tensor_model_parallel_size", 1)),
    )
    groups = compute_pp_cp_inner_dp_layout(
        world_size=int(torch.distributed.get_world_size()),
        tp_size=int(tp_size),
        cp_size=int(cp_size),
        pp_size=int(pp_size),
        order=_parallel_order(args),
    )
    _group, local_index = find_pp_cp_inner_dp_group_for_rank(
        int(torch.distributed.get_rank()),
        groups,
    )
    return int(local_index)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Drop-in equivalent of ``mock.train_valid_test_datasets_provider``.

    Reads the Megatron args set by ``arguments.py`` plus a few new flags
    documented inline.
    """
    args = get_args()

    backend = getattr(args, "dataset_backend", "mantis-instruct")
    if isinstance(backend, str) and "," in backend:
        backend = [b.strip() for b in backend.split(",") if b.strip()]
    root = getattr(args, "dataset_root", None)
    subsets = getattr(args, "dataset_subsets", None)
    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(",") if s.strip()]
    split = getattr(args, "dataset_split", "train")
    pack_samples_per_item = int(getattr(args, "pack_samples_per_item", 1))
    pack_scan_multiplier = int(getattr(args, "pack_scan_multiplier", 1))
    dataloader_sequence_packing = bool(
        getattr(args, "dataloader_sequence_packing", False)
    )
    emits_prepacked_batch = bool(
        getattr(args, "dynamic_context_parallel", False)
        or getattr(args, "pack_sequences", False)
        or getattr(args, "use_packed_sequence", False)
        or pack_samples_per_item > 1
        or dataloader_sequence_packing
    )
    if emits_prepacked_batch and int(getattr(args, "micro_batch_size", 1)) != 1:
        raise ValueError(
            "direct blend THD batches currently require --micro-batch-size 1."
        )
    if dataloader_sequence_packing:
        # ``pack_sequences`` was the b436-era spelling.  The split PR stack
        # exposes ``use_packed_sequence`` to the model/runtime, so keep both
        # views in sync when the direct-blend loader enables THD packing.
        args.pack_sequences = True
        args.use_packed_sequence = True

    mdp_enabled = bool(getattr(args, "mdp_encoder_mode", True))
    metadata_only_batch = mdp_enabled
    loader_prepartition = mdp_enabled
    inner_scope = getattr(args, "mdp_inner_dp_scope", "cp")
    if loader_prepartition and dataloader_sequence_packing:
        if inner_scope not in ("cp", "pp_cp"):
            raise ValueError(
                "loader_prepartition with dataloader-side packing supports "
                "mdp_inner_dp_scope in {'cp', 'pp_cp'} only."
            )
    elif loader_prepartition and inner_scope == "pp_cp":
        raise ValueError(
            "loader_prepartition with mdp_inner_dp_scope=pp_cp requires "
            "--dataloader-sequence-packing"
        )

    cp_size = _safe_parallel_int(
        parallel_state.get_context_parallel_world_size,
        int(getattr(args, "context_parallel_size", 1)),
    )
    cp_rank = _safe_parallel_int(parallel_state.get_context_parallel_rank, 0)
    pp_size = _safe_parallel_int(
        parallel_state.get_pipeline_model_parallel_world_size,
        int(getattr(args, "pipeline_model_parallel_size", 1)),
    )
    pp_rank = _safe_parallel_int(parallel_state.get_pipeline_model_parallel_rank, 0)
    if dataloader_sequence_packing:
        dp_rank = _safe_parallel_int(
            lambda: parallel_state.get_data_parallel_rank(with_context_parallel=True),
            0,
        ) // max(1, int(cp_size))
    else:
        dp_rank = _safe_parallel_int(parallel_state.get_data_parallel_rank, 0)

    prepartition_rank = cp_rank
    prepartition_world = cp_size
    prepartition_encoder_stage = (pp_rank == 0 or pp_size == 1)
    if (
        loader_prepartition
        and dataloader_sequence_packing
        and inner_scope == "pp_cp"
    ):
        prepartition_rank = _pp_cp_prepartition_rank(args, pp_size, cp_size)
        prepartition_world = int(pp_size) * int(cp_size)
        prepartition_encoder_stage = True

    kwargs = dict(
        thd_static_packing=getattr(args, "thd_static_packing", False),
        thd_max_packed_sequences=getattr(args, "thd_max_packed_sequences", 32),
        max_seqlen_per_dp_cp_rank=getattr(args, "max_seqlen_per_dp_cp_rank", None),
        backend=backend,
        root=root,
        seq_length=getattr(args, "total_seq_length", 4096),
        vocab_size=getattr(args, "padded_vocab_size", QWEN35_VL_VOCAB_SIZE),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        patch_size=getattr(args, "patch_size", 16),
        temporal_patch_size=getattr(args, "temporal_patch_size", 2),
        spatial_merge_size=getattr(args, "spatial_merge_size", 2),
        image_size_max=int(getattr(args, "image_size_max", 0)),
        image_max_pixels=int(getattr(args, "image_max_pixels", 0)),
        image_min_pixels=int(getattr(args, "image_min_pixels", 0)),
        cp_size=int(getattr(args, "context_parallel_size", 1)),
        emit_cu_seqlens=emits_prepacked_batch,
        pack_samples_per_item=pack_samples_per_item,
        pack_scan_multiplier=pack_scan_multiplier,
        dataloader_sequence_packing=dataloader_sequence_packing,
        dataloader_dp_rank=dp_rank,
        metadata_only_batch=metadata_only_batch,
        mdp_loader_prepartition=loader_prepartition,
        mdp_loader_prepartition_rank=prepartition_rank,
        mdp_loader_prepartition_world=prepartition_world,
        mdp_loader_prepartition_encoder_stage=prepartition_encoder_stage,
        mdp_loader_prepartition_hidden=int(getattr(args, "vision_hidden_size", 1280)),
        subsets=subsets,
        split=split,
    )
    train_ds = Qwen35VLDataset(max_samples=train_val_test_num_samples[0], **kwargs)
    val_ds   = Qwen35VLDataset(max_samples=train_val_test_num_samples[1], **kwargs)
    test_ds  = Qwen35VLDataset(max_samples=train_val_test_num_samples[2], **kwargs)
    return train_ds, val_ds, test_ds
