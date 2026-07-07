# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Conversation tokenization and supervision masks for Qwen3.5-VL."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

import torch

_GENERATION_BLOCK_RE = re.compile(r"{%-?\s*generation\b")
_TURN_RE = re.compile(
    r"(?im)^(system|user|human|assistant|gpt|model)\s*:\s*"
)


class Qwen35ConversationEncoder:
    """Apply a tokenizer chat template and build an assistant-only mask."""

    def __init__(
        self,
        *,
        tokenizer,
        seq_length: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
        excluded_ids = {
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }
        excluded_ids.update(
            int(token_id)
            for token_id in (getattr(tokenizer, "all_special_ids", []) or [])
            if token_id is not None
        )
        self._excluded_token_ids = tuple(sorted(excluded_ids))
        self._excluded_token_id_set = frozenset(excluded_ids)

    @staticmethod
    def normalize_role(role: Any) -> str:
        role = str(role or "user").lower()
        if role in ("human", "user"):
            return "user"
        if role in ("gpt", "assistant", "model"):
            return "assistant"
        if role == "system":
            return "system"
        return "user"

    @staticmethod
    def content_to_text(content: Any) -> str:
        if not isinstance(content, list):
            return str(content)
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif part.get("type") == "image":
                parts.append("<image>")
            elif part.get("type") == "video":
                parts.append("<video>")
        return "".join(parts)

    @classmethod
    def turns_from_payload(cls, payload: Mapping[str, Any]) -> list[dict[str, str]]:
        text = payload.get("text")
        if text is not None:
            text = str(text)
            markers = list(_TURN_RE.finditer(text))
            if not markers:
                return [{"role": "user", "content": text}]
            turns = []
            for index, marker in enumerate(markers):
                end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
                content = text[marker.end() : end].strip()
                if content:
                    turns.append(
                        {
                            "role": cls.normalize_role(marker.group(1)),
                            "content": content,
                        }
                    )
            return turns

        conversation = (
            payload.get("conversation")
            or payload.get("conversations")
            or payload.get("messages")
            or []
        )
        turns = []
        for turn in conversation:
            if isinstance(turn, dict):
                turns.append(
                    {
                        "role": cls.normalize_role(
                            turn.get("role", turn.get("from", "user"))
                        ),
                        "content": cls.content_to_text(
                            turn.get("content", turn.get("value", ""))
                        ),
                    }
                )
        return turns

    @staticmethod
    def _normalize_token_ids(ids) -> list[int]:
        if ids is None:
            return []
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()
        elif hasattr(ids, "ids"):
            ids = ids.ids
        return [int(value) for value in ids]

    def excluded_token_ids(self) -> tuple[int, ...]:
        return self._excluded_token_ids

    def _mask_special_tokens(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        for token_id in self.excluded_token_ids():
            loss_mask[input_ids == token_id] = 0.0
        return loss_mask

    @staticmethod
    def _find_subsequence(
        values: Sequence[int], pattern: Sequence[int], start: int = 0
    ) -> int:
        if not pattern:
            return -1
        last = len(values) - len(pattern)
        for index in range(max(0, int(start)), last + 1):
            if list(values[index : index + len(pattern)]) == list(pattern):
                return index
        return -1

    def _split_text_with_images(
        self, text: str, *, start_image: int, max_images: int
    ) -> tuple[list[dict[str, str]], int]:
        parts = []
        image_index = int(start_image)
        cursor = 0
        for match in re.finditer(r"<image>", text or ""):
            if match.start() > cursor:
                parts.append({"type": "text", "text": text[cursor : match.start()]})
            if image_index < max_images:
                parts.append({"type": "image", "image": str(image_index)})
                image_index += 1
            cursor = match.end()
        if cursor < len(text or ""):
            parts.append({"type": "text", "text": text[cursor:]})
        return parts, image_index

    def _conversation_with_images(
        self, turns: Sequence[Mapping[str, Any]], num_images: int
    ) -> list[dict[str, Any]]:
        conversation = []
        image_index = 0
        first_user = None
        for turn in turns:
            role = self.normalize_role(turn.get("role", "user"))
            if first_user is None and role == "user":
                first_user = len(conversation)
            text = self.content_to_text(turn.get("content", ""))
            parts, image_index = self._split_text_with_images(
                text, start_image=image_index, max_images=num_images
            )
            content = parts if any(part.get("type") == "image" for part in parts) else text
            conversation.append({"role": role, "content": content})

        if image_index < num_images:
            media = [
                {"type": "image", "image": str(index)}
                for index in range(image_index, num_images)
            ]
            if not conversation:
                conversation.append({"role": "user", "content": media})
            else:
                target = first_user if first_user is not None else 0
                content = conversation[target]["content"]
                if isinstance(content, list):
                    conversation[target]["content"] = media + content
                elif content:
                    conversation[target]["content"] = media + [
                        {"type": "text", "text": str(content)}
                    ]
                else:
                    conversation[target]["content"] = media
        return conversation

    def _generation_mask(
        self, conversation: Sequence[Mapping[str, Any]]
    ) -> Optional[tuple[list[int], list[float]]]:
        template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(template, str) or _GENERATION_BLOCK_RE.search(template) is None:
            return None
        try:
            rendered = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(rendered, Mapping) or "assistant_masks" not in rendered:
            return None
        token_ids = self._normalize_token_ids(rendered.get("input_ids"))
        mask = [float(value) for value in self._normalize_token_ids(rendered["assistant_masks"])]
        if len(token_ids) != len(mask):
            return None
        return token_ids, mask

    def _chatml_mask(
        self,
        conversation: Sequence[Mapping[str, Any]],
        token_ids: Sequence[int],
        target_roles: set[str],
    ) -> Optional[list[float]]:
        end_ids = self._normalize_token_ids(
            self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        )
        if not end_ids:
            return None
        mask = [0.0] * len(token_ids)
        cursor = 0
        for turn in conversation:
            role = self.normalize_role(turn.get("role", "user"))
            start_ids = self._normalize_token_ids(
                self.tokenizer.encode(
                    f"<|im_start|>{role}\n", add_special_tokens=False
                )
            )
            start = self._find_subsequence(token_ids, start_ids, cursor)
            if start < 0:
                return None
            content_start = start + len(start_ids)
            content_end = self._find_subsequence(token_ids, end_ids, content_start)
            if content_end < 0:
                return None
            if role in target_roles:
                mask[content_start:content_end] = [1.0] * (content_end - content_start)
            cursor = content_end + len(end_ids)
        return mask

    def _apply_template(
        self, conversation: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], list[float]]:
        has_assistant = any(
            self.normalize_role(turn.get("role", "user")) == "assistant"
            for turn in conversation
        )
        if has_assistant:
            generated = self._generation_mask(conversation)
            if generated is not None and any(generated[1]):
                return generated

        token_ids = self._normalize_token_ids(
            self.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=False
            )
        )
        target_roles = {"assistant"} if has_assistant else {"user", "assistant"}
        mask = self._chatml_mask(conversation, token_ids, target_roles)
        if mask is None:
            raise ValueError(
                "The tokenizer chat template must expose assistant masks or "
                "ChatML role boundaries."
            )
        return token_ids, mask

    def _fallback_encode(
        self,
        turns: Sequence[Mapping[str, Any]],
        image_token_counts: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        parts = [self._image_block(count) for count in image_token_counts]
        masks = [torch.zeros(part.numel(), dtype=torch.float32) for part in parts]
        supervise_all = not any(
            self.normalize_role(turn.get("role", "user")) == "assistant"
            for turn in turns
        )
        for turn in turns:
            role = self.normalize_role(turn.get("role", "user"))
            prefix = self._tokenize(f"{role}: ")
            content = self._tokenize(self.content_to_text(turn.get("content", "")))
            suffix = self._tokenize("\n")
            parts.extend((prefix, content, suffix))
            masks.extend(
                (
                    torch.zeros(prefix.numel(), dtype=torch.float32),
                    torch.full(
                        (content.numel(),),
                        1.0 if role == "assistant" or supervise_all else 0.0,
                    ),
                    torch.zeros(suffix.numel(), dtype=torch.float32),
                )
            )
        input_ids = torch.cat([part for part in parts if part.numel()])[: self.seq_length]
        assistant_mask = torch.cat([part for part in masks if part.numel()])[: self.seq_length]
        return input_ids, self._mask_special_tokens(input_ids, assistant_mask), len(
            image_token_counts
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self._normalize_token_ids(
                self.tokenizer.encode(text or "", add_special_tokens=False)
            ),
            dtype=torch.long,
        )

    def _image_block(self, count: int) -> torch.Tensor:
        return torch.cat(
            (
                torch.tensor([self.vision_start_token_id], dtype=torch.long),
                torch.full((int(count),), self.image_token_id, dtype=torch.long),
                torch.tensor([self.vision_end_token_id], dtype=torch.long),
            )
        )

    def encode(
        self,
        turns: Sequence[Mapping[str, Any]],
        image_token_counts: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return token IDs, assistant supervision mask, and images retained."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            result = self._fallback_encode(turns, image_token_counts)
            self._validate_supervision(turns, result[1])
            return result

        conversation = self._conversation_with_images(turns, len(image_token_counts))
        token_ids, assistant_masks = self._apply_template(conversation)
        output = []
        mask_output = []
        image_index = 0
        index = 0
        while index < len(token_ids):
            token_id = int(token_ids[index])
            token_mask = float(assistant_masks[index])
            if token_id == self.image_token_id and image_index < len(image_token_counts):
                tail = []
                skip = 1
                if (
                    index + 1 < len(token_ids)
                    and int(token_ids[index + 1]) == self.vision_end_token_id
                ):
                    tail = [self.vision_end_token_id]
                    skip = 2
                count = int(image_token_counts[image_index])
                if len(output) + count + len(tail) > self.seq_length:
                    if output and output[-1] == self.vision_start_token_id:
                        output.pop()
                        mask_output.pop()
                    break
                output.extend([self.image_token_id] * count + tail)
                mask_output.extend([0.0] * (count + len(tail)))
                image_index += 1
                index += skip
                continue
            if len(output) >= self.seq_length:
                break
            output.append(token_id)
            if token_id in self._excluded_token_id_set:
                token_mask = 0.0
            mask_output.append(token_mask)
            index += 1

        input_ids = torch.tensor(output, dtype=torch.long)
        assistant_mask = self._mask_special_tokens(
            input_ids, torch.tensor(mask_output, dtype=torch.float32)
        )
        self._validate_supervision(turns, assistant_mask)
        return input_ids, assistant_mask, image_index

    def _validate_supervision(
        self, turns: Sequence[Mapping[str, Any]], assistant_mask: torch.Tensor
    ) -> None:
        has_answer = any(
            self.normalize_role(turn.get("role", "user")) == "assistant"
            and self.content_to_text(turn.get("content", "")).strip()
            for turn in turns
        )
        if has_answer and torch.count_nonzero(assistant_mask).item() == 0:
            raise ValueError(
                "Chat template produced no supervised tokens for a non-empty "
                "assistant turn."
            )
