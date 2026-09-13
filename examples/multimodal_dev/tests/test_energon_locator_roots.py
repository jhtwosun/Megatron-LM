# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static locator authority; no dataset image bytes are read by these tests."""

import argparse
from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.data.energon.materializer import (
    authorize_locator_storage_path,
    validate_locator_storage_roots,
)
from examples.multimodal_dev.data.energon.provider import _validate_provider_args


@pytest.mark.parametrize("roots", [None, [], (), "/tmp", ["/"], ["relative"], ["/tmp/../tmp"]])
def test_invalid_storage_authority(roots):
    with pytest.raises(ValueError):
        validate_locator_storage_roots(roots)


def test_root_aliases_and_duplicates_are_rejected(tmp_path):
    alias = tmp_path / "alias"
    root = tmp_path / "data"
    root.mkdir()
    alias.symlink_to(root, target_is_directory=True)
    assert validate_locator_storage_roots([str(alias)]) == (str(root),)
    for roots in ([str(root), str(root)], [str(root), str(alias)]):
        with pytest.raises(ValueError, match="duplicate"):
            validate_locator_storage_roots(roots)


def test_authority_roots_must_exist_and_be_directories(tmp_path):
    file = tmp_path / "file"
    file.touch()
    for path in (file, tmp_path / "missing"):
        with pytest.raises(ValueError, match="existing directories"):
            validate_locator_storage_roots([str(path)])


def test_prefix_sibling_and_symlink_escape_do_not_grant_authority(tmp_path):
    root, sibling = tmp_path / "rootA", tmp_path / "rootAB"
    root.mkdir()
    sibling.mkdir()
    (root / "escape").symlink_to(sibling, target_is_directory=True)
    roots = validate_locator_storage_roots([str(root)])
    for path in (sibling / "image.jpg", root / "escape/image.jpg", tmp_path / "outside.jpg"):
        with pytest.raises(ValueError, match="outside"):
            authorize_locator_storage_path(str(path), roots)
    assert authorize_locator_storage_path(str(root / "image.jpg"), roots) == (
        str(root),
        str(root / "image.jpg"),
    )


def test_multiple_roots_do_not_resolve_relative_descriptors_by_order(tmp_path):
    roots = []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        roots.append(str(directory))
    roots = validate_locator_storage_roots(roots)
    with pytest.raises(ValueError, match="absolute"):
        authorize_locator_storage_path("image.jpg", roots)
    for root in roots:
        assert authorize_locator_storage_path(root + "/image.jpg", roots)[0] == root


def test_nested_roots_have_deterministic_most_specific_authority(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    roots = validate_locator_storage_roots([str(tmp_path), str(nested)])
    assert authorize_locator_storage_path(str(nested / "image.jpg"), roots)[0] == str(nested)
    assert authorize_locator_storage_path(str(nested), (str(nested),), allow_root=True) == (
        str(nested),
        str(nested),
    )
    with pytest.raises(ValueError, match="outside"):
        authorize_locator_storage_path(str(nested), (str(nested),))


def test_cli_and_provider_preserve_no_root_default(tmp_path):
    parser = add_multimodal_args(argparse.ArgumentParser())
    parsed = parser.parse_args([])
    assert parsed.energon_vision_storage_roots is None
    explicit = parser.parse_args(["--energon-vision-storage-roots", str(tmp_path)])
    args = SimpleNamespace(
        dataloader_type="external",
        energon_path="/unchanged/metadataset.yaml",
        energon_split="train",
        energon_val_split="val",
        num_workers=0,
        energon_packing_buffer_size=1,
        energon_max_samples_per_sequence=1,
        energon_shuffle_buffer_size=1,
        energon_prefetch_factor=1,
    )
    _validate_provider_args(args)
    assert not hasattr(args, "energon_vision_storage_roots")
    args.energon_vision_storage_roots = explicit.energon_vision_storage_roots
    _validate_provider_args(args)
    assert args.energon_vision_storage_roots == (str(tmp_path),)
