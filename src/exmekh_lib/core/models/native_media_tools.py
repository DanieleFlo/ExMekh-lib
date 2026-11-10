"""Explicit native media tools using the existing attachment delivery path."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Literal

from PIL import Image

from exmekh_lib.config.native_suite import NATIVE_SUITE, ImageCount, ImageDimension
from exmekh_lib.core.models.native_suite import (
    ImageBatch,
    NativeSuiteClient,
    native_suite_client,
)
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from exmekh_lib.interfaces.operation_progress import operation_progress_observer


def workspace_media(work_root: Path, path: str) -> Path:
    root = work_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(NATIVE_SUITE.invalid_path)
    return resolved


def generate_images(
    work_root: Path,
    prompt: str,
    images: list[str] | None = None,
    n: ImageCount = 1,
    width: ImageDimension = NATIVE_SUITE.image_default_dimension,
    height: ImageDimension = NATIVE_SUITE.image_default_dimension,
    output_format: Literal["png"] = "png",
    background: Literal["auto", "opaque", "transparent"] = "auto",
    seed: int | None = None,
    idempotency_key: str | None = None,
    include_image_in_result: bool = NATIVE_SUITE.include_image_in_result,
    preview_scale: float = NATIVE_SUITE.preview_scale,
) -> dict[str, object]:
    NativeSuiteClient.validate_image_options(
        n=n,
        width=width,
        height=height,
        output_format=output_format,
        background=background,
        seed=seed,
    )
    if not 0 < preview_scale <= 1:
        raise ValueError(NATIVE_SUITE.invalid_scale)
    paths = [workspace_media(work_root, path) for path in images or []]
    if any(not path.is_file() for path in paths):
        raise ValueError(NATIVE_SUITE.invalid_reference)
    directory = workspace_media(
        work_root, f".exmekh_lib/generated-media/{uuid.uuid4().hex}"
    )
    with native_suite_client() as suite:
        references = [suite.upload(path, kind="image") for path in paths]
        response = suite.images(
            prompt,
            images=references,
            n=n,
            width=width,
            height=height,
            output_format=output_format,
            background=background,
            seed=seed,
            idempotency_key=idempotency_key,
            on_progress=operation_progress_observer(),
        )
    return _save_images(response, directory, include_image_in_result, preview_scale)


def _save_images(
    response: ImageBatch,
    directory: Path,
    include_image_in_result: bool,
    preview_scale: float,
) -> dict[str, object]:
    # Verify the complete batch before publishing any local file.
    payloads = [item.png_bytes() for item in response.data]
    directory.mkdir(parents=True)
    outputs: list[dict[str, object]] = []
    previews: list[Image.Image] = []
    for i, (item, raw) in enumerate(zip(response.data, payloads, strict=True)):
        path = directory / f"{i}.png"
        path.write_bytes(raw)
        outputs.append({**item.ds4.model_dump(), "path": str(path)})
        if include_image_in_result:
            with Image.open(io.BytesIO(raw)) as decoded:
                preview = decoded.copy()
            if preview_scale != 1:
                preview = preview.resize(
                    (
                        max(1, round(preview.width * preview_scale)),
                        max(1, round(preview.height * preview_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            previews.append(preview)
    result: dict[str, object] = {"images": outputs, "ds4": response.ds4}
    if previews:
        return AttachedToolResult(
            result,
            UserMessage(
                [NATIVE_SUITE.image_result_context, ImagesUserMessage(images=previews)]
            ),
        )
    return result
