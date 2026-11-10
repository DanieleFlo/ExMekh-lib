"""Native DS4 service identities and tool presentation settings."""

from dataclasses import dataclass
from typing import Literal

ImageDimension = Literal[
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
    992,
    1024,
    1056,
    1088,
    1120,
    1152,
    1184,
    1216,
    1248,
    1280,
    1312,
    1344,
    1376,
    1408,
    1440,
    1472,
    1504,
    1536,
    1568,
    1600,
    1632,
    1664,
    1696,
    1728,
    1760,
    1792,
    1824,
    1856,
    1888,
    1920,
    1952,
    1984,
    2016,
    2048,
]
ImageCount = Literal[1, 2, 3, 4, 5, 6, 7, 8]


@dataclass(frozen=True, slots=True)
class NativeSuiteSettings:
    image_model: Literal["Qwen/Qwen-Image-2.1-Turbo"] = "Qwen/Qwen-Image-2.1-Turbo"
    embedding_model: Literal["google/embeddinggemma-2"] = "google/embeddinggemma-2"
    dimensions: int = 768
    include_image_in_result: bool = True
    preview_scale: float = 1.0
    timeout_seconds: float = 600
    image_default_dimension: ImageDimension = 2048
    image_batch_pixels: int = 4_194_304
    image_description: str = (
        "Generate or edit images using the configured native image model. Supply an "
        "ordered list of reference paths for editing. Returns media IDs, local PNG "
        "paths, seeds, recipe and dimensions. include_image_in_result attaches "
        "labeled previews to the next model request; preview_scale changes previews only. "
        "Width and height are independent multiples of 32 from 32 through 2048, "
        "listed in their enums; n * width * height must not exceed 4194304 pixels. "
        "Use 2K for fine detail unless the user requests a smaller image: "
        "2048x2048 square, 2048x1536 landscape, 1536x2048 portrait, or 2048x1152 for 16:9. "
        "Smaller dimensions reduce available detail; 1920x1080 and 2560x1440 are unsupported. "
        "Generate multiple full-resolution images in separate calls to respect the batch pixel limit. "
        "Generation waits for completion before inference continues; background execution is unavailable."
    )
    image_result_context: str = (
        "These images are previews from the completed generate-image tool call. "
        "The original PNGs, file IDs, seeds and recipe are in its tool result. "
        "Continue the current request using that completed result."
    )
    unsupported_provider: str = "This provider does not expose the native suite."
    invalid_response: str = "The native service returned an invalid response."
    incomplete_stream: str = "The native progress stream ended without a terminal result."
    operation_failed: str = "The native image operation failed; inspect its structured cause before retrying."
    invalid_scale: str = "preview_scale must be greater than zero and at most one."
    invalid_path: str = "Media paths must stay inside the workspace."
    invalid_dimensions: str = "Width and height must each be a supported integer multiple of 32 from 32 to 2048."
    invalid_count: str = "n must be an integer from 1 to 8."
    invalid_image_batch: str = "n * width * height must not exceed 4194304 pixels."
    invalid_image_format: str = "The native image service supports PNG output only."
    invalid_image_background: str = "background must be auto, opaque or transparent."
    invalid_image_seed: str = (
        "seed must be a uint64 integer and leave room for all images in the batch."
    )
    invalid_reference: str = (
        "Every image reference must be an existing file inside the workspace."
    )


NATIVE_SUITE = NativeSuiteSettings()
