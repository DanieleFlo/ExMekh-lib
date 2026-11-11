"""Versioned DS4 media operations on the same OpenAI transport as chat."""

from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, get_args

from httpx2 import Request as HTTPRequest
from httpx2 import Response as HTTPResponse
from openai import APIStatusError, OpenAI, Stream
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from exmekh_lib.config.native_suite import NATIVE_SUITE, ImageCount, ImageDimension
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.interfaces.operation_progress import OperationProgress

_IMAGE_UPLOAD_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


class TextContent(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class MediaContent(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["image", "video", "audio"]
    file_id: str
    audio: Literal["include", "exclude"] | None = None


class EmbeddingItem(BaseModel):
    model_config = {"extra": "forbid"}
    content: list[TextContent | MediaContent] = Field(min_length=1)


class GeneratedMedia(BaseModel):
    file_id: str
    seed: int
    width: int
    height: int
    recipe: dict[str, Any]


class GeneratedImage(BaseModel):
    b64_json: str
    ds4: GeneratedMedia

    def png_bytes(self) -> bytes:
        """Decode the wire payload and verify its declared format and geometry."""
        try:
            raw = base64.b64decode(self.b64_json, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.size != (
                    self.ds4.width,
                    self.ds4.height,
                ):
                    raise ValueError(NATIVE_SUITE.invalid_response)
                image.verify()
        except (binascii.Error, OSError, SyntaxError, ValueError) as error:
            raise ValueError(NATIVE_SUITE.invalid_response) from error
        return raw


class ImageBatch(BaseModel):
    created: int
    data: list[GeneratedImage]
    ds4: dict[str, Any]

    def validate_geometry(self, *, n: int, width: int, height: int) -> None:
        if len(self.data) != n or any(
            image.ds4.width != width or image.ds4.height != height
            for image in self.data
        ):
            raise ValueError(NATIVE_SUITE.invalid_response)


class NativeCapabilities(BaseModel):
    version: Literal[1]
    revision: str
    profile: Literal["bf16-native-v1"]
    operations: list[Literal["image_generate", "image_edit", "embed"]]
    input_modalities: list[Literal["text", "image", "video", "audio"]]
    configured: bool
    resident: bool
    ready: bool
    image_progress_version: Literal[1] | None = None


class NativeModel(BaseModel):
    id: str
    ds4: NativeCapabilities


class NativeSuiteClient:
    """No residency operations: each method waits for restored service state."""

    def __init__(
        self, client: OpenAI, *, image_model: str = NATIVE_SUITE.image_model
    ) -> None:
        self._transport = client
        self._image_model = image_model

    def models(self) -> list[NativeModel]:
        data = self._transport.get("/models", cast_to=dict[str, Any])["data"]
        return [
            NativeModel.model_validate(item)
            for item in data
            if item["id"] in (self._image_model, NATIVE_SUITE.embedding_model)
        ]

    def upload(
        self, path: Path, *, kind: Literal["image", "video", "audio"]
    ) -> MediaContent:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        mime = {"audio/x-wav": "audio/wav", "audio/x-flac": "audio/flac"}.get(
            mime, mime
        )
        if kind == "image" and mime not in _IMAGE_UPLOAD_MIME_TYPES:
            # The suite accepts PNG, JPEG and WebP. Decode other raster inputs
            # once and reuse the normal PNG upload path.
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                mode = "RGBA" if image.has_transparency_data else "RGB"
                return self.upload_image(
                    image.convert(mode), filename=path.with_suffix(".png").name
                )
        with path.open("rb") as source:
            response = self._transport.files.create(
                file=(path.name, source, mime),
                purpose="ds4_media",  # type: ignore[arg-type]
            )
        return MediaContent(type=kind, file_id=response.id)

    def upload_image(
        self, image: Image.Image, *, filename: str = "image.png"
    ) -> MediaContent:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        response = self._transport.files.create(
            file=(filename, stream.getvalue(), "image/png"),
            purpose="ds4_media",  # type: ignore[arg-type]
        )
        return MediaContent(type="image", file_id=response.id)

    def content(self, file_id: str) -> bytes:
        return self._transport.files.content(file_id).read()

    def embeddings(
        self,
        items: Sequence[EmbeddingItem] | Sequence[str],
        *,
        dimensions: int = NATIVE_SUITE.dimensions,
        task: Literal[
            "raw", "search_query", "document", "code_query", "similarity"
        ] = "raw",
        title: str | None = None,
    ) -> dict[str, Any]:
        extension: dict[str, Any] = {"version": 1, "task": task, "overflow": "error"}
        if title is not None:
            extension["title"] = title
        response = self._transport.post(
            "/embeddings",
            cast_to=dict[str, Any],
            body={
                "model": NATIVE_SUITE.embedding_model,
                "dimensions": dimensions,
                "encoding_format": "float",
                "input": [
                    item.model_dump(exclude_none=True)
                    if isinstance(item, EmbeddingItem)
                    else item
                    for item in items
                ],
                "ds4": extension,
            },
        )
        if len(response["data"]) != len(items) or any(
            row["index"] != i or len(row["embedding"]) != dimensions
            for i, row in enumerate(response["data"])
        ):
            raise ValueError(NATIVE_SUITE.invalid_response)
        return response

    @staticmethod
    def validate_image_options(
        *,
        n: int,
        width: int,
        height: int,
        output_format: str,
        background: str,
        seed: int | None,
    ) -> None:
        """Check native request options before transport or reference uploads."""
        if any(
            type(value) is not int or value not in get_args(ImageDimension)
            for value in (width, height)
        ):
            raise ValueError(NATIVE_SUITE.invalid_dimensions)
        if type(n) is not int or n not in get_args(ImageCount):
            raise ValueError(NATIVE_SUITE.invalid_count)
        if n * width * height > NATIVE_SUITE.image_batch_pixels:
            raise ValueError(NATIVE_SUITE.invalid_image_batch)
        if output_format != "png":
            raise ValueError(NATIVE_SUITE.invalid_image_format)
        if background not in ("auto", "opaque", "transparent"):
            raise ValueError(NATIVE_SUITE.invalid_image_background)
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**64 - n):
            raise ValueError(NATIVE_SUITE.invalid_image_seed)

    def images(
        self,
        prompt: str,
        *,
        images: Sequence[MediaContent] = (),
        n: ImageCount = 1,
        width: ImageDimension = NATIVE_SUITE.image_default_dimension,
        height: ImageDimension = NATIVE_SUITE.image_default_dimension,
        output_format: Literal["png"] = "png",
        background: Literal["auto", "opaque", "transparent"] = "auto",
        seed: int | None = None,
        idempotency_key: str | None = None,
        on_progress: Callable[[OperationProgress], None] | None = None,
    ) -> ImageBatch:
        self.validate_image_options(
            n=n,
            width=width,
            height=height,
            output_format=output_format,
            background=background,
            seed=seed,
        )
        extension: dict[str, Any] = {"version": 1, "sampling": "checkpoint"}
        if seed is not None:
            extension["seed"] = seed
        body: dict[str, Any] = {
            "model": self._image_model,
            "prompt": prompt,
            "n": n,
            "size": f"{width}x{height}",
            "output_format": output_format,
            "background": background,
            "ds4": extension,
        }
        if images:
            body["images"] = [item.file_id for item in images]
        path = "/images/edits" if images else "/images/generations"
        headers = {"Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        # Image calls can have effects before a transport failure. Recovery
        # must be explicit; 409 cancellation and 503 resource errors are final.
        transport = self._transport.with_options(max_retries=0)
        completion = None
        if on_progress is None:
            response = transport.post(
                path, cast_to=dict[str, Any], body=body, options={"headers": headers}
            )
        else:
            headers["Accept"] = "text/event-stream"
            with transport.post(
                path,
                cast_to=dict[str, Any],
                body=body,
                options={"headers": headers},
                stream=True,
                stream_cls=Stream[dict[str, Any]],
            ) as events:
                content_type = events.response.headers.get("content-type", "")
                if (
                    content_type.partition(";")[0].strip().lower()
                    == "text/event-stream"
                ):
                    response, completion = self._image_events(events, on_progress)
                else:
                    # An older server can return ordinary JSON to negotiation.
                    events.response.read()
                    response = events.response.json()
        batch = ImageBatch.model_validate(response)
        batch.validate_geometry(n=n, width=width, height=height)
        if completion is not None and on_progress is not None:
            on_progress(completion)
        return batch

    @staticmethod
    def _image_events(
        events: Stream[dict[str, Any]],
        callback: Callable[[OperationProgress], None],
    ) -> tuple[dict[str, Any], OperationProgress]:
        previous = None
        for payload in events:
            event = OperationProgress.model_validate(payload)
            if not event.follows(previous):
                raise ValueError(NATIVE_SUITE.invalid_response)
            if event.state == "completed":
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise ValueError(NATIVE_SUITE.invalid_response)
                return result, event
            if event.state != "running":
                failure = NativeSuiteClient._image_stream_error(
                    events.response.request, event.request_id, payload
                )
                callback(event)
                raise failure
            callback(event)
            previous = event
        raise ValueError(NATIVE_SUITE.incomplete_stream)

    @staticmethod
    def _image_stream_error(
        request: HTTPRequest, request_id: str, payload: dict[str, Any]
    ) -> APIStatusError:
        failure = payload.get("failure")
        code = payload.get("status_code")
        if (
            not isinstance(code, int)
            or not 400 <= code <= 599
            or not isinstance(failure, dict)
        ):
            raise ValueError(NATIVE_SUITE.invalid_response)
        detail = failure.get("error")
        message = detail.get("message") if isinstance(detail, dict) else None
        return APIStatusError(
            message
            if isinstance(message, str) and message
            else NATIVE_SUITE.operation_failed,
            response=HTTPResponse(
                code,
                request=request,
                headers={"x-request-id": request_id},
                json=failure,
            ),
            body=failure,
        )


_suite_factory: ContextVar[Callable[[], NativeSuiteClient] | None] = ContextVar(
    "native_suite_factory", default=None
)


@contextmanager
def native_suite_context(factory: Callable[[], NativeSuiteClient]) -> Iterator[None]:
    token = _suite_factory.set(factory)
    try:
        yield
    finally:
        _suite_factory.reset(token)


@contextmanager
def native_suite_client() -> Iterator[NativeSuiteClient]:
    """Borrow the calling provider's transport, or own one for this operation."""
    factory = _suite_factory.get()
    if factory is not None:
        yield factory()
    else:
        settings = DEFAULT_PROVIDER_SETTINGS
        with OpenAI(
            base_url=os.environ[settings.base_url_env_var],
            api_key=os.getenv(settings.api_key_env_var) or settings.fallback_api_key,
            timeout=NATIVE_SUITE.timeout_seconds,
        ) as transport:
            yield NativeSuiteClient(transport)
