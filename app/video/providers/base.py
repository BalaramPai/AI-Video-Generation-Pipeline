from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Mapping, Sequence


@dataclass(frozen=True)
class VideoProviderCapabilities:
    text_to_video: bool
    image_to_video: bool
    reference_images: bool
    first_frame: bool
    last_frame: bool
    video_extension: bool
    supported_aspect_ratios: tuple[str, ...]


@dataclass(frozen=True)
class VideoGenerationResult:
    video_path: Path
    provider: str
    generation_id: str
    duration_seconds: int
    metadata: Mapping[str, object] = field(default_factory=dict)


class VideoProviderError(RuntimeError):
    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        retryable: bool,
        provider: str = "video",
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.reason = reason
        self.retryable = retryable
        super().__init__(
            f"{provider} {operation} failed "
            f"({'retryable' if retryable else 'not retryable'}): {reason}"
        )


class VideoGenerationTimeout(VideoProviderError, TimeoutError):
    def __init__(
        self,
        timeout_seconds: float,
        *,
        provider: str = "veo",
    ) -> None:
        super().__init__(
            "generation",
            f"operation did not complete within {timeout_seconds:g} seconds",
            retryable=True,
            provider=provider,
        )


class VideoGenerationCancelled(VideoProviderError):
    def __init__(self, *, provider: str = "veo") -> None:
        super().__init__(
            "generation",
            "polling was cancelled; the remote operation may continue",
            retryable=False,
            provider=provider,
        )


class VideoGenerationProvider(ABC):
    name: str
    capabilities: VideoProviderCapabilities

    @abstractmethod
    def generate_video(
        self,
        *,
        prompt: str,
        output_path: Path,
        duration_seconds: int | float | str | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        reference_images: Sequence[Path] = (),
        aspect_ratio: str = "16:9",
        cancel_event: Event | None = None,
    ) -> VideoGenerationResult:
        raise NotImplementedError
