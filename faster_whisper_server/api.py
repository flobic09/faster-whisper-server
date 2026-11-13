"""FastAPI application and endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Union

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .audio import decode_wav
from .models import (
    AudioJsonResponse,
    AudioVerboseJsonResponse,
    WhisperSegment,
    get_transcriber_for_request,
    initialize_from_env,
)


def _decode_upload_sync(file: UploadFile) -> np.ndarray:
    return decode_wav(file.file.read(), sampling_rate=16000)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    initialize_from_env()
    yield


def _get_version() -> str:
    """Return the installed package version.

    The version is read from the installed distribution metadata, which is
    generated from ``pyproject.toml`` at build time. This keeps the FastAPI
    application version in sync with the project metadata without needing
    ``pyproject.toml`` to be shipped alongside the package.
    """
    try:
        return version("faster-whisper-server")
    except PackageNotFoundError:
        return "0.0.0"


app = FastAPI(title="faster-whisper-server", version=_get_version(), lifespan=_lifespan)


def _merge_options(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def _select_transcriber(model_name: str, task: str):
    try:
        return get_transcriber_for_request(model_name, task)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        ) from exc


async def _decode_upload(file: UploadFile) -> np.ndarray:
    try:
        return await run_in_threadpool(_decode_upload_sync, file)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        ) from exc
    finally:
        await file.close()


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[AudioJsonResponse, AudioVerboseJsonResponse]:
    """OpenAI compatible transcription endpoint."""
    transcriber, base_options, batch_size = _select_transcriber(
        model_name, "transcribe"
    )
    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )

    audio_input = await _decode_upload(file)

    options = _merge_options(
        base_options,
        {
            "task": "transcribe",
            "language": language,
            "initial_prompt": prompt,
            "temperature": temperature,
        },
    )
    if batch_size > 1:
        segments, info = transcriber.transcribe(
            audio_input, batch_size=batch_size, **options
        )
    else:
        segments, info = transcriber.transcribe(audio_input, **options)
    segments_list = list(segments)
    transcription = "".join(segment.text for segment in segments_list)

    if response_format == "verbose_json":
        return AudioVerboseJsonResponse(
            task="transcribe",
            language=info.language,
            duration=info.duration,
            text=transcription,
            segments=[
                WhisperSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                )
                for index, segment in enumerate(segments_list)
            ],
        )

    return AudioJsonResponse(text=transcription)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[AudioJsonResponse, AudioVerboseJsonResponse]:
    """OpenAI compatible translation endpoint."""
    transcriber, base_options, batch_size = _select_transcriber(model_name, "translate")
    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )

    audio_input = await _decode_upload(file)

    options = _merge_options(
        base_options,
        {
            "task": "translate",
            "initial_prompt": prompt,
            "temperature": temperature,
        },
    )
    if batch_size > 1:
        segments, info = transcriber.transcribe(
            audio_input, batch_size=batch_size, **options
        )
    else:
        segments, info = transcriber.transcribe(audio_input, **options)
    segments_list = list(segments)
    transcription = "".join(segment.text for segment in segments_list)

    if response_format == "verbose_json":
        return AudioVerboseJsonResponse(
            task="translate",
            language=info.language,
            duration=info.duration,
            text=transcription,
            segments=[
                WhisperSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                )
                for index, segment in enumerate(segments_list)
            ],
        )

    return AudioJsonResponse(text=transcription)
