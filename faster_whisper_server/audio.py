"""Decode the PCM WAV uploads accepted by the speech API."""

from __future__ import annotations

import struct

import numpy as np


def _read_format(chunk: bytes) -> tuple[int, int, int, int, int]:
    if len(chunk) < 16:
        raise ValueError("Invalid WAV format chunk")

    (
        audio_format,
        channels,
        sample_rate,
        _byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", chunk)
    if audio_format == 0xFFFE:
        if len(chunk) < 40:
            raise ValueError("Invalid extensible WAV format chunk")
        audio_format = struct.unpack_from("<H", chunk, 24)[0]

    if audio_format not in (1, 3):
        raise ValueError("WAV must contain PCM or IEEE float audio")
    if channels < 1 or sample_rate < 1 or bits_per_sample not in (8, 16, 24, 32, 64):
        raise ValueError("Unsupported WAV format")
    if audio_format == 3 and bits_per_sample not in (32, 64):
        raise ValueError("IEEE float WAV must use 32-bit or 64-bit samples")

    bytes_per_sample = (bits_per_sample + 7) // 8
    expected_block_align = channels * bytes_per_sample
    if block_align != expected_block_align:
        raise ValueError("Invalid WAV block alignment")

    return audio_format, channels, sample_rate, bits_per_sample, block_align


def decode_wav(data: bytes, sampling_rate: int = 16000) -> np.ndarray:
    """Decode a PCM/float WAV to mono float32 at ``sampling_rate``.

    This deliberately supports only the WAV formats sent by Dovah Voice VR,
    avoiding a runtime FFmpeg/PyAV dependency in the bundled server.
    """
    if sampling_rate < 1:
        raise ValueError("Target sampling rate must be positive")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Uploaded audio must be a RIFF/WAVE file")

    format_chunk = None
    audio_chunk = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            raise ValueError("Truncated WAV chunk")
        if chunk_id == b"fmt ":
            format_chunk = data[chunk_start:chunk_end]
        elif chunk_id == b"data" and audio_chunk is None:
            audio_chunk = data[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size & 1)

    if format_chunk is None or audio_chunk is None:
        raise ValueError("WAV is missing a format or data chunk")

    audio_format, channels, source_rate, bits, block_align = _read_format(format_chunk)
    if source_rate < 1:
        raise ValueError("Invalid WAV sample rate")
    if len(audio_chunk) % block_align != 0:
        raise ValueError("WAV data is not aligned to complete audio frames")

    if audio_format == 3:
        dtype = "<f4" if bits == 32 else "<f8"
        samples = np.frombuffer(audio_chunk, dtype=dtype).astype(np.float32)
    elif bits == 8:
        samples = np.frombuffer(audio_chunk, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif bits == 16:
        samples = np.frombuffer(audio_chunk, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 24:
        packed = np.frombuffer(audio_chunk, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = (values ^ 0x800000) - 0x800000
        samples = values.astype(np.float32) / 8388608.0
    elif bits == 32:
        samples = np.frombuffer(audio_chunk, dtype="<i4").astype(np.float32)
        samples = samples / 2147483648.0
    else:
        raise ValueError("Unsupported PCM WAV sample width")

    frames = samples.reshape(-1, channels)
    mono = frames.mean(axis=1, dtype=np.float32)
    mono = np.nan_to_num(mono, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(mono, -1.0, 1.0, out=mono)

    if source_rate == sampling_rate or mono.size == 0:
        return mono.astype(np.float32, copy=False)

    if sampling_rate < source_rate:
        # Apply an anti-aliasing low-pass filter before downsampling.
        ratio = sampling_rate / source_rate
        cutoff = 0.5 * ratio * 0.94
        taps = 63
        positions = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
        kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * positions)
        kernel *= np.hamming(taps)
        kernel /= kernel.sum()
        filtered = np.convolve(mono, kernel.astype(np.float32), mode="full")
        start = taps // 2
        mono = filtered[start : start + mono.size]

    target_length = int(round(mono.size * sampling_rate / source_rate))
    if target_length == 0:
        return np.empty(0, dtype=np.float32)
    source_positions = np.arange(mono.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_length, dtype=np.float64) / sampling_rate
    return np.interp(target_positions, source_positions, mono).astype(np.float32)
