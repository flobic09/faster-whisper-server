import io
import struct
import wave

import numpy as np
import pytest

from faster_whisper_server.audio import decode_wav


def make_wav(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def test_decode_pcm16_mono_and_resample():
    source = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    decoded = decode_wav(make_wav(source, 8000), sampling_rate=16000)

    assert decoded.dtype == np.float32
    assert decoded.shape == (8,)
    np.testing.assert_allclose(decoded[::2], source.astype(np.float32) / 32768.0)


def test_decode_48k_mono_to_16k():
    positions = np.arange(4800, dtype=np.float64) / 48000.0
    source = (np.sin(2.0 * np.pi * 440.0 * positions) * 16383.0).astype(np.int16)
    decoded = decode_wav(make_wav(source, 48000), sampling_rate=16000)

    assert decoded.dtype == np.float32
    assert decoded.shape == (1600,)
    assert np.isfinite(decoded).all()


def test_decode_stereo_to_mono():
    source = np.array([[16384, -16384], [32767, 32767]], dtype=np.int16)
    decoded = decode_wav(make_wav(source, 16000, channels=2))

    np.testing.assert_allclose(decoded, [0.0, 32767.0 / 32768.0], atol=1e-6)


def test_decode_pcm24():
    # 24-bit little-endian PCM samples: +0.5 and -0.5.
    pcm24 = bytes([0x00, 0x00, 0x40, 0x00, 0x00, 0xC0])
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 48000, 3, 24)
    data_chunk = b"data" + struct.pack("<I", len(pcm24)) + pcm24
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body = b"WAVE" + fmt_chunk + data_chunk
    wav = b"RIFF" + struct.pack("<I", len(body)) + body

    np.testing.assert_allclose(decode_wav(wav), [0.5, -0.5], atol=1e-6)


@pytest.mark.parametrize("invalid", [b"", b"not wav", b"RIFF\x00\x00\x00\x00WAVE"])
def test_reject_invalid_wav(invalid: bytes):
    with pytest.raises(ValueError):
        decode_wav(invalid)


def test_reject_compressed_wav():
    source = np.zeros(100, dtype=np.int16)
    wav = bytearray(make_wav(source, 16000))
    # Change WAVE format tag from PCM to ADPCM.
    struct.pack_into("<H", wav, 20, 2)

    with pytest.raises(ValueError, match="PCM or IEEE float"):
        decode_wav(bytes(wav))
