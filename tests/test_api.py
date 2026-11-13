from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from faster_whisper_server import api


class FakeTranscriber:
    def __init__(self):
        self.audio = None

    def transcribe(self, audio, **_options):
        self.audio = audio
        return [SimpleNamespace(text="decoded wav")], None


def make_wav() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.full(1600, 1000, dtype="<i2").tobytes())
    return buffer.getvalue()


def test_transcription_endpoint_passes_decoded_array(monkeypatch):
    transcriber = FakeTranscriber()
    monkeypatch.setattr(
        api,
        "get_transcriber_for_request",
        lambda _model, _task: (transcriber, {}, 1),
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", make_wav(), "audio/wav")},
            data={"model_name": "skyrim-medium", "language": "en"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "decoded wav"}
    assert isinstance(transcriber.audio, np.ndarray)
    assert transcriber.audio.dtype == np.float32
    assert transcriber.audio.shape == (1600,)


def test_transcription_endpoint_rejects_non_wav(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_transcriber_for_request",
        lambda _model, _task: (FakeTranscriber(), {}, 1),
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.mp3", b"not wav", "audio/mpeg")},
            data={"model_name": "skyrim-medium"},
        )

    assert response.status_code == 400
    assert "RIFF/WAVE" in response.json()["detail"]["error"]["message"]
