# speech/transcribe.py — speech-to-text for voice goals (local + OpenAI)

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_WHISPER_MODEL,
    SPEECH_CLOUD_STT,
    SPEECH_LOCAL_WHISPER_MODEL,
)

_local_whisper_model = None


def _local_whisper_ready() -> bool:
    if not SPEECH_LOCAL_WHISPER_MODEL:
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def warmup_local_whisper() -> str:
    """Pre-load local Whisper model so first MIC press is faster."""
    if not _local_whisper_ready():
        return "disabled (pip install faster-whisper)"
    global _local_whisper_model
    if _local_whisper_model is not None:
        return f"{SPEECH_LOCAL_WHISPER_MODEL} already loaded"
    from faster_whisper import WhisperModel

    t0 = time.time()
    _local_whisper_model = WhisperModel(
        SPEECH_LOCAL_WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
    )
    return f"{SPEECH_LOCAL_WHISPER_MODEL} ready in {time.time() - t0:.1f}s"


def speech_config() -> dict[str, Any]:
    local_ok = _local_whisper_ready()
    openai_ok = bool(OPENAI_API_KEY)
    record_ok = local_ok or openai_ok
    return {
        "cloud_stt": SPEECH_CLOUD_STT,
        "record_available": record_ok,
        "cloud_available": record_ok,
        "local_whisper": local_ok,
        "local_whisper_model": SPEECH_LOCAL_WHISPER_MODEL if local_ok else None,
        "openai_whisper": openai_ok,
        "cloud_provider": (
            "local_faster_whisper"
            if local_ok and not openai_ok
            else "openai_whisper"
            if openai_ok
            else None
        ),
        "whisper_model": OPENAI_WHISPER_MODEL if openai_ok else SPEECH_LOCAL_WHISPER_MODEL,
        "browser_stt": True,
        "browser_recommended": False,
        "hint": (
            "Use Record mode (not Browser) on Chromium/Ubuntu. "
            "Hold MIC → release; audio is transcribed on the Jetson."
            if record_ok
            else (
                "Install local STT: pip install faster-whisper  "
                "OR set OPENAI_API_KEY for cloud Whisper."
            )
        ),
    }


def transcribe_audio(audio_bytes: bytes, *, filename: str = "speech.webm") -> dict[str, Any]:
    if not audio_bytes:
        return {"status": "error", "reason": "Empty audio upload."}

    order = SPEECH_CLOUD_STT.lower()
    try_local = order in ("local", "auto", "record") and _local_whisper_ready()
    try_openai = order in ("openai", "auto", "cloud") and bool(OPENAI_API_KEY)

    if order == "openai" and not OPENAI_API_KEY:
        try_local = _local_whisper_ready()

    errors: list[str] = []
    if try_local:
        try:
            text = _local_faster_whisper(audio_bytes, filename=filename)
            return {
                "status": "done",
                "text": text,
                "provider": "local_faster_whisper",
                "model": SPEECH_LOCAL_WHISPER_MODEL,
            }
        except Exception as e:
            errors.append(f"local: {e}")

    if try_openai:
        try:
            text = _openai_whisper(audio_bytes, filename=filename)
            return {"status": "done", "text": text, "provider": "openai_whisper"}
        except Exception as e:
            errors.append(f"openai: {e}")

    if not try_local and not try_openai:
        return {
            "status": "error",
            "reason": (
                "No STT backend. On Jetson run: pip install faster-whisper  "
                "OR export OPENAI_API_KEY=sk-..."
            ),
        }

    return {"status": "error", "reason": "; ".join(errors) or "Transcription failed."}


def _write_wav_pcm(path: str, pcm: bytes, *, rate: int, channels: int = 1) -> None:
    import struct
    import wave

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _decode_with_pyav(audio_bytes: bytes, wav_path: str) -> bool:
    try:
        import io

        import av
    except ImportError:
        return False

    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return False
            stream = audio_streams[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            chunks: list[bytes] = []
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    if out.planes:
                        chunks.append(bytes(out.planes[0]))
            if not chunks:
                return False
            _write_wav_pcm(wav_path, b"".join(chunks), rate=16000)
            return True
    except Exception:
        return False


def _wav_stats(wav_path: str) -> tuple[float, float]:
    import math
    import struct
    import wave

    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate() or 16000
        dur = wf.getnframes() / float(rate)
    if len(frames) < 4:
        return dur, 0.0
    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    return dur, rms


def _is_valid_wav(audio_bytes: bytes) -> bool:
    return (
        len(audio_bytes) > 44
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )


def _audio_to_wav_path(audio_bytes: bytes, filename: str, tmpdir: str) -> str:
    if len(audio_bytes) < 500:
        raise RuntimeError("Audio upload too small — hold MIC longer while speaking.")

    ext = os.path.splitext(filename or "speech.webm")[1] or ".webm"
    wav = os.path.join(tmpdir, "speech.wav")

    if ext.lower() == ".wav" and _is_valid_wav(audio_bytes):
        with open(wav, "wb") as f:
            f.write(audio_bytes)
        return wav

    src = os.path.join(tmpdir, f"speech{ext}")
    with open(src, "wb") as f:
        f.write(audio_bytes)

    ffmpeg = shutil.which("ffmpeg")
    errors: list[str] = []
    if ffmpeg:
        attempts = [
            [ffmpeg, "-y", "-i", src, "-ar", "16000", "-ac", "1", wav],
            [ffmpeg, "-y", "-f", "webm", "-i", src, "-ar", "16000", "-ac", "1", wav],
            [ffmpeg, "-y", "-f", "ogg", "-i", src, "-ar", "16000", "-ac", "1", wav],
        ]
        for cmd in attempts:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and os.path.isfile(wav) and os.path.getsize(wav) > 500:
                _, rms = _wav_stats(wav)
                if rms >= 15:
                    return wav
                errors.append(f"ffmpeg decoded silent audio (rms={rms:.0f})")
            else:
                errors.append(proc.stderr[-200:].strip())

    if _decode_with_pyav(audio_bytes, wav):
        _, rms = _wav_stats(wav)
        if rms >= 15:
            return wav
        errors.append(f"pyav decoded silent audio (rms={rms:.0f})")

    if not ffmpeg:
        raise RuntimeError("ffmpeg not found (sudo apt install ffmpeg).")

    raise RuntimeError(
        "Could not decode microphone audio (silent or corrupt recording). "
        "Hold MIC at least 2 seconds, speak clearly, and confirm ★ Jabra is selected. "
        f"Details: {errors[-1] if errors else 'unknown'}"
    )


def _looks_like_hallucination(text: str) -> bool:
    import re
    from collections import Counter

    if not text or not text.strip():
        return False
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 2:
        return False
    if len(words) >= 3:
        _word, count = Counter(words).most_common(1)[0]
        if count >= 3 and count / len(words) >= 0.55:
            return True
    junk = {
        "you", "yeah", "um", "uh", "the", "a", "i", "it",
        "thank", "thanks", "watching", "subscribe",
    }
    if all(w in junk for w in words):
        return True
    return False


def _whisper_text(wav_path: str, *, vad_filter: bool, language: str | None = "en") -> str:
    global _local_whisper_model
    segments, _ = _local_whisper_model.transcribe(
        wav_path,
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _local_faster_whisper(audio_bytes: bytes, *, filename: str) -> str:
    global _local_whisper_model
    from faster_whisper import WhisperModel

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = _audio_to_wav_path(audio_bytes, filename, tmpdir)
        dur, rms = _wav_stats(wav)
        if dur < 0.4:
            raise RuntimeError(
                f"Recording too short ({dur:.1f}s). Hold MIC at least 2 seconds while speaking."
            )
        if rms < 15:
            raise RuntimeError(
                f"Audio too quiet (RMS {rms:.0f}) — select ★ Jabra, speak louder into the mic."
            )

        if _local_whisper_model is None:
            _local_whisper_model = WhisperModel(
                SPEECH_LOCAL_WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
            )

        # No VAD first — VAD was stripping real speech on Jabra clips
        text = _whisper_text(wav, vad_filter=False)
        if not text:
            text = _whisper_text(wav, vad_filter=True)
        if not text:
            raise RuntimeError(
                f"No speech detected in {dur:.1f}s clip (RMS {rms:.0f}). "
                "Select ★ Jabra, hold MIC 2–3s, speak directly into the mic."
            )
        if _looks_like_hallucination(text):
            retry = _whisper_text(wav, vad_filter=True, language=None)
            if retry and not _looks_like_hallucination(retry):
                text = retry
            else:
                raise RuntimeError(
                    f"Whisper hallucinated: {text!r}. Hold MIC 2–3s and speak clearly into ★ Jabra."
                )
    return text


def _openai_whisper(audio_bytes: bytes, *, filename: str) -> str:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    model = OPENAI_WHISPER_MODEL

    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(f"\r\n--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n')
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode())
    except HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI Whisper HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"OpenAI Whisper network error: {e}") from e

    text = (payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("Whisper returned empty transcript.")
    return text
