# Serverless Transcriber

RunPod serverless endpoint for GPU-accelerated audio transcription with speaker diarization.

**Models used:**
- WhisperX (whisper base.en) — transcription
- PyAnnote (speaker-diarization-3.1) — speaker labels

## API

### POST /run

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "model": "base.en",
    "diarize": true
  }
}
```

**Parameters:**
- `audio_url` — direct URL to audio file (mp3, wav, m4a, etc.)
- `audio_data` — base64-encoded audio (alternative to URL)
- `model` — whisper model: `tiny.en`, `base.en`, `small.en`, `medium`, `large-v3`
- `diarize` — boolean, enable speaker identification

**Response:**
```json
{
  "transcript": "[00:00] [SPEAKER_00] Hello world",
  "segments": [
    {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "Hello world"}
  ],
  "language": "en",
  "model": "base.en",
  "duration_seconds": 42.0
}
```
