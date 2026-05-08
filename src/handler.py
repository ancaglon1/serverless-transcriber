import os
import json
import tempfile
import urllib.request
import base64
import warnings
import runpod

warnings.filterwarnings("ignore")


def handler(job):
    job_input = job["input"]
    audio_url = job_input.get("audio_url")
    audio_data = job_input.get("audio_data")
    model_name = job_input.get("model", "base.en")
    do_diarize = job_input.get("diarize", True)

    if not audio_url and not audio_data:
        return {"error": "Provide audio_url or audio_data"}

    audio_path = None
    try:
        if audio_url:
            ext = ".mp3"
            for e in [".wav", ".mp3", ".m4a", ".mp4", ".ogg"]:
                if e in audio_url.split("?")[0]:
                    ext = e
                    break
            audio_path = tempfile.mktemp(suffix=ext)
            urllib.request.urlretrieve(audio_url, audio_path)
        else:
            audio_path = tempfile.mktemp(suffix=".wav")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_data))

        import whisperx
        audio = whisperx.load_audio(audio_path)

        model = whisperx.load_model(
            model_name, "cuda", compute_type="float16",
            asr_options={"max_new_tokens": 128}
        )
        result = model.transcribe(audio, batch_size=8)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device="cuda"
        )
        result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda")

        if do_diarize:
            diarize_model = whisperx.DiarizationPipeline(device="cuda")
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=6)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "speaker": seg.get("speaker"),
                "text": seg["text"].strip(),
            })

        transcript_lines = []
        for s in segments:
            ts = f"{int(s['start']//60):02d}:{int(s['start']%60):02d}"
            sp = f" [{s['speaker']}]" if s["speaker"] else ""
            transcript_lines.append(f"[{ts}]{sp} {s['text']}")

        return {
            "transcript": "\n".join(transcript_lines),
            "segments": segments,
            "language": result["language"],
            "model": model_name,
            "duration_seconds": segments[-1]["end"] if segments else 0,
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
