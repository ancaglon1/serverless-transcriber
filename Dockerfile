FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<4.36" \
    whisperx \
    pyannote.audio \
    runpod

# Pre-download whisper base.en model
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null || true

# Pre-download alignment model
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null || true

COPY src/handler.py /handler.py
RUN chmod +x /handler.py

CMD ["python3", "-u", "/handler.py"]
