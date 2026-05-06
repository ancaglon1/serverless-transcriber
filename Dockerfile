FROM docker.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other deps
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<4.36" \
    whisperx \
    pyannote.audio \
    runpod \
    && pip cache purge

# Pre-download whisper model (small, fast)
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null || true
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null || true

COPY src/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
