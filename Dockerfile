FROM docker.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Don't pin torch to a specific version or index — let pip resolve from PyPI.
# PyPI's default torch comes bundled with CUDA 12.1 compat libs which work on CUDA 12.3.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<5" \
    whisperx \
    pyannote.audio \
    runpod \
    && pip cache purge

# Pre-cache base.en model and alignment model
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null || true
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null || true

COPY src/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
