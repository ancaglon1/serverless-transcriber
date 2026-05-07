FROM docker.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install torch for CUDA 12.1 (compatible with 12.3 runtime)
RUN pip install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install whisperx + pyannote — torch/torchaudio already at versions they need
# No pins required since CUDA versions now match
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

# Verify imports at build time
RUN python3 -c "import whisperx; import pyannote.audio; import runpod; print('All imports OK')"

CMD ["python3", "-u", "/handler.py"]
