FROM docker.io/runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip cache purge && rm -rf /root/.cache/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install whisperx + pyannote (this may pull newer torchaudio - that's OK for now)
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<4.36" \
    whisperx \
    pyannote.audio \
    runpod

# CRITICAL: Re-pin torchaudio to match the base image's torch 2.1.0+cu118
# pyannote may have upgraded it, but our CUDA 11.8 runtime can't use CUDA 12.x builds
RUN pip install --no-cache-dir \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall \
    && pip cache purge

# Pre-cache models
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null || true
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null || true

COPY src/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
